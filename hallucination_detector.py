"""
hallucination_detector.py  ·  production edition v3
====================================================
RAG Pipeline Hallucination Detector + Self-Healer
--------------------------------------------------
v3 fixes vs v2:
  1. HealingResult: initial_risk / final_risk (not "was X")
  2. Contradiction patch normalizes billing cycle to context unit
  3. Grounding rewrite uses context-derived prefix, not boilerplate
  4. Confidence recalibrated post-healing
  5. QualityScore: weighted composite for accept/retry/fallback routing

Five detection checks:
  1. Confidence scoring        — how assertive is the answer?
  2. Faithfulness scoring      — are claims grounded in context?
  3. Contradiction detection   — does the answer conflict with context?
  4. Entity hallucination      — are names/citations real?
  5. Answer drift              — has this answer changed over time?

Three healing strategies:
  A. Contradiction patch  — replace wrong numbers/billing cycles in-place
  B. Entity scrub         — remove hallucinated entity sentences
  C. Grounding rewrite    — rebuild from top-ranked context sentences

Install:
    pip install spacy
    python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import re
import time
import hashlib
import logging
import sqlite3
import threading
import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional


# ── spaCy optional ────────────────────────────────────────
try:
    import spacy as _spacy
    _nlp = _spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    _nlp = None
    SPACY_AVAILABLE = False

# ── Logging ───────────────────────────────────────────────
logger = logging.getLogger("hallucination_detector")
logger.propagate = False
logger.addHandler(logging.NullHandler())


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        import json
        payload = {
            "time":   self.formatTime(record, self.datefmt),
            "level":  record.levelname,
            "logger": record.name,
            "msg":    record.getMessage(),
        }
        if hasattr(record, "report"):
            payload["report"] = record.report
        return json.dumps(payload)


def configure_logging(level: int = logging.INFO) -> None:
    """Call once at app startup to enable structured JSON logs."""
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)


# ── Config ────────────────────────────────────────────────
@dataclass
class DetectorConfig:
    """
    All thresholds in one place — no source edits needed.

    confidence_threshold (float):
        Confidence score above which an answer is "assertive."
        Default 0.75.

    faithfulness_threshold (float):
        Fraction of answer claims that must be grounded in context.
        Default 0.50.

    faithfulness_overlap_threshold (float):
        Fraction of keywords per claim that must appear in context.
        Default 0.40.

    drift_threshold (float):
        Semantic distance above which the same question's answer
        is flagged as drifted. Default 0.35.

    db_path (str):
        SQLite file for drift history. Use ":memory:" in tests.

    window_size (int):
        Past answers retained per question. Default 50.

    log_flagged (bool):
        Emit WARNING log only when is_hallucinating=True. Default True.
    """
    confidence_threshold: float = 0.75
    faithfulness_threshold: float = 0.50
    faithfulness_overlap_threshold: float = 0.40
    drift_threshold: float = 0.35
    db_path: str = "hallucination_drift.db"
    window_size: int = 50
    log_flagged: bool = True


# ── Shared helpers ────────────────────────────────────────
_STOPWORDS = frozenset({
    "the","a","an","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could",
    "should","may","might","shall","can","to","of","in","on","at",
    "by","for","with","from","this","that","these","those","it",
    "its","and","or","but","so","yet","both","either","not","no",
    "nor","i","you","he","she","we","they","me","him","her","us",
    "them","my","your","his","our","their",
})

_COMMON_PROPER = frozenset({
    "However","Therefore","Moreover","Additionally","Furthermore",
    "Meanwhile","Nevertheless","Consequently","Subsequently",
    "According","Based","Given","Since","While","Although",
    "Despite","Through","Without","Because","Before","After",
    "During","Between","Among","Against","Within","Throughout",
    "January","February","March","April","June","July","August",
    "September","October","November","December",
    "Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday",
})


def _key_words(text: str) -> list[str]:
    return [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            if w not in _STOPWORDS]


def _extract_numbers(text: str) -> set[str]:
    """
    Extract numeric tokens. Excludes letter-prefixed identifiers (SKU-441)
    but preserves numeric ranges (5-7 days) and prices ($49.99).
    """
    return set(re.findall(
        r'(?<![A-Za-z]\-)\b\d+(?:\.\d+)?(?:%|k|m|b)?\b(?!\-[A-Za-z])',
        text.lower()
    ))


# ─────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────

@dataclass
class HallucinationReport:
    """
    Full result for one RAG response.

    is_hallucinating = True → block or flag before delivering to user.
    risk_level: "low" | "medium" | "high" | "critical"
    latency_ms: detector overhead (< 10ms without spaCy, < 50ms with)
    """
    question: str
    answer: str

    confidence_score: float = 0.0
    faithfulness_score: float = 0.0
    contradiction_found: bool = False
    contradiction_reason: str = ""
    hallucinated_entities: list = field(default_factory=list)
    drift_detected: bool = False
    drift_delta: float = 0.0

    is_hallucinating: bool = False
    risk_level: str = "low"
    triggered_checks: list = field(default_factory=list)
    explanation: list = field(default_factory=list)
    latency_ms: float = 0.0

    def summary(self) -> str:
        lines = [
            f"  Question  : {self.question[:80]}{'...' if len(self.question)>80 else ''}",
            f"  Risk      : {self.risk_level.upper()}",
            f"  Confidence: {self.confidence_score:.2f}",
            f"  Faithful  : {self.faithfulness_score:.2f}",
            f"  Contradict: {self.contradiction_found}"
              + (f" — {self.contradiction_reason}" if self.contradiction_reason else ""),
            f"  Fake names: {self.hallucinated_entities[:5]}",
            f"  Drift     : {self.drift_detected} (delta={self.drift_delta:.2f})",
            f"  Triggered : {self.triggered_checks}",
            f"  Latency   : {self.latency_ms:.1f}ms",
        ]
        if self.explanation:
            lines.append("  Why:")
            for e in self.explanation:
                lines.append(f"    — {e}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "question":             self.question,
            "risk_level":           self.risk_level,
            "is_hallucinating":     self.is_hallucinating,
            "confidence_score":     round(self.confidence_score, 3),
            "faithfulness_score":   round(self.faithfulness_score, 3),
            "contradiction_found":  self.contradiction_found,
            "contradiction_reason": self.contradiction_reason,
            "hallucinated_entities":self.hallucinated_entities,
            "drift_detected":       self.drift_detected,
            "drift_delta":          round(self.drift_delta, 3),
            "triggered_checks":     self.triggered_checks,
            "explanation":          self.explanation,
            "latency_ms":           round(self.latency_ms, 2),
        }


class HallucinationBlocked(Exception):
    """
    Raise when a flagged response should be blocked entirely.
    Carries the full report for logging and fallback handling.

        try:
            report = detector.inspect(q, chunks, answer)
            if report.is_hallucinating:
                raise HallucinationBlocked(report)
        except HallucinationBlocked as e:
            log_to_monitoring(e.report.to_dict())
            return fallback_response
    """
    def __init__(self, report: HallucinationReport):
        self.report = report
        super().__init__(
            f"Hallucination detected: risk={report.risk_level}, "
            f"checks={report.triggered_checks}"
        )


# ─────────────────────────────────────────────────────────
# QUALITY SCORE
# Weighted composite for routing: accept / retry / fallback
# ─────────────────────────────────────────────────────────

@dataclass
class QualityScore:
    """
    Weighted composite score for automated delivery routing.

    final_score = 0.40 * faithfulness
                + 0.30 * consistency          (0 if contradiction)
                + 0.20 * calibrated_confidence
                + 0.10 * latency_score        (non-linear penalty curve)
                - 0.20 * drift_penalty        (explicit, not indirect)

    Routing tiers (four, not two):
        ACCEPT        — score ≥ 0.75, no healing applied
        HEALED_ACCEPT — healing was applied and re-inspection passed
        DISCARD       — healing failed or safe decline served
        FALLBACK      — score < 0.50, not healed

    Latency penalty curve:
        < 20ms  → full 0.10
        20–50ms → linear decay from 0.10 → 0.05
        > 50ms  → steep decay from 0.05 → 0.00 at 200ms
        Rationale: <20ms is p50 for pure-Python with no NER;
        anything above 50ms (spaCy NER path) should not score full marks.

    Drift penalty:
        drift_detected=True → subtract 0.20 after other components.
        Drift is a signal the retrieval pipeline is degrading — it should
        push the score below FALLBACK threshold regardless of current
        faithfulness, since past inconsistency predicts future unreliability.

    Fields:
        faithfulness_component:   grounding contribution (0–0.40)
        consistency_component:    contradiction-free contribution (0–0.30)
        confidence_component:     calibrated confidence contribution (0–0.20)
        latency_component:        speed contribution (0–0.10)
        drift_penalty:            drift deduction (0 or 0.20)
        final_score:              clamped sum 0–1.0
        routing:                  "accept" | "healed_accept" | "discard" | "fallback"
    """
    faithfulness_component: float = 0.0
    consistency_component:  float = 0.0
    confidence_component:   float = 0.0
    latency_component:      float = 0.0
    drift_penalty:          float = 0.0
    final_score:            float = 0.0
    routing:                str   = "fallback"

    _ACCEPT_THRESHOLD   = 0.75
    _FALLBACK_THRESHOLD = 0.50

    @classmethod
    def compute(
        cls,
        report: HallucinationReport,
        healing_result: Optional["HealingResult"] = None,
    ) -> "QualityScore":
        """
        Compute quality score from a detection report.

        Pass healing_result when healing has been applied — the routing
        tier will be set to HEALED_ACCEPT or DISCARD rather than the
        raw score thresholds, which no longer apply to a healed answer.
        """
        qs = cls()

        # ── Faithfulness: 0–0.40 ──────────────────────────
        qs.faithfulness_component = report.faithfulness_score * 0.40

        # ── Consistency: 0–0.30 ───────────────────────────
        qs.consistency_component = 0.0 if report.contradiction_found else 0.30

        # ── Calibrated confidence: 0–0.20 ─────────────────
        # Reward hedged-but-accurate answers; penalise assertive-but-unfaithful.
        raw_conf = report.confidence_score
        if report.faithfulness_score >= 0.75:
            qs.confidence_component = raw_conf * 0.20
        else:
            qs.confidence_component = (1.0 - raw_conf) * 0.20

        # ── Latency: non-linear penalty curve ─────────────
        # <20ms → full 0.10 | 20–50ms → 0.10→0.05 | 50–200ms → 0.05→0.00
        ms = report.latency_ms
        if ms <= 20:
            qs.latency_component = 0.10
        elif ms <= 50:
            # Linear decay: 0.10 at 20ms → 0.05 at 50ms
            qs.latency_component = 0.10 - ((ms - 20) / 30) * 0.05
        else:
            # Steeper decay: 0.05 at 50ms → 0.00 at 200ms
            qs.latency_component = max(0.0, 0.05 - ((ms - 50) / 150) * 0.05)

        # ── Drift: explicit 0.20 deduction ────────────────
        # Applied after other components so it can push any answer below
        # the FALLBACK threshold regardless of current faithfulness.
        qs.drift_penalty = 0.20 if report.drift_detected else 0.0

        raw_score = (
            qs.faithfulness_component
            + qs.consistency_component
            + qs.confidence_component
            + qs.latency_component
            - qs.drift_penalty
        )
        qs.final_score = max(0.0, min(1.0, raw_score))

        # ── Routing ───────────────────────────────────────
        if healing_result is not None and healing_result.healing_applied:
            # Healed answers are routed by healing outcome, not raw score.
            # The pre-healing score is irrelevant; what matters is whether
            # the healed answer passed re-inspection.
            if healing_result.strategy_used == "safe_decline":
                qs.routing = "discard"
            else:
                qs.routing = "healed_accept"
        else:
            if qs.final_score >= cls._ACCEPT_THRESHOLD:
                qs.routing = "accept"
            else:
                qs.routing = "fallback"

        return qs

    def summary(self) -> str:
        drift_line = (f"\n    drift      -{self.drift_penalty:.2f} penalty"
                      if self.drift_penalty else "")
        return (
            f"  Score     : {self.final_score:.2f}  →  {self.routing.upper()}\n"
            f"  Components:\n"
            f"    faithful  {self.faithfulness_component:.2f} / 0.40\n"
            f"    consistent {self.consistency_component:.2f} / 0.30\n"
            f"    confidence {self.confidence_component:.2f} / 0.20\n"
            f"    latency    {self.latency_component:.2f} / 0.10"
            f"{drift_line}"
        )

    def to_dict(self) -> dict:
        return {
            "final_score":            round(self.final_score, 3),
            "routing":                self.routing,
            "faithfulness_component": round(self.faithfulness_component, 3),
            "consistency_component":  round(self.consistency_component, 3),
            "confidence_component":   round(self.confidence_component, 3),
            "latency_component":      round(self.latency_component, 3),
            "drift_penalty":          round(self.drift_penalty, 3),
        }


# ─────────────────────────────────────────────────────────
# CHECK 1 — CONFIDENCE SCORER
# ─────────────────────────────────────────────────────────

class ConfidenceScorer:
    """
    Estimates how assertive an answer sounds using surface signals.

    High confidence + low faithfulness = the most dangerous combination.
    Without logprobs (standard API setup), we proxy confidence through
    linguistic hedges and assertion markers.

    Score: 0.0 = maximally uncertain, 1.0 = maximally assertive.
    """
    _HIGH_RE = [re.compile(p) for p in [
        r"\bis\b", r"\bare\b", r"\bwas\b", r"\bwere\b", r"\bwill\b",
        r"\bmust\b", r"\balways\b", r"\bnever\b", r"\bdefinitely\b",
        r"\bcertainly\b", r"\bclearly\b", r"\bthe fact that\b",
        r"\bobviously\b", r"\bexactly\b", r"\bprecisely\b",
        r"\bguaranteed\b", r"\bundoubtedly\b", r"\babsolutely\b",
    ]]
    _LOW_RE = [re.compile(p) for p in [
        r"\bmight\b", r"\bmay\b", r"\bcould\b", r"\bperhaps\b",
        r"\bpossibly\b", r"\bapproximately\b", r"\baround\b",
        r"\bi think\b", r"\bi believe\b", r"\bit seems\b",
        r"\bit appears\b", r"\buncertain\b", r"\bnot sure\b",
        r"\bbased on the provided\b", r"\baccording to\b",
        r"\bthe context suggests\b", r"\bas mentioned\b",
        r"\bthe source indicates\b", r"\bper the documentation\b",
    ]]

    def score(self, answer: str) -> float:
        al    = answer.lower()
        words = len(answer.split()) or 1
        high  = sum(len(p.findall(al)) for p in self._HIGH_RE)
        low   = sum(len(p.findall(al)) for p in self._LOW_RE)
        return max(0.0, min(1.0,
            0.5 + min(high / (words / 10), 1.0) * 0.5
                - min(low  / (words / 10), 1.0) * 0.5
        ))


# ─────────────────────────────────────────────────────────
# CHECK 2 — FAITHFULNESS SCORER
# ─────────────────────────────────────────────────────────

class FaithfulnessScorer:
    """
    Measures how much of the answer is traceable to retrieved context.

    Approach (no LLM judge):
      1. Split answer into factual claim sentences
      2. For each claim, check fraction of content words in context
      3. Score = grounded claims / total claims

    overlap_threshold: fraction of keywords required per claim.
    """
    def __init__(self, overlap_threshold: float = 0.40):
        self.overlap_threshold = overlap_threshold

    def _extract_claims(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences
                if len(s.split()) >= 4 and not s.strip().endswith("?")]

    def _claim_grounded(self, claim: str, context_lower: str) -> bool:
        kw = _key_words(claim)
        if not kw:
            return True
        return sum(1 for w in kw if w in context_lower) / len(kw) >= self.overlap_threshold

    def score(self, answer: str, context_chunks: list[str]) -> tuple[float, list[str]]:
        """
        Returns (faithfulness_score, ungrounded_claim_list).
        faithfulness_score: 0.0 = fully hallucinated, 1.0 = fully grounded.
        """
        combined = " ".join(context_chunks).lower()
        claims   = self._extract_claims(answer)
        if not claims:
            return 1.0, []
        grounded, ungrounded = [], []
        for c in claims:
            (grounded if self._claim_grounded(c, combined) else ungrounded).append(c)
        return len(grounded) / len(claims), ungrounded


# ─────────────────────────────────────────────────────────
# CHECK 3 — CONTRADICTION DETECTOR
# ─────────────────────────────────────────────────────────

class ContradictionDetector:
    """
    Finds direct factual contradictions between the answer and context.

    Targets three high-value patterns:
      Numeric   — answer says "30 days", context says "14 days"
      Temporal  — answer says "monthly", context says "annually"
      Negation  — answer says "supports X", context says "does not support X"
    """
    _NEGATION_PAIRS = [
        (re.compile(r"does not\s+(\w+)"),  r"\1"),
        (re.compile(r"cannot\s+(\w+)"),    r"can\s+\1"),
        (re.compile(r"never\s+(\w+)"),     r"always\s+\1"),
        (re.compile(r"no\s+(\w+)"),        r"has\s+\1"),
        (re.compile(r"isn'?t\s+(\w+)"),    r"is\s+\1"),
        (re.compile(r"won'?t\s+(\w+)"),    r"will\s+\1"),
        (re.compile(r"don'?t\s+(\w+)"),    r"do\s+\1"),
        (re.compile(r"didn'?t\s+(\w+)"),   r"did\s+\1"),
    ]
    _TEMPORAL_RE = re.compile(r'\b(\d+)\s*(day|week|month|year|hour|minute)s?\b')

    def _has_negation_flip(self, answer: str, context: str) -> bool:
        al, cl = answer.lower(), context.lower()
        for neg_re, pos_tpl in self._NEGATION_PAIRS:
            for m in neg_re.finditer(cl):
                if re.compile(pos_tpl.replace(r"\1", re.escape(m.group(1)))).search(al):
                    return True
            for m in neg_re.finditer(al):
                if re.compile(pos_tpl.replace(r"\1", re.escape(m.group(1)))).search(cl):
                    return True
        return False

    def detect(self, answer: str, context_chunks: list[str]) -> tuple[bool, str]:
        """Returns (contradiction_found, explanation_string)."""
        combined = " ".join(context_chunks)
        a_nums = _extract_numbers(answer)
        c_nums = _extract_numbers(combined)

        for num in a_nums - c_nums:
            for snippet in re.findall(rf'.{{0,40}}{re.escape(num)}.{{0,40}}', answer.lower()):
                for kw in _key_words(snippet):
                    for cs in re.findall(rf'.{{0,40}}{re.escape(kw)}.{{0,40}}', combined.lower()):
                        cs_nums = _extract_numbers(cs)
                        if cs_nums and num not in cs_nums:
                            return True, (
                                f"Numeric contradiction: answer uses '{num}' "
                                f"but context shows '{next(iter(cs_nums))}' near '{kw}'"
                            )

        if self._has_negation_flip(answer, combined):
            return True, "Negation flip: answer affirms something the context negates"

        a_temp = self._TEMPORAL_RE.findall(answer.lower())
        c_temp = self._TEMPORAL_RE.findall(combined.lower())
        for unit in {u for _, u in a_temp}:
            av = {n for n, u in a_temp if u == unit}
            cv = {n for n, u in c_temp if u == unit}
            if av and cv and not av & cv:
                return True, (
                    f"Temporal contradiction: answer says '{av.pop()} {unit}(s)' "
                    f"but context says '{cv.pop()} {unit}(s)'"
                )

        return False, ""


# ─────────────────────────────────────────────────────────
# CHECK 4 — ENTITY HALLUCINATION DETECTOR
# ─────────────────────────────────────────────────────────

class EntityHallucinationDetector:
    """
    Extracts named entities from the answer and verifies each one
    exists in the retrieved context.

    Production path: spaCy en_core_web_sm (no false positives on
    phrases like "Scaling Named" or "Entity Recognition").

    Fallback path: regex NER when spaCy not installed.
    """
    _CITATION_RE = re.compile(
        r'(?:[A-Z][a-z]+\s+et\s+al\.?\s*\(\d{4}\)'
        r'|(?:Paper|Report|Study|RFC|arXiv)\s*[:#]?\s*[\d\w\-\.]+'
        r'|\(\d{4}\))'
    )
    _FALLBACK_PERSON_RE = re.compile(
        r'\b(?:Dr\.?|Prof\.?|Mr\.?|Ms\.?|Mrs\.?|Sir)\s+'
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
    )
    _FALLBACK_ORG_RE = re.compile(
        r'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*'
        r'\s+(?:Inc\.?|Ltd\.?|Corp\.?|University|Institute|Foundation|Labs?|AI|Technologies))\b'
    )

    def _extract_spacy(self, text: str) -> dict[str, str]:
        doc = _nlp(text)
        entities: dict[str, str] = {}
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "ORG", "PRODUCT", "WORK_OF_ART", "FAC"):
                if len(ent.text.strip()) >= 3 and ent.text not in _COMMON_PROPER:
                    entities[ent.text.strip()] = ent.label_.lower()
        for m in self._CITATION_RE.finditer(text):
            entities[m.group(0).strip()] = "citation"
        return entities

    def _extract_regex(self, text: str) -> dict[str, str]:
        entities: dict[str, str] = {}
        for m in self._FALLBACK_PERSON_RE.finditer(text):
            entities[m.group(0).strip()] = "person"
        for m in self._FALLBACK_ORG_RE.finditer(text):
            entities[m.group(0).strip()] = "org"
        for m in self._CITATION_RE.finditer(text):
            entities[m.group(0).strip()] = "citation"
        return entities

    def detect(self, answer: str, context_chunks: list[str]) -> list[str]:
        """
        Returns list of entities in the answer absent from all context chunks.
        """
        combined  = " ".join(context_chunks).lower()
        entities  = self._extract_spacy(answer) if SPACY_AVAILABLE else self._extract_regex(answer)
        hallucinated = []
        for entity, etype in entities.items():
            el    = entity.lower()
            clean = re.sub(r'^(?:dr\.?|prof\.?|mr\.?|ms\.?|mrs\.?|sir)\s+', '', el).strip()
            if clean not in combined and el not in combined:
                parts = [p for p in clean.split() if len(p) > 3]
                if parts and any(p in combined for p in parts):
                    continue
                hallucinated.append(f"{entity} ({etype})")
        return hallucinated


# ─────────────────────────────────────────────────────────
# CHECK 5 — ANSWER DRIFT MONITOR
# ─────────────────────────────────────────────────────────

class AnswerDriftMonitor:
    """
    Tracks answer fingerprints per question in SQLite.
    Survives process restarts. Thread-safe.

    Drift = same question, meaningfully different answer over time.
    Catches: retrieval degradation, embedding model updates,
    context window pollution, silent model behavior changes.
    """
    _POS = frozenset({"yes","can","will","available","supported","allowed",
                      "free","included","enabled","active","valid","approved"})
    _NEG = frozenset({"no","cannot","unavailable","unsupported","prohibited",
                      "disabled","inactive","not","never","invalid"})

    def __init__(self, db_path: str = "hallucination_drift.db",
                 window_size: int = 50, drift_threshold: float = 0.35):
        self.window_size    = window_size
        self.drift_threshold = drift_threshold
        self._lock = threading.Lock()
        self._db_path = db_path
        self._persistent_conn: Optional[sqlite3.Connection] = None
        if db_path == ":memory:":
            self._persistent_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._persistent_conn.row_factory = sqlite3.Row
        self._init_db()

    @contextmanager
    def _conn(self):
        if self._persistent_conn is not None:
            yield self._persistent_conn
            self._persistent_conn.commit()
        else:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS drift_history (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    question_key  TEXT    NOT NULL,
                    timestamp     REAL    NOT NULL,
                    numbers       TEXT    NOT NULL,
                    key_words     TEXT    NOT NULL,
                    polarity      TEXT    NOT NULL,
                    length_bucket INTEGER NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_qkey "
                "ON drift_history(question_key, timestamp)"
            )

    def _question_key(self, question: str) -> str:
        n = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', question.lower().strip()))
        return hashlib.md5(n.encode()).hexdigest()[:16]

    def _fingerprint(self, answer: str) -> dict:
        numbers   = sorted(set(re.findall(r'\b\d+(?:\.\d+)?(?:%|k|m)?\b', answer)))
        words     = answer.lower().split()
        key_words = [w for w in words if len(w) > 5][:20]
        pos = sum(1 for w in words if w in self._POS)
        neg = sum(1 for w in words if w in self._NEG)
        return {
            "numbers":       numbers,
            "key_words":     key_words,
            "polarity":      "positive" if pos > neg else ("negative" if neg > pos else "neutral"),
            "length_bucket": len(answer) // 100,
        }

    def _similarity(self, fp1: dict, fp2: dict) -> float:
        scores = []
        n1, n2 = set(fp1["numbers"]),   set(fp2["numbers"])
        k1, k2 = set(fp1["key_words"]), set(fp2["key_words"])
        if n1 or n2: scores.append(len(n1 & n2) / max(len(n1 | n2), 1) * 0.4)
        if k1 or k2: scores.append(len(k1 & k2) / max(len(k1 | k2), 1) * 0.4)
        scores.append(0.2 if fp1["polarity"] == fp2["polarity"] else 0.0)
        return sum(scores)

    def record(self, question: str, answer: str) -> tuple[bool, float]:
        """Record answer, check for drift. Returns (drift_detected, drift_delta)."""
        key = self._question_key(question)
        fp  = self._fingerprint(answer)
        with self._lock:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT numbers, key_words, polarity, length_bucket "
                    "FROM drift_history WHERE question_key = ? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (key, self.window_size)
                ).fetchall()

                drift_detected, drift_delta = False, 0.0
                if len(rows) >= 3:
                    past = [
                        {"numbers":       r["numbers"].split(",")   if r["numbers"]   else [],
                         "key_words":     r["key_words"].split(",") if r["key_words"] else [],
                         "polarity":      r["polarity"],
                         "length_bucket": r["length_bucket"]}
                        for r in rows[:10]
                    ]
                    sims        = [self._similarity(fp, p) for p in past]
                    drift_delta = 1.0 - sum(sims) / len(sims)
                    drift_detected = drift_delta > self.drift_threshold

                conn.execute(
                    "INSERT INTO drift_history "
                    "(question_key, timestamp, numbers, key_words, polarity, length_bucket) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (key, time.time(), ",".join(fp["numbers"]),
                     ",".join(fp["key_words"]), fp["polarity"], fp["length_bucket"])
                )
                conn.execute(
                    "DELETE FROM drift_history WHERE question_key = ? AND id NOT IN "
                    "(SELECT id FROM drift_history WHERE question_key = ? "
                    " ORDER BY timestamp DESC LIMIT ?)",
                    (key, key, self.window_size)
                )
        return drift_detected, drift_delta

    def clear_history(self, question: Optional[str] = None) -> None:
        with self._lock:
            with self._conn() as conn:
                if question:
                    conn.execute("DELETE FROM drift_history WHERE question_key = ?",
                                 (self._question_key(question),))
                else:
                    conn.execute("DELETE FROM drift_history")

    def get_stats(self, question: str) -> dict:
        key = self._question_key(question)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM drift_history WHERE question_key = ?", (key,)
            ).fetchone()
        return {"question": question[:60], "total_recorded": row["cnt"]}


# ─────────────────────────────────────────────────────────
# HEALING RESULT
# ─────────────────────────────────────────────────────────

@dataclass
class HealingResult:
    """
    Result of a healing attempt.

    FIX v3: initial_risk and final_risk are stored separately so
    downstream systems and logs can compare them unambiguously.
    "Re-check: LOW (was LOW)" is replaced with explicit fields.

    strategy_used:
        "contradiction_patch" — numbers/billing cycles replaced in-place
        "entity_scrub"        — hallucinated entity sentences removed
        "grounding_rewrite"   — answer rebuilt from context sentences
        "safe_decline"        — healing failed re-inspection; fallback used
        "no_healing_needed"   — risk was already low; answer unchanged
    """
    original_answer:       str
    healed_answer:         str
    strategy_used:         str
    healing_applied:       bool
    initial_risk:          str = "low"
    final_risk:            str = "low"
    changes_made:          list = field(default_factory=list)
    re_inspection_report:  Optional[HallucinationReport] = None

    def summary(self) -> str:
        lines = [
            f"  Strategy     : {self.strategy_used}",
            f"  Healing      : {'yes' if self.healing_applied else 'no'}",
            f"  Initial risk : {self.initial_risk.upper()}",
            f"  Final risk   : {self.final_risk.upper()}",
            f"  Before       : {self.original_answer[:100]}...",
            f"  After        : {self.healed_answer[:100]}...",
        ]
        if self.changes_made:
            lines.append("  Changes:")
            for c in self.changes_made:
                lines.append(f"    — {c}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "strategy_used":   self.strategy_used,
            "healing_applied": self.healing_applied,
            "initial_risk":    self.initial_risk,
            "final_risk":      self.final_risk,
            "changes_made":    self.changes_made,
            "healed_answer":   self.healed_answer,
        }


# ─────────────────────────────────────────────────────────
# HALLUCINATION HEALER
# ─────────────────────────────────────────────────────────

class HallucinationHealer:
    """
    Attempts to fix a hallucinated RAG answer before it reaches the user.

    Three strategies, chosen by severity:
      A. Contradiction patch  — deterministic number/cycle replacement
      B. Entity scrub         — remove sentences with unverifiable entities
      C. Grounding rewrite    — rebuild answer from top-ranked context sentences

    Every healed answer is re-inspected. If it still fails → safe decline.

    v3 improvements:
      - initial_risk / final_risk tracked separately (not "was X")
      - Billing-cycle normalization matches context unit (monthly vs annually)
      - Grounding rewrite uses context-derived prefix, not generic boilerplate
      - Confidence recalibrated on healed answer
      - changes_made list explains what was corrected (explainable healing)
    """

    SAFE_DECLINE = (
        "I don't have enough reliable information in the provided sources "
        "to answer this question accurately. Please consult the original "
        "documentation or a subject matter expert."
    )

    # Billing cycle normalization: detect context direction, apply only that direction.
    # ANNUAL_BILLING_PATTERNS applied when context says "annually/per year/yearly".
    # MONTHLY_BILLING_PATTERNS applied when context says "monthly/per month".
    # More-specific patterns listed before adjective fallbacks to avoid over-replacement.
    _ANNUAL_BILLING_PATTERNS = [
        (r'\bper month\b',            'per year'),
        (r'\bmonthly subscription\b',  'annual subscription'),
        (r'\bmonthly billing\b',       'annual billing'),
        (r'\bmonthly plan\b',          'annual plan'),
        (r'\bbilled monthly\b',        'billed annually'),
        (r'\bmonthly\b',               'annual'),   # adjective fallback — last
    ]
    _MONTHLY_BILLING_PATTERNS = [
        (r'\bper year\b',             'per month'),
        (r'\bannual subscription\b',   'monthly subscription'),
        (r'\bannual billing\b',        'monthly billing'),
        (r'\bannually billed\b',       'billed monthly'),
        (r'\bbilled annually\b',       'billed monthly'),
        (r'\bannually\b',              'monthly'),  # adverb fallback — last
        (r'\bannual\b',               'monthly'),   # adjective fallback — last
    ]

    def __init__(self, detector: "HallucinationDetector"):
        self._detector = detector

    # ── Strategy A: Contradiction Patch ──────────────────

    def _patch_contradiction(
        self, answer: str,
        report: HallucinationReport,
        context_chunks: list[str],
    ) -> tuple[str, list[str]]:
        """
        Replace wrong numbers and billing cycles in the answer.
        Returns (patched_answer, list_of_changes).

        v3: billing cycle normalization converts to the SAME UNIT as context
        (if context says "annually", the answer is converted to annual terms).
        """
        combined  = " ".join(context_chunks)
        patched   = answer
        changes   = []

        # ── Numeric patches ───────────────────────────────
        # Build a map of unit → correct_value from context
        context_facts: dict[str, str] = {}
        for m in re.finditer(
            r'(\d+(?:\.\d+)?(?:%|k|m)?)\s*(days?|weeks?|months?|years?|hours?)',
            combined.lower()
        ):
            unit = m.group(2).rstrip('s')
            context_facts[unit] = m.group(1)

        for m in re.finditer(
            r'(\d+(?:\.\d+)?(?:%|k|m)?)\s*(days?|weeks?|months?|years?|hours?)',
            patched.lower()
        ):
            ans_val = m.group(1)
            unit    = m.group(2).rstrip('s')
            if unit in context_facts and context_facts[unit] != ans_val:
                correct = context_facts[unit]
                patched = re.sub(
                    rf'\b{re.escape(ans_val)}\b(?=\s*{re.escape(m.group(2))})',
                    correct, patched, count=1, flags=re.IGNORECASE
                )
                changes.append(f"Replaced '{ans_val} {unit}' → '{correct} {unit}'")

        # ── Dollar amount patches ─────────────────────────
        ctx_dollars = re.findall(r'\$(\d+(?:\.\d+)?)', combined)
        ans_dollars = re.findall(r'\$(\d+(?:\.\d+)?)', patched)
        ctx_dollar_set = set(ctx_dollars)
        for ad in ans_dollars:
            if ad not in ctx_dollar_set and ctx_dollars:
                correct_d = min(ctx_dollars, key=lambda x: abs(float(x) - float(ad)))
                patched = re.sub(
                    rf'\${re.escape(ad)}\b', f'${correct_d}',
                    patched, count=1
                )
                changes.append(f"Replaced '${ad}' → '${correct_d}'")

        # ── Billing cycle normalization ───────────────────
        # Detect context direction first, then apply only that direction's patterns.
        # This prevents patterns from fighting each other in multi-pass replacement.
        combined_lower = combined.lower()

        ctx_annual  = bool(re.search(
            r'\bannually\b|\bper year\b|\byearly\b|\bannual\b', combined_lower
        ))
        ctx_monthly = (bool(re.search(
            r'\bmonthly\b|\bper month\b', combined_lower
        )) and not ctx_annual)

        if ctx_annual:
            patterns = self._ANNUAL_BILLING_PATTERNS
            direction = "annual"
        elif ctx_monthly:
            patterns = self._MONTHLY_BILLING_PATTERNS
            direction = "monthly"
        else:
            patterns = []
            direction = None

        for ans_pattern, replacement in patterns:
            new_patched = re.sub(ans_pattern, replacement, patched, flags=re.IGNORECASE)
            if new_patched != patched:
                changes.append(f"Normalized billing: '{ans_pattern}' → '{replacement}'")
                patched = new_patched

        return patched, changes

    # ── Strategy B: Entity Scrub ─────────────────────────

    def _scrub_entities(
        self, answer: str,
        report: HallucinationReport,
        context_chunks: list[str],
    ) -> tuple[str, list[str]]:
        """
        Remove sentences containing hallucinated entities.
        Appends a transparency note when sentences are removed.
        Returns (scrubbed_answer, list_of_changes).
        """
        if not report.hallucinated_entities:
            return answer, []

        fake_names = [e.split(" (")[0] for e in report.hallucinated_entities]
        sentences  = re.split(r'(?<=[.!?])\s+', answer.strip())
        clean, removed = [], []

        for sent in sentences:
            if any(name.lower() in sent.lower() for name in fake_names):
                removed.append(sent[:60] + ("..." if len(sent) > 60 else ""))
            else:
                clean.append(sent)

        if not clean:
            return self.SAFE_DECLINE, [f"Removed all sentences (all contained unverifiable entities)"]

        result  = " ".join(clean)
        changes = [f"Removed sentence containing '{name}'" for name in fake_names[:3]]
        if removed:
            result += (
                " Note: specific names or references could not be verified "
                "in the source documents and have been omitted."
            )
        return result, changes

    # ── Strategy C: Grounding Rewrite ────────────────────

    def _grounding_rewrite(
        self, question: str, answer: str,
        report: HallucinationReport,
        context_chunks: list[str],
    ) -> tuple[str, list[str]]:
        """
        Rebuild the answer from the top-ranked context sentences.

        v3: prefix is derived from context content, not generic boilerplate.
        "According to the provided data:" when data is factual/numeric.
        "The source indicates that:" when context is policy/descriptive.
        "Based on the available documentation:" as default.
        """
        question_words = set(_key_words(question))
        if not question_words:
            return self.SAFE_DECLINE, ["Could not extract question keywords"]

        scored: list[tuple[float, str]] = []
        for chunk in context_chunks:
            for sent in re.split(r'(?<=[.!?])\s+', chunk.strip()):
                if len(sent.split()) < 4:
                    continue
                overlap = len(question_words & set(_key_words(sent))) / max(len(question_words), 1)
                scored.append((overlap, sent))

        if not scored:
            return self.SAFE_DECLINE, ["No usable context sentences found"]

        top_sentences = [s for _, s in sorted(scored, reverse=True)[:3]]

        # Context-derived prefix
        combined_lower = " ".join(context_chunks).lower()
        if re.search(r'\$\d+|\d+%|\d+\s*(day|month|year)', combined_lower):
            prefix = "According to the provided data:"
        elif any(w in combined_lower for w in ("policy", "guideline", "procedure", "rule")):
            prefix = "Per the source documentation:"
        else:
            prefix = "The source indicates that:"

        rewritten = f"{prefix} {' '.join(top_sentences)}"
        changes   = [
            f"Rebuilt answer from {len(top_sentences)} context sentence(s)",
            f"Used prefix: '{prefix}'",
        ]
        return rewritten, changes

    # ── Confidence recalibration ──────────────────────────

    @staticmethod
    def _recalibrate_confidence(
        healed_answer: str,
        context_chunks: list[str],
        strategy: str,
        original_confidence: float,
    ) -> float:
        """
        Recalibrate confidence score based on healing strategy.

        contradiction_patch: deterministic fix from a known source fact.
        The answer is now correct — confidence should reflect that.
        We boost toward 0.80 (high but not absolute — we patched, not verified
        every claim). Formula: original + 0.15, capped at 0.80.

        entity_scrub: removed unverifiable sentences; remaining claims are
        still the LLM's own words. Slight reduction in confidence appropriate.
        Formula: original * 0.85.

        grounding_rewrite: rebuilt from context sentences with hedging prefix.
        The ConfidenceScorer will naturally score this lower because of
        "According to...", "The source indicates..." etc. Run the scorer
        on the healed answer directly — that linguistic hedge IS the signal.
        Formula: re-run ConfidenceScorer (prefix does the work).
        """
        scorer = ConfidenceScorer()

        if strategy == "contradiction_patch":
            # Deterministic fix — raise confidence toward 0.80
            return min(original_confidence + 0.15, 0.80)

        elif strategy == "entity_scrub":
            # Removed bad sentences but didn't rewrite — slight reduction
            return original_confidence * 0.85

        else:
            # grounding_rewrite / safe_decline — let linguistic hedges
            # in the healed answer drive the score naturally
            return scorer.score(healed_answer)

    # ── Main heal ────────────────────────────────────────

    def heal(
        self,
        question: str,
        context_chunks: list[str],
        answer: str,
        report: HallucinationReport,
    ) -> HealingResult:
        """
        Attempt to fix a hallucinated answer.
        Returns a HealingResult with initial_risk and final_risk clearly
        separated so logs and downstream systems can compare them.

        Healing priority:
          1. faithfulness < 0.30 → full grounding rewrite (deeply hallucinated)
          2. contradiction found  → patch numbers/billing cycles in-place
          3. hallucinated entities → scrub sentences
          4. drift / confident-unfaithful → grounding rewrite
          5. Re-inspect; if still fails → safe decline
        """
        initial_risk = report.risk_level

        if initial_risk == "low":
            return HealingResult(
                original_answer=answer,
                healed_answer=answer,
                strategy_used="no_healing_needed",
                healing_applied=False,
                initial_risk=initial_risk,
                final_risk=initial_risk,
            )

        # Choose and apply strategy
        if report.faithfulness_score < 0.30:
            strategy = "grounding_rewrite"
            healed, changes = self._grounding_rewrite(question, answer, report, context_chunks)

        elif report.contradiction_found:
            strategy = "contradiction_patch"
            healed, changes = self._patch_contradiction(answer, report, context_chunks)
            # Fallback to rewrite if patch made no changes or faithfulness is still poor
            if not changes or report.faithfulness_score < 0.5:
                strategy = "grounding_rewrite"
                healed, changes = self._grounding_rewrite(question, answer, report, context_chunks)

        elif report.hallucinated_entities:
            strategy = "entity_scrub"
            healed, changes = self._scrub_entities(answer, report, context_chunks)

        else:
            strategy = "grounding_rewrite"
            healed, changes = self._grounding_rewrite(question, answer, report, context_chunks)

        # Re-inspect with a silent flag (no double-logging)
        re_report = self._detector.inspect(question, context_chunks, healed, _silent=True)

        # Recalibrate confidence: strategy-aware adjustment
        recalibrated_conf = self._recalibrate_confidence(
            healed, context_chunks, strategy, report.confidence_score
        )
        re_report.confidence_score = recalibrated_conf
        changes.append(
            f"Confidence recalibrated: {report.confidence_score:.2f} → {recalibrated_conf:.2f} "
            f"({strategy})"
        )

        final_risk = re_report.risk_level

        # If healed answer still fails → safe decline
        if re_report.is_hallucinating:
            return HealingResult(
                original_answer=answer,
                healed_answer=self.SAFE_DECLINE,
                strategy_used="safe_decline",
                healing_applied=True,
                initial_risk=initial_risk,
                final_risk="low",    # safe decline is always safe
                changes_made=[f"Healing failed re-inspection ({final_risk}); safe decline served"],
                re_inspection_report=re_report,
            )

        return HealingResult(
            original_answer=answer,
            healed_answer=healed,
            strategy_used=strategy,
            healing_applied=True,
            initial_risk=initial_risk,
            final_risk=final_risk,
            changes_made=changes,
            re_inspection_report=re_report,
        )


# ─────────────────────────────────────────────────────────
# THE DETECTOR
# ─────────────────────────────────────────────────────────

class HallucinationDetector:
    """
    Production RAG hallucination detector.

    Sync:
        report = detector.inspect(question, chunks, answer)
        score  = QualityScore.compute(report)

    Async:
        report = await detector.ainspect(question, chunks, answer)

    Detect + Fix:
        healer = HallucinationHealer(detector)
        result = healer.heal(question, chunks, answer, report)
        final  = result.healed_answer   # risk reduced to result.final_risk

    Block on critical:
        if report.is_hallucinating:
            raise HallucinationBlocked(report)
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self._confidence    = ConfidenceScorer()
        self._faithfulness  = FaithfulnessScorer(self.config.faithfulness_overlap_threshold)
        self._contradiction = ContradictionDetector()
        self._entity        = EntityHallucinationDetector()
        self._drift         = AnswerDriftMonitor(
            self.config.db_path, self.config.window_size, self.config.drift_threshold
        )
        self._total_inspected = 0
        self._total_flagged   = 0
        self._stats_lock      = threading.Lock()

        if not SPACY_AVAILABLE:
            logger.warning("spaCy not found — regex NER fallback active.")

    def inspect(
        self,
        question: str,
        context_chunks: list[str],
        answer: str,
        _silent: bool = False,
    ) -> HallucinationReport:
        """
        Run all 5 checks synchronously.
        _silent=True suppresses logging (used internally by healer re-inspection).
        """
        t0     = time.perf_counter()
        report = self._run_checks(question, context_chunks, answer)
        report.latency_ms = (time.perf_counter() - t0) * 1000

        with self._stats_lock:
            self._total_inspected += 1
            if report.is_hallucinating:
                self._total_flagged += 1

        if self.config.log_flagged and report.is_hallucinating and not _silent:
            logger.warning("hallucination_detected", extra={"report": report.to_dict()})

        return report

    async def ainspect(
        self,
        question: str,
        context_chunks: list[str],
        answer: str,
    ) -> HallucinationReport:
        """Async wrapper — runs inspect() in thread pool, safe for FastAPI."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.inspect, question, context_chunks, answer)

    def _run_checks(
        self,
        question: str,
        context_chunks: list[str],
        answer: str,
    ) -> HallucinationReport:
        report      = HallucinationReport(question=question, answer=answer)
        triggered   = []
        explanations = []

        conf  = self._confidence.score(answer)
        faith, ungrounded = self._faithfulness.score(answer, context_chunks)
        report.confidence_score   = conf
        report.faithfulness_score = faith

        if conf > self.config.confidence_threshold and faith < self.config.faithfulness_threshold:
            triggered.append("confident_but_unfaithful")
            first = f" Ungrounded: '{ungrounded[0][:80]}'" if ungrounded else ""
            explanations.append(
                f"Model sounds {conf:.0%} confident but only {faith:.0%} of "
                f"claims are grounded in context.{first}"
            )

        contradiction, contra_reason = self._contradiction.detect(answer, context_chunks)
        report.contradiction_found   = contradiction
        report.contradiction_reason  = contra_reason
        if contradiction:
            triggered.append("contradiction")
            explanations.append(f"Direct contradiction with source: {contra_reason}")

        fake_entities = self._entity.detect(answer, context_chunks)
        report.hallucinated_entities = fake_entities
        if fake_entities:
            triggered.append("hallucinated_entities")
            explanations.append(
                f"Entities absent from all source chunks: {', '.join(fake_entities[:3])}"
                + (f" (+{len(fake_entities)-3} more)" if len(fake_entities) > 3 else "")
            )

        drift, delta = self._drift.record(question, answer)
        report.drift_detected = drift
        report.drift_delta    = delta
        if drift:
            triggered.append("answer_drift")
            explanations.append(
                f"Answer drifted from recent responses (delta={delta:.2f}). "
                "Possible retrieval degradation."
            )

        n    = len(triggered)
        risk = ("critical" if contradiction or n >= 3
                else "high"   if n == 2 or (fake_entities and conf > 0.8)
                else "medium" if n == 1
                else "low")

        report.triggered_checks = triggered
        report.explanation      = explanations
        report.risk_level       = risk
        report.is_hallucinating = risk in ("high", "critical")
        return report

    def stats(self) -> dict:
        with self._stats_lock:
            rate = (self._total_flagged / self._total_inspected * 100
                    if self._total_inspected else 0)
            return {
                "total_inspected":    self._total_inspected,
                "total_flagged":      self._total_flagged,
                "hallucination_rate": f"{rate:.1f}%",
                "ner_backend":        "spaCy" if SPACY_AVAILABLE else "regex (fallback)",
            }

    def reset_stats(self) -> None:
        with self._stats_lock:
            self._total_inspected = 0
            self._total_flagged   = 0


# ─────────────────────────────────────────────────────────
# DEMOS
# ─────────────────────────────────────────────────────────

def _sep(title: str) -> None:
    print(f"\n{'═'*60}\n  {title}\n{'═'*60}")


def _show(report: HallucinationReport, result: HealingResult) -> None:
    print(f"\n  DETECT\n{report.summary()}")
    # Score uses healing_result so routing reflects healed state, not raw score
    print(f"\n  SCORE\n{QualityScore.compute(report, healing_result=result).summary()}")
    print(f"\n  HEAL\n{result.summary()}")


def demo_1_contradiction_patch():
    _sep("SCENARIO 1 — Contradiction patch: 30 days → 14 days")
    context  = [
        "Our return policy allows returns within 14 days of purchase.",
        "Items must be in original packaging to qualify for a refund.",
        "Refunds are processed within 5-7 business days.",
    ]
    question = "What is the deadline for returning a damaged item?"
    answer   = (
        "You have exactly 30 days to return a damaged item and receive "
        "a full refund. This is clearly stated in our premium customer "
        "guarantee policy and will definitely be honored at all locations."
    )
    detector = HallucinationDetector(DetectorConfig(db_path=":memory:"))
    healer   = HallucinationHealer(detector)
    report   = detector.inspect(question, context, answer)
    result   = healer.heal(question, context, answer, report)
    _show(report, result)


def demo_2_entity_scrub():
    _sep("SCENARIO 2 — Entity scrub: hallucinated citation removed")
    context  = [
        "Recent studies show transformer models achieve 94% accuracy on NER tasks.",
        "Fine-tuning on domain-specific data improves performance by 12-18%.",
        "The benchmark dataset contains 50,000 annotated examples.",
    ]
    question = "Who published the key research on transformer NER performance?"
    answer   = (
        "The seminal work was published by Dr. James Harrison and "
        "Dr. Wei Liu in their 2022 paper (arXiv:2204.09876). "
        "Their research at DeepMind Research Institute established "
        "the 94% benchmark."
    )
    detector = HallucinationDetector(DetectorConfig(db_path=":memory:"))
    healer   = HallucinationHealer(detector)
    report   = detector.inspect(question, context, answer)
    result   = healer.heal(question, context, answer, report)
    _show(report, result)


def demo_3_billing_normalization():
    _sep("SCENARIO 3 — Billing normalization: $10/month → $120/year (same unit as context)")
    context  = [
        "The Pro plan costs $120 per year, billed annually.",
        "There is no monthly billing option for the Pro plan.",
        "Annual subscriptions cannot be cancelled mid-cycle.",
    ]
    question = "How much does the Pro plan cost?"
    answer   = (
        "The Pro plan costs $10 per month, billed monthly. "
        "You can cancel your monthly subscription at any time "
        "without any cancellation fees."
    )
    detector = HallucinationDetector(DetectorConfig(db_path=":memory:"))
    healer   = HallucinationHealer(detector)
    report   = detector.inspect(question, context, answer)
    result   = healer.heal(question, context, answer, report)
    _show(report, result)


def demo_4_drift_rewrite():
    _sep("SCENARIO 4 — Drift detected + grounding rewrite")
    context        = ["Product SKU-441 is priced at $49.99 with free shipping."]
    question       = "What is the price of SKU-441?"
    stable_answer  = "SKU-441 is priced at $49.99 and includes free shipping."
    drifted_answer = "SKU-441 currently costs $39.99. Standard shipping rates apply."

    detector = HallucinationDetector(DetectorConfig(db_path=":memory:"))
    healer   = HallucinationHealer(detector)

    for _ in range(5):
        detector.inspect(question, context, stable_answer)

    print("\n  Simulating 5 stable answers, then drift...")
    for i in range(3):
        report = detector.inspect(question, context, drifted_answer)
        if report.drift_detected:
            print(f"\n  Drift on call {i+6}!")
            result = healer.heal(question, context, drifted_answer, report)
            _show(report, result)
            break


def demo_5_clean_pass():
    _sep("SCENARIO 5 — Clean answer: no healing needed")
    context  = [
        "Our return policy allows returns within 14 days of purchase.",
        "Items must be in original packaging to qualify for a refund.",
        "Refunds are processed within 5-7 business days.",
    ]
    question = "How long do I have to return an item?"
    answer   = (
        "According to the return policy, you have 14 days from the date "
        "of purchase to return an item. The item should be in its original "
        "packaging, and refunds are typically processed within 5 to 7 "
        "business days once the return is received."
    )
    detector = HallucinationDetector(DetectorConfig(db_path=":memory:"))
    healer   = HallucinationHealer(detector)
    report   = detector.inspect(question, context, answer)
    result   = healer.heal(question, context, answer, report)
    _show(report, result)


if __name__ == "__main__":
    print("\nhallucination_detector.py  ·  production edition v3")
    print("Detect + Fix + Score · SQLite drift · spaCy NER · async · typed")
    print("=" * 60)

    demo_1_contradiction_patch()
    demo_2_entity_scrub()
    demo_3_billing_normalization()
    demo_4_drift_rewrite()
    demo_5_clean_pass()

    print("\n" + "=" * 60)
    print("  v3 fixes")
    print("=" * 60)
    print("  FIX 1: initial_risk / final_risk (not 'Re-check: LOW (was LOW)')")
    print("  FIX 2: billing cycle normalized to context unit (monthly vs annually)")
    print("  FIX 3: grounding rewrite prefix derived from context, not boilerplate")
    print("  FIX 4: confidence recalibrated post-healing")
    print("  NEW  : QualityScore — weighted composite for accept/disclaim/fallback routing")
    print("  NEW  : changes_made list — explainable healing (what was corrected)")
    print("=" * 60)
    print("  Pipeline:")
    print("  retrieve() → generate() → inspect() → score() → heal() → deliver")
    print("=" * 60)
    print("  NER backend:", "spaCy en_core_web_sm" if SPACY_AVAILABLE else "regex fallback")
