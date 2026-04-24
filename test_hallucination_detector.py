"""
tests/test_hallucination_detector.py
=====================================
Full pytest suite for HallucinationDetector.

Covers:
  - All 5 risk levels across all 5 checks
  - Thread safety (concurrent inspect() calls)
  - Async ainspect()
  - SQLite drift persistence across detector instances
  - Config overrides
  - Edge cases: empty answer, empty context, unicode, very long text
  - HallucinationBlocked exception
  - Report.to_dict() structure
  - Stats accuracy

Run:
    pytest tests/ -v
    pytest tests/ -v --tb=short   # faster output
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
import threading
import time
import pytest
from hallucination_detector import (
    HallucinationDetector,
    HallucinationReport,
    HallucinationBlocked,
    HallucinationHealer,
    HealingResult,
    DetectorConfig,
    QualityScore,
    ConfidenceScorer,
    FaithfulnessScorer,
    ContradictionDetector,
    EntityHallucinationDetector,
    AnswerDriftMonitor,
    SPACY_AVAILABLE,
)


# ─────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────

@pytest.fixture
def detector():
    """Fresh in-memory detector for each test."""
    return HallucinationDetector(DetectorConfig(db_path=":memory:", log_flagged=False))


@pytest.fixture
def return_policy_context():
    return [
        "Our return policy allows returns within 14 days of purchase.",
        "Items must be in original packaging to qualify for a refund.",
        "Refunds are processed within 5-7 business days.",
    ]


@pytest.fixture
def pricing_context():
    return [
        "The Pro plan costs $120 per year, billed annually.",
        "There is no monthly billing option for the Pro plan.",
        "Annual subscriptions cannot be cancelled mid-cycle.",
    ]


@pytest.fixture
def research_context():
    return [
        "Recent studies show transformer models achieve 94% accuracy on NER tasks.",
        "Fine-tuning on domain-specific data improves performance by 12-18%.",
        "The benchmark dataset contains 50,000 annotated examples.",
    ]


# ─────────────────────────────────────────────────────────
# CONFIDENCE SCORER
# ─────────────────────────────────────────────────────────

class TestConfidenceScorer:
    def setup_method(self):
        self.scorer = ConfidenceScorer()

    def test_assertive_answer_scores_high(self):
        answer = "The policy definitely expires on March 31. It is clearly stated and will always apply."
        score = self.scorer.score(answer)
        assert score > 0.70

    def test_hedged_answer_scores_low(self):
        answer = "According to the context, it might be around 14 days, but I'm not certain."
        score = self.scorer.score(answer)
        assert score < 0.50

    def test_score_bounded(self):
        for text in ["", "yes", "a" * 1000, "DEFINITELY ALWAYS CERTAINLY GUARANTEED"]:
            score = self.scorer.score(text)
            assert 0.0 <= score <= 1.0

    def test_empty_string(self):
        score = self.scorer.score("")
        assert 0.0 <= score <= 1.0

    def test_unicode_answer(self):
        score = self.scorer.score("La politique est définitivement valide. Il est clairement établi.")
        assert 0.0 <= score <= 1.0


# ─────────────────────────────────────────────────────────
# FAITHFULNESS SCORER
# ─────────────────────────────────────────────────────────

class TestFaithfulnessScorer:
    def setup_method(self):
        self.scorer = FaithfulnessScorer()

    def test_grounded_answer_scores_high(self):
        context = ["Returns are accepted within 14 days in original packaging."]
        answer  = "According to the policy, returns are accepted within 14 days and the item must be in original packaging."
        score, ungrounded = self.scorer.score(answer, context)
        assert score >= 0.80
        assert len(ungrounded) == 0

    def test_fabricated_answer_scores_low(self):
        context = ["Returns are accepted within 14 days."]
        answer  = "You have 60 days to return items. Our VIP members get a lifetime guarantee."
        score, ungrounded = self.scorer.score(answer, context)
        assert score <= 0.50
        assert len(ungrounded) > 0

    def test_empty_answer_returns_perfect_score(self):
        score, ungrounded = self.scorer.score("", ["some context"])
        assert score == 1.0
        assert ungrounded == []

    def test_question_only_answer_excluded(self):
        # Questions shouldn't be counted as claims
        score, ungrounded = self.scorer.score("What is the return policy?", ["Context here."])
        assert score == 1.0

    def test_custom_overlap_threshold(self):
        strict = FaithfulnessScorer(overlap_threshold=0.90)
        context = ["Returns accepted within 14 days."]
        answer  = "You can return items within 14 days if they meet policy requirements."
        strict_score, _ = strict.score(answer, context)
        lax_score, _    = self.scorer.score(answer, context)
        assert strict_score <= lax_score


# ─────────────────────────────────────────────────────────
# CONTRADICTION DETECTOR
# ─────────────────────────────────────────────────────────

class TestContradictionDetector:
    def setup_method(self):
        self.detector = ContradictionDetector()

    def test_numeric_contradiction(self):
        context = ["The Pro plan costs $120 per year."]
        answer  = "The Pro plan costs $10 per month."
        found, reason = self.detector.detect(answer, context)
        assert found
        assert "10" in reason or "120" in reason

    def test_no_contradiction_matching_numbers(self):
        context = ["Refunds processed in 5-7 business days."]
        answer  = "Your refund will arrive in 5 to 7 business days."
        found, _ = self.detector.detect(answer, context)
        assert not found

    def test_negation_flip(self):
        context = ["The platform does not support real-time collaboration."]
        answer  = "The platform supports real-time collaboration features."
        found, reason = self.detector.detect(answer, context)
        assert found

    def test_temporal_contradiction(self):
        context = ["Annual subscriptions renew every 12 months."]
        answer  = "Your subscription renews every 1 month."
        found, reason = self.detector.detect(answer, context)
        assert found
        assert "1" in reason or "12" in reason

    def test_no_contradiction_clean_answer(self):
        context = ["Returns accepted within 14 days of purchase."]
        answer  = "You have 14 days from the date of purchase to return an item."
        found, _ = self.detector.detect(answer, context)
        assert not found

    def test_empty_context(self):
        found, _ = self.detector.detect("Some answer.", [])
        assert not found

    def test_empty_answer(self):
        found, _ = self.detector.detect("", ["Some context."])
        assert not found


# ─────────────────────────────────────────────────────────
# ENTITY HALLUCINATION DETECTOR
# ─────────────────────────────────────────────────────────

class TestEntityHallucinationDetector:
    def setup_method(self):
        self.detector = EntityHallucinationDetector()

    def test_fabricated_person_flagged(self):
        context  = ["Studies show 94% accuracy on NER tasks."]
        answer   = "Dr. James Harrison published the seminal paper on this."
        entities = self.detector.detect(answer, context)
        # Should flag James Harrison as absent from context
        assert any("Harrison" in e or "James" in e for e in entities)

    def test_fabricated_citation_flagged(self):
        context  = ["Transformers achieve high accuracy on NER benchmarks."]
        answer   = "As shown in Smith et al. (2021), performance exceeds prior work."
        entities = self.detector.detect(answer, context)
        assert any("2021" in e or "Smith" in e for e in entities)

    def test_entity_present_in_context_not_flagged(self):
        context  = ["Dr. Jane Smith at MIT conducted the research."]
        answer   = "Dr. Jane Smith's research confirms the findings."
        entities = self.detector.detect(answer, context)
        assert not any("Jane Smith" in e for e in entities)

    def test_empty_answer(self):
        assert self.detector.detect("", ["context"]) == []

    def test_no_false_positive_on_common_words(self):
        # "However" and similar words should not be flagged as entities
        context = ["The policy requires original packaging."]
        answer  = "However, according to the policy, original packaging is required."
        entities = self.detector.detect(answer, context)
        assert not any("However" in e for e in entities)


# ─────────────────────────────────────────────────────────
# ANSWER DRIFT MONITOR
# ─────────────────────────────────────────────────────────

class TestAnswerDriftMonitor:
    def _monitor(self):
        return AnswerDriftMonitor(db_path=":memory:", window_size=50, drift_threshold=0.35)

    def test_no_drift_on_first_answer(self):
        mon = self._monitor()
        detected, delta = mon.record("What is the price?", "The price is $49.99.")
        assert not detected
        assert delta == 0.0

    def test_no_drift_on_consistent_answers(self):
        mon = self._monitor()
        q = "What is the price of SKU-441?"
        a = "SKU-441 is priced at $49.99 with free shipping."
        for _ in range(6):
            detected, _ = mon.record(q, a)
        assert not detected

    def test_drift_detected_after_change(self):
        mon = self._monitor()
        q       = "What is the price of SKU-441?"
        stable  = "SKU-441 is priced at $49.99 with free shipping."
        drifted = "SKU-441 currently costs $39.99. Standard rates apply."
        for _ in range(5):
            mon.record(q, stable)
        detected, delta = mon.record(q, drifted)
        assert detected
        assert delta > 0.35

    def test_different_questions_dont_interfere(self):
        mon = self._monitor()
        q1, q2 = "What is the price?", "What is the return policy?"
        a1, a2 = "Price is $49.99.", "Returns accepted in 14 days."
        for _ in range(5):
            mon.record(q1, a1)
            mon.record(q2, a2)
        detected1, _ = mon.record(q1, a1)
        detected2, _ = mon.record(q2, a2)
        assert not detected1
        assert not detected2

    def test_persistence_across_instances(self, tmp_path):
        db_file = str(tmp_path / "drift.db")
        q = "What is the price?"
        a_stable  = "The price is $49.99 with free shipping included."
        a_drifted = "Currently the price is $29.99. Shipping not included."

        mon1 = AnswerDriftMonitor(db_path=db_file, window_size=50, drift_threshold=0.35)
        for _ in range(5):
            mon1.record(q, a_stable)

        # New instance, same DB
        mon2 = AnswerDriftMonitor(db_path=db_file, window_size=50, drift_threshold=0.35)
        detected, delta = mon2.record(q, a_drifted)
        assert detected, "Drift should be detected using history from mon1"

    def test_clear_history(self):
        mon = self._monitor()
        q = "How much does it cost?"
        for _ in range(5):
            mon.record(q, "The price is $49.99.")
        mon.clear_history(q)
        stats = mon.get_stats(q)
        assert stats["total_recorded"] == 0


# ─────────────────────────────────────────────────────────
# FULL DETECTOR — INTEGRATION TESTS
# ─────────────────────────────────────────────────────────

class TestHallucinationDetector:

    # ── Core scenarios ────────────────────────────────────

    def test_scenario_1_confident_lie(self, detector, return_policy_context):
        answer = (
            "You have exactly 30 days to return a damaged item and receive "
            "a full refund. This is clearly stated in our premium customer "
            "guarantee and will definitely be honored at all locations."
        )
        report = detector.inspect(
            "What is the deadline for returning a damaged item?",
            return_policy_context, answer
        )
        assert report.risk_level in ("high", "critical")
        assert report.is_hallucinating

    def test_scenario_2_hallucinated_entities(self, detector, research_context):
        answer = (
            "Dr. James Harrison and Dr. Wei Liu published this work "
            "in arXiv:2204.09876, establishing the 94% benchmark."
        )
        report = detector.inspect(
            "Who published the NER research?", research_context, answer
        )
        assert report.hallucinated_entities  # entities should be found
        assert "hallucinated_entities" in report.triggered_checks

    def test_scenario_3_contradiction(self, detector, pricing_context):
        answer = "The Pro plan costs $10 per month, billed monthly."
        report = detector.inspect("How much does the Pro plan cost?", pricing_context, answer)
        assert report.contradiction_found
        assert "contradiction" in report.triggered_checks
        assert report.risk_level in ("high", "critical")

    def test_scenario_4_drift_detected(self, detector):
        context = ["SKU-441 is priced at $49.99 with free shipping."]
        q       = "What is the price of SKU-441?"
        stable  = "SKU-441 is $49.99 with free shipping included."
        drifted = "SKU-441 costs $39.99. Standard shipping rates apply."

        for _ in range(5):
            detector.inspect(q, context, stable)

        report = detector.inspect(q, context, drifted)
        assert report.drift_detected

    def test_scenario_5_clean_answer_passes(self, detector, return_policy_context):
        answer = (
            "According to the return policy, you have 14 days from the date "
            "of purchase to return an item. The item should be in its original "
            "packaging, and refunds are processed within 5 to 7 business days."
        )
        report = detector.inspect("How long do I have to return?", return_policy_context, answer)
        assert not report.is_hallucinating
        assert report.risk_level == "low"
        assert report.triggered_checks == []

    # ── Risk levels ───────────────────────────────────────

    def test_risk_critical_on_contradiction(self, detector, pricing_context):
        answer = "The Pro plan costs $10 per month."
        report = detector.inspect("Cost?", pricing_context, answer)
        assert report.risk_level == "critical"

    def test_risk_low_on_clean_answer(self, detector, return_policy_context):
        answer = "According to policy, returns are accepted within 14 days."
        report = detector.inspect("Return window?", return_policy_context, answer)
        assert report.risk_level == "low"

    # ── Edge cases ────────────────────────────────────────

    def test_empty_answer(self, detector):
        report = detector.inspect("Any question?", ["Some context."], "")
        assert isinstance(report, HallucinationReport)
        assert not report.is_hallucinating  # empty answer can't assert anything

    def test_empty_context(self, detector):
        answer = "The price is $49.99."
        report = detector.inspect("What is the price?", [], answer)
        assert isinstance(report, HallucinationReport)

    def test_very_long_answer(self, detector):
        context = ["The return window is 14 days."]
        answer  = "The return window is 14 days. " * 500
        report  = detector.inspect("Return window?", context, answer)
        assert isinstance(report, HallucinationReport)
        assert 0.0 <= report.confidence_score <= 1.0
        assert 0.0 <= report.faithfulness_score <= 1.0

    def test_unicode_content(self, detector):
        context = ["Le délai de retour est de 14 jours."]
        answer  = "Selon la politique, vous avez 14 jours pour retourner un article."
        report  = detector.inspect("Délai de retour?", context, answer)
        assert isinstance(report, HallucinationReport)

    def test_single_word_answer(self, detector):
        report = detector.inspect("Is it available?", ["The product is available."], "Yes.")
        assert isinstance(report, HallucinationReport)

    # ── Report structure ──────────────────────────────────

    def test_report_has_latency(self, detector, return_policy_context):
        report = detector.inspect("Question?", return_policy_context, "Answer.")
        assert report.latency_ms >= 0.0

    def test_to_dict_has_required_keys(self, detector, return_policy_context):
        report = detector.inspect("Question?", return_policy_context, "Answer.")
        d = report.to_dict()
        required = {
            "question", "risk_level", "is_hallucinating",
            "confidence_score", "faithfulness_score",
            "contradiction_found", "contradiction_reason",
            "hallucinated_entities", "drift_detected", "drift_delta",
            "triggered_checks", "explanation", "latency_ms",
        }
        assert required.issubset(d.keys())

    def test_summary_is_string(self, detector, return_policy_context):
        report = detector.inspect("Question?", return_policy_context, "Answer.")
        assert isinstance(report.summary(), str)
        assert len(report.summary()) > 0

    # ── HallucinationBlocked ──────────────────────────────

    def test_hallucination_blocked_exception(self, detector, pricing_context):
        answer = "The Pro plan costs $10 per month."
        report = detector.inspect("Cost?", pricing_context, answer)
        if report.is_hallucinating:
            with pytest.raises(HallucinationBlocked) as exc_info:
                raise HallucinationBlocked(report)
            assert exc_info.value.report is report
            assert "risk=" in str(exc_info.value)

    # ── Config ────────────────────────────────────────────

    def test_strict_faithfulness_threshold(self, return_policy_context):
        strict = HallucinationDetector(DetectorConfig(
            faithfulness_threshold=0.99,
            confidence_threshold=0.01,
            db_path=":memory:",
            log_flagged=False,
        ))
        # Even a partial answer should now fail faithfulness
        answer = "Returns are allowed within some time period."
        report = strict.inspect("Return window?", return_policy_context, answer)
        assert "confident_but_unfaithful" in report.triggered_checks

    def test_stats_tracking(self, detector, return_policy_context, pricing_context):
        # Clean answer — should not be flagged
        detector.inspect(
            "Return window?", return_policy_context,
            "According to policy, returns are accepted within 14 days."
        )
        # Contradiction — should be flagged
        detector.inspect(
            "Cost?", pricing_context,
            "The Pro plan costs $10 per month, billed monthly."
        )
        stats = detector.stats()
        assert stats["total_inspected"] == 2
        assert stats["total_flagged"] >= 1
        assert "%" in stats["hallucination_rate"]

    def test_reset_stats(self, detector, return_policy_context):
        detector.inspect("Q?", return_policy_context, "A.")
        detector.reset_stats()
        stats = detector.stats()
        assert stats["total_inspected"] == 0
        assert stats["total_flagged"] == 0

    # ── Thread safety ─────────────────────────────────────

    def test_concurrent_inspect_thread_safe(self, detector, return_policy_context):
        results = []
        errors  = []

        def worker():
            try:
                r = detector.inspect(
                    "How long do I have to return?",
                    return_policy_context,
                    "You have 14 days to return items per policy.",
                )
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors in concurrent calls: {errors}"
        assert len(results) == 20
        for r in results:
            assert isinstance(r, HallucinationReport)

    # ── Async ─────────────────────────────────────────────

    def test_ainspect_returns_report(self, detector, return_policy_context):
        async def _run():
            return await detector.ainspect(
                "How long do I have to return?",
                return_policy_context,
                "According to policy, returns are accepted within 14 days.",
            )
        report = asyncio.run(_run())
        assert isinstance(report, HallucinationReport)
        assert not report.is_hallucinating

    def test_ainspect_detects_hallucination(self, detector, pricing_context):
        async def _run():
            return await detector.ainspect(
                "Cost?", pricing_context,
                "The Pro plan costs $10 per month.",
            )
        report = asyncio.run(_run())
        assert report.is_hallucinating

    def test_ainspect_concurrent(self, return_policy_context):
        async def _run():
            det = HallucinationDetector(DetectorConfig(db_path=":memory:", log_flagged=False))
            tasks = [
                det.ainspect(
                    "Return window?", return_policy_context,
                    "Returns accepted within 14 days of purchase."
                )
                for _ in range(10)
            ]
            results = await asyncio.gather(*tasks)
            return results

        reports = asyncio.run(_run())
        assert len(reports) == 10
        assert all(isinstance(r, HallucinationReport) for r in reports)

    # ── NER backend reporting ──────────────────────────────

    def test_stats_reports_ner_backend(self, detector):
        stats = detector.stats()
        assert "ner_backend" in stats
        assert stats["ner_backend"] in ("spaCy", "regex (fallback)")


# ─────────────────────────────────────────────────────────
# QUALITY SCORE — v3 additions
# ─────────────────────────────────────────────────────────

class TestQualityScore:

    def _make_report(self, **kwargs) -> HallucinationReport:
        defaults = dict(
            question="Q?", answer="A.",
            confidence_score=0.5, faithfulness_score=0.8,
            contradiction_found=False, drift_detected=False,
            risk_level="low", is_hallucinating=False,
            triggered_checks=[], explanation=[], latency_ms=5.0,
        )
        defaults.update(kwargs)
        return HallucinationReport(**defaults)

    # ── Routing tiers ─────────────────────────────────────

    def test_accept_on_high_score_no_healing(self):
        report = self._make_report(faithfulness_score=1.0, contradiction_found=False,
                                   confidence_score=0.8, latency_ms=5.0)
        qs = QualityScore.compute(report)
        assert qs.routing == "accept"
        assert qs.final_score >= 0.75

    def test_fallback_on_low_score_no_healing(self):
        report = self._make_report(faithfulness_score=0.0, contradiction_found=True,
                                   confidence_score=0.9, latency_ms=5.0)
        qs = QualityScore.compute(report)
        assert qs.routing == "fallback"

    def test_healed_accept_when_healing_applied(self):
        report = self._make_report(faithfulness_score=0.5, contradiction_found=True,
                                   latency_ms=5.0)
        heal_result = HealingResult(
            original_answer="bad", healed_answer="good",
            strategy_used="contradiction_patch", healing_applied=True,
            initial_risk="critical", final_risk="low",
        )
        qs = QualityScore.compute(report, healing_result=heal_result)
        assert qs.routing == "healed_accept"

    def test_discard_when_safe_decline_served(self):
        report = self._make_report(faithfulness_score=0.0, contradiction_found=True)
        heal_result = HealingResult(
            original_answer="bad", healed_answer="safe decline",
            strategy_used="safe_decline", healing_applied=True,
            initial_risk="critical", final_risk="low",
        )
        qs = QualityScore.compute(report, healing_result=heal_result)
        assert qs.routing == "discard"

    def test_no_healing_result_uses_score_threshold(self):
        report = self._make_report(faithfulness_score=0.5, contradiction_found=False,
                                   confidence_score=0.4, latency_ms=5.0)
        qs = QualityScore.compute(report)
        assert qs.routing in ("accept", "fallback")   # not healed_accept

    # ── Latency curve ─────────────────────────────────────

    def test_latency_full_score_under_20ms(self):
        report = self._make_report(latency_ms=10.0)
        qs = QualityScore.compute(report)
        assert abs(qs.latency_component - 0.10) < 0.001

    def test_latency_reduced_at_35ms(self):
        fast = QualityScore.compute(self._make_report(latency_ms=10.0))
        slow = QualityScore.compute(self._make_report(latency_ms=35.0))
        assert slow.latency_component < fast.latency_component
        assert slow.latency_component > 0.05   # still above the 50ms floor

    def test_latency_steep_penalty_at_60ms(self):
        at_50  = QualityScore.compute(self._make_report(latency_ms=50.0))
        at_100 = QualityScore.compute(self._make_report(latency_ms=100.0))
        assert at_100.latency_component < at_50.latency_component
        # At 60ms we should be noticeably penalised vs 20ms
        at_60 = QualityScore.compute(self._make_report(latency_ms=60.0))
        assert at_60.latency_component < 0.05

    def test_latency_floor_at_zero(self):
        qs = QualityScore.compute(self._make_report(latency_ms=500.0))
        assert qs.latency_component == 0.0

    # ── Drift penalty ──────────────────────────────────────

    def test_drift_subtracts_020(self):
        no_drift   = QualityScore.compute(self._make_report(drift_detected=False))
        with_drift = QualityScore.compute(self._make_report(drift_detected=True))
        assert abs((no_drift.final_score - with_drift.final_score) - 0.20) < 0.01

    def test_drift_penalty_field(self):
        qs = QualityScore.compute(self._make_report(drift_detected=True))
        assert qs.drift_penalty == 0.20

    def test_no_drift_no_penalty(self):
        qs = QualityScore.compute(self._make_report(drift_detected=False))
        assert qs.drift_penalty == 0.0

    def test_drift_score_floored_at_zero(self):
        # Even with drift + contradiction + unfaithful, score can't go negative
        report = self._make_report(
            faithfulness_score=0.0, contradiction_found=True,
            drift_detected=True, confidence_score=1.0,
        )
        qs = QualityScore.compute(report)
        assert qs.final_score >= 0.0

    # ── Confidence recalibration by strategy ──────────────

    def test_contradiction_patch_boosts_confidence(self):
        original = 0.50
        recalibrated = HallucinationHealer._recalibrate_confidence(
            "The price is $120 per year.", [], "contradiction_patch", original
        )
        assert recalibrated > original
        assert recalibrated <= 0.80

    def test_contradiction_patch_caps_at_080(self):
        recalibrated = HallucinationHealer._recalibrate_confidence(
            "The policy is clear.", [], "contradiction_patch", 0.90
        )
        assert recalibrated == 0.80

    def test_entity_scrub_reduces_confidence(self):
        original = 0.70
        recalibrated = HallucinationHealer._recalibrate_confidence(
            "Some answer.", [], "entity_scrub", original
        )
        assert recalibrated < original
        assert abs(recalibrated - original * 0.85) < 0.001

    def test_grounding_rewrite_uses_linguistic_score(self):
        hedged = "According to the provided data: returns are accepted in 14 days."
        assertive = "Returns are definitely accepted in 14 days absolutely."
        score_hedged    = HallucinationHealer._recalibrate_confidence(hedged,    [], "grounding_rewrite", 0.9)
        score_assertive = HallucinationHealer._recalibrate_confidence(assertive, [], "grounding_rewrite", 0.9)
        assert score_hedged < score_assertive

    # ── to_dict includes drift_penalty ────────────────────

    def test_to_dict_has_drift_penalty(self):
        qs = QualityScore.compute(self._make_report(drift_detected=True))
        d  = qs.to_dict()
        assert "drift_penalty" in d
        assert d["drift_penalty"] == 0.20
