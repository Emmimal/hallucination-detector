"""
demo.py — hallucination-detector production scenarios
======================================================
Five real failure modes, each with detect → score → heal output.

Run:
    python demo.py
    python demo.py --scenario 1   # run a single scenario

Scenarios:
    1  Confident lie          — 30 days vs 14 days return policy
    2  Hallucinated citation  — Dr. James Harrison, arXiv:2204.09876
    3  Billing contradiction  — $10/month vs $120/year
    4  Answer drift           — SKU-441 price shifts from $49.99 to $39.99
    5  Clean answer           — passes through unchanged, no healing needed
"""

import argparse
from hallucination_detector import (
    HallucinationDetector,
    HallucinationHealer,
    DetectorConfig,
    QualityScore,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sep(title: str) -> None:
    width = 64
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def _show(label: str, report, result, score) -> None:
    print(f"\n── DETECT {'─' * 54}")
    print(report.summary())
    print(f"\n── SCORE {'─' * 55}")
    print(score.summary())
    print(f"\n── HEAL {'─' * 56}")
    print(result.summary())


def _make_detector() -> tuple:
    cfg      = DetectorConfig(db_path=":memory:", log_flagged=False)
    detector = HallucinationDetector(cfg)
    healer   = HallucinationHealer(detector)
    return detector, healer


# ── Scenario 1 ────────────────────────────────────────────────────────────────

def scenario_1():
    _sep("SCENARIO 1 — Confident lie: 30 days vs 14 days return policy")
    print("""
  The model read the correct policy (14 days) and stated a different one
  (30 days) with complete confidence. This ran in production for six weeks.

  Expected outcome: contradiction_patch replaces 30 with 14 in-place.
    """)

    context = [
        "Our return policy allows returns within 14 days of purchase.",
        "Items must be in original packaging to qualify for a refund.",
        "Refunds are processed within 5-7 business days.",
    ]
    question = "What is the deadline for returning a damaged item?"
    answer = (
        "You have exactly 30 days to return a damaged item and receive "
        "a full refund. This is clearly stated in our premium customer "
        "guarantee policy and will definitely be honored at all locations."
    )

    detector, healer = _make_detector()
    report = detector.inspect(question, context, answer)
    result = healer.heal(question, context, answer, report)
    score  = QualityScore.compute(report, healing_result=result)
    _show("scenario_1", report, result, score)


# ── Scenario 2 ────────────────────────────────────────────────────────────────

def scenario_2():
    _sep("SCENARIO 2 — Hallucinated citation: Dr. James Harrison, arXiv:2204.09876")
    print("""
  The model invented two researchers and a paper citation. None of them
  appear anywhere in the retrieved context.

  Expected outcome: entity_scrub removes the offending sentences and
  appends a transparency note.
    """)

    context = [
        "Recent studies show transformer models achieve 94% accuracy on NER tasks.",
        "Fine-tuning on domain-specific data improves performance by 12-18%.",
        "The benchmark dataset contains 50,000 annotated examples.",
    ]
    question = "Who published the key research on transformer NER performance?"
    answer = (
        "The seminal work was published by Dr. James Harrison and "
        "Dr. Wei Liu in their 2022 paper (arXiv:2204.09876). "
        "Their research at DeepMind Research Institute established "
        "the 94% benchmark that the field now uses as a standard."
    )

    detector, healer = _make_detector()
    report = detector.inspect(question, context, answer)
    result = healer.heal(question, context, answer, report)
    score  = QualityScore.compute(report, healing_result=result)
    _show("scenario_2", report, result, score)


# ── Scenario 3 ────────────────────────────────────────────────────────────────

def scenario_3():
    _sep("SCENARIO 3 — Billing contradiction: $10/month vs $120/year")
    print("""
  The model swapped a monthly price for an annual plan. Context is explicit:
  $120/year, billed annually, no monthly option.

  Expected outcome: contradiction_patch replaces $10 → $120 and normalizes
  all billing-cycle language to annual terms in a single ordered pass.
    """)

    context = [
        "The Pro plan costs $120 per year, billed annually.",
        "There is no monthly billing option for the Pro plan.",
        "Annual subscriptions cannot be cancelled mid-cycle.",
    ]
    question = "How much does the Pro plan cost?"
    answer = (
        "The Pro plan costs $10 per month, billed monthly. "
        "You can cancel your monthly subscription at any time "
        "without any cancellation fees."
    )

    detector, healer = _make_detector()
    report = detector.inspect(question, context, answer)
    result = healer.heal(question, context, answer, report)
    score  = QualityScore.compute(report, healing_result=result)
    _show("scenario_3", report, result, score)


# ── Scenario 4 ────────────────────────────────────────────────────────────────

def scenario_4():
    _sep("SCENARIO 4 — Answer drift: SKU-441 price shifts from $49.99 to $39.99")
    print("""
  Five stable answers at $49.99. Then the price silently changes to $39.99.
  No single answer is obviously wrong — each one is locally plausible.
  Only the drift monitor catches it by comparing fingerprints over time.

  Expected outcome: drift flagged on call 6, grounding_rewrite rebuilds
  the answer from the context sentence.
    """)

    context        = ["Product SKU-441 is priced at $49.99 with free shipping."]
    question       = "What is the price of SKU-441?"
    stable_answer  = "SKU-441 is priced at $49.99 and includes free shipping."
    drifted_answer = "SKU-441 currently costs $39.99. Standard shipping rates apply."

    detector, healer = _make_detector()

    print("  Seeding 5 stable answers...")
    for i in range(5):
        detector.inspect(question, context, stable_answer)
        print(f"    Call {i+1}: ${49.99} — stable")

    print("\n  Injecting drifted answer...")
    for attempt in range(1, 4):
        report = detector.inspect(question, context, drifted_answer)
        print(f"    Call {5+attempt}: $39.99 — drift_detected={report.drift_detected}")
        if report.drift_detected:
            result = healer.heal(question, context, drifted_answer, report)
            score  = QualityScore.compute(report, healing_result=result)
            _show("scenario_4", report, result, score)
            break


# ── Scenario 5 ────────────────────────────────────────────────────────────────

def scenario_5():
    _sep("SCENARIO 5 — Clean answer: no healing needed")
    print("""
  A well-grounded, hedged answer that matches the retrieved context exactly.
  All five checks pass. The system does nothing — which is the correct behavior.

  Expected outcome: risk LOW, routing ACCEPT, strategy no_healing_needed.
    """)

    context = [
        "Our return policy allows returns within 14 days of purchase.",
        "Items must be in original packaging to qualify for a refund.",
        "Refunds are processed within 5-7 business days.",
    ]
    question = "How long do I have to return an item?"
    answer = (
        "According to the return policy, you have 14 days from the date "
        "of purchase to return an item. The item should be in its original "
        "packaging, and refunds are typically processed within 5 to 7 "
        "business days once the return is received."
    )

    detector, healer = _make_detector()
    report = detector.inspect(question, context, answer)
    result = healer.heal(question, context, answer, report)
    score  = QualityScore.compute(report, healing_result=result)
    _show("scenario_5", report, result, score)


# ── Entry point ───────────────────────────────────────────────────────────────

SCENARIOS = {
    1: scenario_1,
    2: scenario_2,
    3: scenario_3,
    4: scenario_4,
    5: scenario_5,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hallucination-detector demo")
    parser.add_argument(
        "--scenario", type=int, choices=[1, 2, 3, 4, 5],
        help="Run a single scenario (1-5). Omit to run all."
    )
    args = parser.parse_args()

    print("\nhallucination-detector · production demo")
    print("Detect + Fix + Score · SQLite drift · spaCy NER · async · typed")
    print("=" * 64)

    if args.scenario:
        SCENARIOS[args.scenario]()
    else:
        for fn in SCENARIOS.values():
            fn()

    print(f"\n{'=' * 64}")
    print("  Pipeline:")
    print("  retrieve() → generate() → inspect() → score() → heal() → deliver")
    print(f"{'=' * 64}\n")
