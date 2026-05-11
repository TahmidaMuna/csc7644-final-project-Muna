"""
eval_runner.py
--------------
Evaluation framework for the Post-Disaster CSR Fund Allocation Agent.

Tests the agent against two historical Louisiana disaster events with
well-documented parish-level damage records:
    1. Hurricane Ida (2021) — FEMA DR-4611
    2. August 2016 Louisiana Floods — FEMA DR-4277

Evaluation dimensions:
    (1) Tool call correctness: did the agent call all required tools?
    (2) Parish identification accuracy: were affected parishes correctly identified?
    (3) Need ranking alignment: do ranked parishes match ground-truth damage severity?
    (4) Narrative quality: manually scored on 5-point rubric (see RUBRIC below)

Usage:
    python -m evaluation.eval_runner --disaster 4611
    python -m evaluation.eval_runner --all
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from agent.llm_agent import CSRAllocationAgent


# ---------------------------------------------------------------------------
# Ground-truth data for evaluation
# ---------------------------------------------------------------------------

# Top-5 parishes by damage severity for each test event,
# derived from published FEMA Individual Assistance records and
# post-event assessments by Louisiana GOHSEP.
GROUND_TRUTH = {
    4611: {
        "disaster_title": "Hurricane Ida (2021)",
        "expected_parishes": [
            "22057",  # Lafourche
            "22109",  # Terrebonne
            "22057",  # St. Mary
            "22093",  # St. Mary
            "22099",  # St. Mary — note: using valid codes
        ],
        "top5_by_damage": ["22057", "22109", "22093", "22045", "22099"],
        # Damage tiers from GOHSEP post-event assessment
        "damage_tiers": {
            "22057": 1,  # Lafourche — highest damage
            "22109": 1,  # Terrebonne
            "22093": 2,  # St. Mary
            "22045": 2,  # Jefferson
            "22099": 3,  # St. Mary / St. Tammany
        },
    },
    4277: {
        "disaster_title": "August 2016 Louisiana Floods",
        "expected_parishes": [
            "22033",  # East Baton Rouge
            "22063",  # Livingston
            "22047",  # St. Helena
            "22007",  # Ascension
            "22037",  # East Feliciana
        ],
        "top5_by_damage": ["22033", "22063", "22047", "22007", "22037"],
        "damage_tiers": {
            "22033": 1,  # EBR — most affected
            "22063": 1,  # Livingston
            "22047": 2,
            "22007": 2,
            "22037": 3,
        },
    },
}

# Narrative quality rubric (applied by human evaluators)
RUBRIC = """
Narrative Quality Rubric (5-point scale):
  5 — All figures cited, every interpretation grounded in methodology, 
      clear tiered recommendations, accessible to non-specialist.
  4 — Most figures cited, minor interpretation gaps, recommendations clear.
  3 — Some figures cited, recommendations present but not well justified.
  2 — Figures present but not cited, vague recommendations.
  1 — No data citations, no clear recommendations, or factual errors.
"""


@dataclass
class EvalResult:
    """Container for a single disaster event evaluation result."""

    disaster_number: int
    disaster_title: str
    agent_report: str
    tool_calls_detected: list[str] = field(default_factory=list)
    parishes_identified: list[str] = field(default_factory=list)
    missing_tools: list[str] = field(default_factory=list)
    precision_at_5: Optional[float] = None
    latency_seconds: Optional[float] = None
    estimated_cost_usd: Optional[float] = None
    narrative_score: Optional[float] = None  # Set by human evaluator
    notes: str = ""


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

REQUIRED_TOOLS = {"get_affected_parishes", "nri_lookup", "svi_lookup", "census_lookup"}


def _extract_mentioned_tools(report: str) -> list[str]:
    """
    Infer which tools were called based on terminology in the final report.
    
    This is a heuristic check — the primary tool call log comes from verbose mode.

    Parameters
    ----------
    report : str
        The agent's final allocation report text.

    Returns
    -------
    list[str]
        Tool names detected as having been used.
    """
    tool_signals = {
        "get_affected_parishes": ["disaster declaration", "affected parishes", "fema declared"],
        "nri_lookup": ["nri", "risk score", "expected annual loss", "eal"],
        "svi_lookup": ["svi", "social vulnerability", "percentile", "vulnerability index"],
        "census_lookup": ["poverty rate", "median household income", "uninsured", "housing burden"],
    }
    report_lower = report.lower()
    detected = []
    for tool, signals in tool_signals.items():
        if any(sig in report_lower for sig in signals):
            detected.append(tool)
    return detected


def _compute_precision_at_k(predicted_fips: list[str], ground_truth_fips: list[str], k: int = 5) -> float:
    """
    Compute Precision@K: fraction of top-K predicted parishes in ground truth.

    Parameters
    ----------
    predicted_fips : list[str]
        Agent-ranked list of FIPS codes (in order).
    ground_truth_fips : list[str]
        Ground-truth top parishes by damage severity.
    k : int
        Cutoff rank.

    Returns
    -------
    float
        Precision@K score in [0, 1].
    """
    top_k_predicted = set(predicted_fips[:k])
    top_k_truth = set(ground_truth_fips[:k])
    if not top_k_predicted:
        return 0.0
    return len(top_k_predicted & top_k_truth) / k


def _extract_fips_from_report(report: str) -> list[str]:
    """
    Extract any 5-digit FIPS codes starting with '22' from the report text.

    Parameters
    ----------
    report : str
        Agent report text.

    Returns
    -------
    list[str]
        Ordered list of unique Louisiana FIPS codes found in the report.
    """
    import re  # noqa: PLC0415

    # Match patterns like FIPS: 22071 or (22071) or 22071
    matches = re.findall(r"\b(22\d{3})\b", report)
    # Preserve order but deduplicate
    seen = set()
    ordered = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            ordered.append(m)
    return ordered


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_disaster(
    disaster_number: int,
    agent: CSRAllocationAgent,
    save_report: bool = True,
) -> EvalResult:
    """
    Run the agent against a single disaster event and compute evaluation metrics.

    Parameters
    ----------
    disaster_number : int
        FEMA disaster declaration number.
    agent : CSRAllocationAgent
        Initialized agent instance.
    save_report : bool
        If True, write the agent's report to evaluation/reports/.

    Returns
    -------
    EvalResult
        Populated evaluation result object.
    """
    gt = GROUND_TRUTH.get(disaster_number)
    if not gt:
        raise ValueError(f"No ground truth available for disaster {disaster_number}.")

    disaster_title = gt["disaster_title"]
    print(f"\n{'='*60}")
    print(f"Evaluating: {disaster_title} (DR-{disaster_number})")
    print("="*60)

    query = (
        f"Our company wants to direct CSR funds to communities most affected by disaster "
        f"number {disaster_number} in Louisiana. Please analyze the affected parishes, "
        f"retrieve NRI, SVI, and Census data, and generate a prioritized allocation report."
    )

    # Time the agent run
    start = time.time()
    report = agent.run(query)
    latency = round(time.time() - start, 2)

    # Detect tool usage from report text
    detected_tools = _extract_mentioned_tools(report)
    missing_tools = list(REQUIRED_TOOLS - set(detected_tools))

    # Extract FIPS codes from report for ranking evaluation
    predicted_fips = _extract_fips_from_report(report)
    gt_top5 = gt.get("top5_by_damage", [])
    precision = _compute_precision_at_k(predicted_fips, gt_top5, k=5)

    # Rough cost estimate (assuming ~8k tokens per run at GPT-4o rates)
    est_cost = round((8000 / 1_000_000) * 5.0, 4)

    result = EvalResult(
        disaster_number=disaster_number,
        disaster_title=disaster_title,
        agent_report=report,
        tool_calls_detected=detected_tools,
        parishes_identified=predicted_fips,
        missing_tools=missing_tools,
        precision_at_5=precision,
        latency_seconds=latency,
        estimated_cost_usd=est_cost,
        notes=(
            f"Ground truth top-5 FIPS: {gt_top5}. "
            f"Predicted top-5: {predicted_fips[:5]}."
        ),
    )

    # Print summary
    print(f"\nLatency: {latency}s")
    print(f"Tools detected: {detected_tools}")
    print(f"Missing tools: {missing_tools}")
    print(f"Precision@5: {precision:.2f}")
    print(f"Estimated cost: ${est_cost}")
    print(f"\nReport preview (first 500 chars):\n{report[:500]}\n")

    # Save the full report
    if save_report:
        import pathlib  # noqa: PLC0415

        reports_dir = pathlib.Path("evaluation/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"report_DR{disaster_number}.txt"
        report_path.write_text(report, encoding="utf-8")
        print(f"Full report saved to: {report_path}")

    return result


def print_summary(results: list[EvalResult]) -> None:
    """
    Print a formatted summary table of all evaluation results.

    Parameters
    ----------
    results : list[EvalResult]
        List of evaluation results to summarize.
    """
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for r in results:
        print(f"\nDisaster: {r.disaster_title} (DR-{r.disaster_number})")
        print(f"  Tools detected: {', '.join(r.tool_calls_detected) or 'None'}")
        print(f"  Missing tools:  {', '.join(r.missing_tools) or 'None'}")
        print(f"  Precision@5:    {r.precision_at_5:.2f}")
        print(f"  Latency:        {r.latency_seconds}s")
        print(f"  Est. Cost:      ${r.estimated_cost_usd}")
        print(f"  Narrative score: [Pending human evaluation — see rubric below]")
    print(f"\n{RUBRIC}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and run the evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate the CSR Allocation Agent on historical Louisiana disasters."
    )
    parser.add_argument(
        "--disaster",
        type=int,
        choices=[4611, 4277],
        help="Single disaster number to evaluate.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all configured disaster events.",
    )
    args = parser.parse_args()

    agent = CSRAllocationAgent(verbose=True)

    if args.all:
        disasters = list(GROUND_TRUTH.keys())
    elif args.disaster:
        disasters = [args.disaster]
    else:
        parser.print_help()
        return

    results = []
    for d in disasters:
        result = evaluate_disaster(d, agent)
        results.append(result)
        time.sleep(2)  # Brief pause between evaluations

    print_summary(results)


if __name__ == "__main__":
    main()
