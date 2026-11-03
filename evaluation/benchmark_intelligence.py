"""Automated Intelligence Evaluation Harness for Mini GPT.

Evaluates trained checkpoints on genuinely held-out, multi-category benchmarks:
1. Factual Recall
2. Code Execution and Test Assertions
3. Logical Reasoning
4. Conceptual Explanations
5. Uncertainty and Hallucination Resistance
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.metrics import calculate_distinct_ngrams, calculate_repetition_rate


def extract_python_code(text: str) -> str:
    """Extract Python function definition or code block from model response."""
    # Check for markdown code blocks
    code_match = re.search(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    # Check for raw function definition starting with 'def '
    def_match = re.search(r"(def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(.*?\):.*?)(?:\n\n|\Z)", text, re.DOTALL)
    if def_match:
        return def_match.group(1).strip()
    return text.strip()


def evaluate_code_execution(code_str: str, test_code: str, timeout: float = 3.0) -> bool:
    """Safely execute extracted Python code against unit test assertions in an isolated subprocess."""
    import subprocess
    import sys

    if not code_str or not code_str.strip():
        return False

    script = f"{code_str}\n\n{test_code}\n"
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


def evaluate_question(
    service_or_model_fn,
    category: str,
    item: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate a single benchmark item and return results."""
    question = item["question"]
    messages = [{"role": "user", "content": question}]

    # Generate response
    res = service_or_model_fn(messages)
    reply = res.get("reply", "") if isinstance(res, dict) else str(res)
    lower_reply = reply.lower()

    passed = False
    details = ""

    if category in ("factual", "reasoning", "explanation", "uncertainty"):
        keywords = [kw.lower() for kw in item.get("expected_keywords", [])]
        matched = [kw for kw in keywords if kw in lower_reply]
        # For multi-concept explanations require at least 2 distinct keywords; for single-answer categories require >= 1
        min_required = item.get("min_keywords", 2 if category == "explanation" else 1)
        required_count = min(min_required, len(keywords)) if keywords else 0
        passed = len(matched) >= required_count and required_count > 0
        details = f"Matched keywords: {matched} / {keywords} (required: {required_count})"

    elif category == "coding":
        extracted = extract_python_code(reply)
        test_code = item.get("test_code", "")
        passed = evaluate_code_execution(extracted, test_code)
        details = f"Code execution {'PASSED' if passed else 'FAILED'}"

    rep_3 = calculate_repetition_rate(reply, n=3)
    dist_2 = calculate_distinct_ngrams(reply, n=2)

    return {
        "id": item.get("id"),
        "category": category,
        "question": question,
        "reply": reply,
        "passed": passed,
        "details": details,
        "repetition_rate_3gram": rep_3,
        "distinct_2grams": dist_2,
    }


def run_intelligence_benchmark(
    service_or_model_fn,
    benchmark_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run complete intelligence benchmark across all categories."""
    if benchmark_path is None:
        benchmark_path = Path(__file__).resolve().parent / "held_out_questions.json"

    data = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    categories = data.get("categories", {})

    results = []
    category_summary = {}

    for cat_name, items in categories.items():
        if not items:
            category_summary[cat_name] = {
                "total": 0,
                "passed": 0,
                "pass_rate": 0.0,
                "avg_3gram_repetition": 0.0,
                "status": "no_questions_tested",
            }
            continue

        cat_passed = 0
        cat_rep = 0.0
        for item in items:
            res = evaluate_question(service_or_model_fn, cat_name, item)
            results.append(res)
            if res["passed"]:
                cat_passed += 1
            cat_rep += res["repetition_rate_3gram"]

        count = len(items)
        category_summary[cat_name] = {
            "total": count,
            "passed": cat_passed,
            "pass_rate": cat_passed / count,
            "avg_3gram_repetition": cat_rep / count,
        }

    total_items = len(results)
    total_passed = sum(1 for r in results if r["passed"])
    overall_pass_rate = (total_passed / total_items) if total_items > 0 else 0.0

    return {
        "total_questions": total_items,
        "total_passed": total_passed,
        "overall_pass_rate": overall_pass_rate,
        "category_summary": category_summary,
        "results": results,
    }
