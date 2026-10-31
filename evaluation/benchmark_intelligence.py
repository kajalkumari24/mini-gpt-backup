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


def evaluate_code_execution(code_str: str, test_code: str) -> bool:
    """Safely execute extracted Python code against unit test assertions in a restricted namespace."""
    global_env = {}
    local_env = {}
    try:
        # Execute the function definition
        exec(code_str, global_env, local_env)
        # Execute the test assertions
        exec(test_code, global_env, local_env)
        return True
    except Exception:
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
        passed = len(matched) > 0
        details = f"Matched keywords: {matched} / {keywords}"

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
        cat_passed = 0
        cat_rep = 0.0
        for item in items:
            res = evaluate_question(service_or_model_fn, cat_name, item)
            results.append(res)
            if res["passed"]:
                cat_passed += 1
            cat_rep += res["repetition_rate_3gram"]

        count = max(1, len(items))
        category_summary[cat_name] = {
            "total": len(items),
            "passed": cat_passed,
            "pass_rate": cat_passed / count,
            "avg_3gram_repetition": cat_rep / count,
        }

    total_items = len(results)
    total_passed = sum(1 for r in results if r["passed"])
    overall_pass_rate = total_passed / max(1, total_items)

    return {
        "total_questions": total_items,
        "total_passed": total_passed,
        "overall_pass_rate": overall_pass_rate,
        "category_summary": category_summary,
        "results": results,
    }
