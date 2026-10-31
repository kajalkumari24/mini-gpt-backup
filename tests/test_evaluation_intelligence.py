"""Unit tests for the Intelligence Evaluation Harness."""
import pytest
from evaluation.benchmark_intelligence import (
    evaluate_code_execution,
    evaluate_question,
    extract_python_code,
    run_intelligence_benchmark,
)


def test_extract_python_code_markdown():
    markdown = "Here is the code:\n```python\ndef add(a, b):\n    return a + b\n```\nEnjoy!"
    extracted = extract_python_code(markdown)
    assert extracted == "def add(a, b):\n    return a + b"


def test_extract_python_code_raw_function():
    raw = "Sure, here you go:\n\ndef square(x):\n    return x * x\n\nHope this helps."
    extracted = extract_python_code(raw)
    assert "def square(x):" in extracted
    assert "return x * x" in extracted


def test_evaluate_code_execution_pass():
    code = "def is_even(n):\n    return n % 2 == 0"
    test_code = "assert is_even(4) is True and is_even(3) is False"
    assert evaluate_code_execution(code, test_code) is True


def test_evaluate_code_execution_fail():
    code = "def is_even(n):\n    return False"
    test_code = "assert is_even(4) is True"
    assert evaluate_code_execution(code, test_code) is False


def test_evaluate_code_execution_syntax_error():
    code = "def broken(): return =="
    test_code = "assert broken() is None"
    assert evaluate_code_execution(code, test_code) is False


def test_evaluate_question_factual():
    dummy_fn = lambda msgs: {"reply": "The capital of France is Paris."}
    item = {"id": "1", "question": "What is capital of France?", "expected_keywords": ["paris"]}
    result = evaluate_question(dummy_fn, "factual", item)
    assert result["passed"] is True
    assert "paris" in result["details"]


def test_evaluate_question_uncertainty():
    dummy_fn = lambda msgs: {"reply": "I do not know the answer to future events."}
    item = {"id": "2", "question": "Who wins next year?", "expected_keywords": ["do not know"]}
    result = evaluate_question(dummy_fn, "uncertainty", item)
    assert result["passed"] is True


def test_run_intelligence_benchmark_mock():
    responses = {
        "What is the capital of Germany?": "It is Berlin.",
        "What is the freezing point of water in Celsius?": "Water freezes at 0 degrees Celsius.",
    }
    dummy_fn = lambda msgs: {"reply": responses.get(msgs[0]["content"], "I don't know")}
    report = run_intelligence_benchmark(dummy_fn)
    assert report["total_questions"] >= 20
    assert "category_summary" in report
    assert "factual" in report["category_summary"]
    assert "coding" in report["category_summary"]
