"""Format evaluation and model comparison results into readable Markdown reports."""
from typing import Dict, Any


def format_evaluation_summary(results: Dict[str, Any]) -> str:
    """Format single-model or multi-model evaluation metrics into a Markdown table."""
    lines = [
        "# Model Evaluation Summary",
        "",
        "| Model | Params | Size (MB) | Eval Loss | Perplexity | Bits/Char | Tokens/sec |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for name, data in results.items():
        if "error" in data:
            lines.append(f"| {name} | ERROR: {data['error']} | - | - | - | - | - |")
            continue

        params = f"{data.get('parameters', 0):,}"
        size_mb = f"{data.get('parameter_mb', 0.0):.2f}"
        loss = f"{data.get('eval_loss', 0.0):.4f}" if "eval_loss" in data else "N/A"
        ppl = f"{data.get('eval_perplexity', 0.0):.2f}" if "eval_perplexity" in data else "N/A"
        bpc = f"{data.get('eval_bits_per_char', 0.0):.2f}" if "eval_bits_per_char" in data else "N/A"
        speed = f"{data.get('tokens_per_second', 0.0):.1f}"

        lines.append(f"| {name} | {params} | {size_mb} | {loss} | {ppl} | {bpc} | {speed} |")

    lines.append("")
    return "\n".join(lines)

