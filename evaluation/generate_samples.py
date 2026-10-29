"""Sample generation utility for qualitative model evaluation."""
from typing import Dict, List, Optional
from model.generation import generate_text


STANDARD_EVAL_PROMPTS = [
    "YOU:\nHi\n\nBOT:\n",
    "YOU:\nWhat is your name?\n\nBOT:\n",
    "YOU:\nWhat is Python?\n\nBOT:\n",
    "YOU:\nTell me a short story.\n\nBOT:\n",
]


def generate_evaluation_samples(
    model,
    tokenizer,
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: Optional[int] = 40,
) -> Dict[str, str]:
    """Generate completions for a list of evaluation prompts."""
    prompts = prompts or STANDARD_EVAL_PROMPTS
    results = {}
    for prompt in prompts:
        output = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        # Strip prompt prefix from output for clarity
        completion = output[len(prompt) :] if output.startswith(prompt) else output
        results[prompt] = completion
    return results

