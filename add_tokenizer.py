import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from pathlib import Path

from unsloth import FastModel
from transformers import CsmForConditionalGeneration, CsmProcessor
import typer


def main(target: Path) -> None:
    """
    Adds the tokenizer to a series of checkpoints in the target directory.
    """
    if not target.exists():
        raise FileNotFoundError(f"Target directory {target} does not exist.")

    samples = target / "samples"

    if not samples.exists():
        samples.mkdir(parents=True, exist_ok=True)

    # We expect a series of checkpoint directories named like "checkpoint-100", "checkpoint-200", etc.
    checkpoints = sorted(
        [d for d in target.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda x: int(x.name.split("-")[1])
    )

    generate_checkpoints: list[Path] = []

    for checkpoint in checkpoints:
        generate_checkpoints.append(checkpoint)

    if not generate_checkpoints:
        raise ValueError("No checkpoints found in the specified range.")

    model, tokenizer = FastModel.from_pretrained(
        model_name='sesame/csm-1b',  # Use the base model name
        # model_name="sesame/csm-1b",  # Use the base model name
        # model_name=checkpoint.name,
        max_seq_length=2048,  # Choose any for long context!
        dtype=None,  # Leave as None for auto-detection
        auto_model=CsmForConditionalGeneration,
        load_in_4bit=False,
        full_finetuning=True,
    )

    for checkpoint in generate_checkpoints:
        tokenizer.save_pretrained(checkpoint)


if __name__ == "__main__":
    typer.run(main)
