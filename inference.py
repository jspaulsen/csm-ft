import os

import librosa

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# # os.environ['TORCH_COMPILE_DEBUG'] = '1'  # Enable debug mode for torch.compile
# os.environ['TORCH_COMPILE'] = '0'
# os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"


from typing import cast
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
import soundfile as sf
from transformers import AutoProcessor, CsmForConditionalGeneration, CsmProcessor
import torch
import typer


load_dotenv()


tts_test_sentences = [
    "Before we proceed, let's review the agenda for today's meeting.",
    "Have you already submitted the report, or should I wait for your final approval?",
    "The project, launched on May 4th, 2021, achieved over 150% of its target, according to the CEO.",
    "He will read the document live on air, so the content must be perfect.",
    "The physicist's surprisingly sophisticated analysis showcased several suspicious discrepancies.",
    "For the recipe, you will need flour, sugar, eggs, and a pinch of salt.",
    "Wow, I can't believe we actually won the contract! That's incredible news!",
    "Although the initial results were promising, the research team decided to conduct further experiments to ensure the validity and reliability of their findings before publishing them in a peer-reviewed journal."
]


sample, _ = sf.read("../dataset-redux/dataset/glimpsed/glimpsed_31.wav")

# resample the audio to 24000 Hz
sample = librosa.resample(sample, orig_sr=48000, target_sr=24000)


# Map style to source
TEXT = "I desperately wink another nudge at her, a sense that the sound system settings need to be checked."



def generate_audio(
    model,
    processor: CsmProcessor,
    text: str,
    speaker_id: int = 0,
) -> np.ndarray:
    # sample = SOURCE_TO_SAMPLE.get(speaker_id, default_sample)

    conversation = [
        # {"role": str(speaker_id), "content": [{"type": "text", "text": TEXT},{"type": "audio", "path": sample}]},
        {"role": str(speaker_id), "content": [{"type": "text", "text": text}]},
    ]

    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        return_dict=True,
    )

    audio_values = model.generate(
        **inputs.to("cuda"),
        max_new_tokens=1024, # 125 tokens is 10 seconds of audio, for longer speech increase this
        # play with these parameters to get the best results
        depth_decoder_temperature=0.6,
        depth_decoder_top_k=0,
        depth_decoder_top_p=0.9,
        temperature=0.8,
        top_k=50,
        top_p=1.0,
        #########################################################
        # output_audio=True
    )

    # need to multiple by 32768 to get the correct amplitude
    # audio_values[0] = audio_values * 32768.0
    audio_values = audio_values[0].to(torch.float32).cpu().numpy()
    return audio_values

    # return (
    #     audio_values[0]
    #         .to(torch.float32)
    #         .cpu()
    #         .numpy()
    # )


def main(
    target: Path,
    checkpoint_min: int = 0,
    checkpoint_max: int | None = None,
    speakers: str | None = None,
) -> None:
    if not target.exists():
        raise FileNotFoundError(f"Target directory {target} does not exist.")

    samples = target / "samples"

    if not samples.exists():
        samples.mkdir(parents=True, exist_ok=True)

    checkpoint_max = checkpoint_max or int(1e9) # Some arbitrarily large number

    # We expect a series of checkpoint directories named like "checkpoint-100", "checkpoint-200", etc.
    checkpoints = sorted(
        [d for d in target.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda x: int(x.name.split("-")[1])
    )

    generate_checkpoints: list[Path] = []

    for checkpoint in checkpoints:
        checkpoint_int = int(checkpoint.name.split("-")[1])

        if checkpoint_int >= checkpoint_min and checkpoint_int <= checkpoint_max:
            generate_checkpoints.append(checkpoint)

    if not generate_checkpoints:
        raise ValueError("No checkpoints found in the specified range.")

    lspeakers = None

    if speakers is not None:
        lspeakers = speakers.split(",")
        lspeakers = [speaker.strip() for speaker in lspeakers if speaker.strip()]

        print(f"Speakers: {lspeakers}")

    processor: CsmProcessor = cast(CsmProcessor, AutoProcessor.from_pretrained("sesame/csm-1b"))

    for checkpoint in generate_checkpoints:
        model = CsmForConditionalGeneration.from_pretrained(str(checkpoint), device_map='cuda')
        # model = CsmForConditionalGeneration.from_pretrained("sesame/csm-1b", device_map='cuda')
        # # load the checkpoint into the model
        # model.load_adapter(checkpoint)

        if lspeakers is None:
            lspeakers = [0]

        for speaker in lspeakers:
            speaker = int(speaker)
            print(f"Generating audio for speaker {speaker} with model {checkpoint.name}")

            for i, text in enumerate(tts_test_sentences):
                audio = generate_audio(model, processor, text, speaker_id=speaker)

                output_file = samples / f"{checkpoint.name}_speaker_{speaker}_sample_{i}.wav"
                sf.write(output_file, audio, 24000)

                print(f"Saved audio to {output_file}")


if __name__ == "__main__":
    # typer.run(main)
    main(
        Path('csm-vctk-base-ft-1e-5e-05lr-1bs-32g-cosine'),
        checkpoint_min=1100,
    )
    # main(Path("csm-expresso-qlora-ft-3e-5e-05lr-1bs-16g-linear"))
