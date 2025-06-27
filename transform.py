from datasets import Audio, Dataset
import torch
from transformers import CsmProcessor


TARGET_SAMPLING_RATE = 24000

 # this should be the max length of audio
MAX_TEXT_LENGTH = 256

# max input_values length of the whole dataset
MAX_AUDIO_LENGTH = 240001  # This is the max input_values length of the whole dataset


def get_example_length(
    example,
    processor: CsmProcessor,
) -> dict | None:
    conversation = [
        {
            "role": str(example['source']),
            "content": [
                {"type": "text", "text": example["text"]},
                {"type": "audio", "path": example["audio"]["array"]},
            ],
        }
    ]

    model_inputs = processor.apply_chat_template( # type: ignore
        conversation,
        tokenize=True,
        return_dict=True,
        output_labels=True,
        common_kwargs={"return_tensors": "pt"},
    )

    required_keys = ["input_ids", "attention_mask", "labels", "input_values", "input_values_cutoffs"]

    ret = {}

    # print(model_inputs.keys())
    for key in required_keys:
        if key not in model_inputs:
            raise KeyError(f"Required key '{key}' not found in processor output for example.")

        value = model_inputs[key][0]
        ret[key] = value

    return {
        'input_ids_length': ret["input_ids"].shape[0],
        'attention_mask_length': ret["attention_mask"].shape[0],
        'labels_length': ret["labels"].shape[0],
        'input_values_length': ret['input_values'].squeeze(0).shape[0]
    }


def process_example(
    example,
    processor: CsmProcessor,
    max_audio_length: int,
    max_text_length: int = MAX_TEXT_LENGTH,
) -> dict | None:
    conversation = [
        {
            "role": str(example['source']),
            "content": [
                {"type": "text", "text": example["text"]},
                {"type": "audio", "path": example["audio"]["array"]},
            ],
        }
    ]

    try:
        model_inputs = processor.apply_chat_template( # type: ignore
            conversation,
            tokenize=True,
            return_dict=True,
            output_labels=True,
            text_kwargs={
                "padding": "max_length", # pad to the max_length
                "max_length": max_text_length, # this should be the max length of audio
                "pad_to_multiple_of": 8,
                "padding_side": "right",
            },
            audio_kwargs={
                "sampling_rate": TARGET_SAMPLING_RATE,
                "max_length": max_audio_length, # max input_values length of the whole dataset
                "padding": "max_length",
            },
            common_kwargs={"return_tensors": "pt"},
        )
    except Exception as e:
        print(f"Error processing example with text '{example['text'][:50]}...': {e}")
        return None

    # TODO: This might not be necessary - this is my last ditch effort to fix the audio ending issue
    # Allegedly, this was fixed by the latest transformers version. Maybe it doesn't fix my issue.
    # https://huggingface.co/eustlb/csm-1b/discussions/2
    #
    # Hotfix: Audio seems to not end when it should. add audio eos token to trainable tokens so it can fix this
    # the audio eos token should be 0, but not sure if this work as intended. using 128003 gave out of vocab error (as its only checking the 2051 audio tokens)
    model_inputs["labels"][model_inputs["input_ids"] == 128003] = 0

    required_keys = ["input_ids", "attention_mask", "labels", "input_values", "input_values_cutoffs"]
    processed_example = {}

    # print(model_inputs.keys())
    for key in required_keys:
        if key not in model_inputs:
            print(f"Warning: Required key '{key}' not found in processor output for example.")
            return None

        value = model_inputs[key][0]
        processed_example[key] = value

    # Final check (optional but good)
    if not all(isinstance(processed_example[key], torch.Tensor) for key in processed_example):
         print(f"Error: Not all required keys are tensors in final processed example. Keys: {list(processed_example.keys())}")
         return None

    return processed_example


def transform_dataset(
    dataset: Dataset,
    processor: CsmProcessor,
 ) -> Dataset:
    if 'source' not in dataset.column_names:
        new_column = ["0"] * len(dataset)
        dataset = dataset.add_column("source", new_column) # type: ignore

    dataset = dataset.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLING_RATE))
    input_values_lengths = []
    input_ids_lengths = []

    for d in dataset:
        lengths = get_example_length(d, processor)

        if lengths is not None:
            input_values_lengths.append(lengths['input_values_length'])
            input_ids_lengths.append(lengths['input_ids_length'])

    max_audio_length = max(input_values_lengths)
    max_text_length = max(input_ids_lengths)

    dataset = dataset.map(
        lambda example: process_example(example, processor, max_audio_length, max_text_length),
        remove_columns=dataset.column_names,
        desc="Processing dataset",
    )

    return dataset
