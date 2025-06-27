import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from typing import cast

from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from transformers import AutoProcessor, CsmForConditionalGeneration, CsmProcessor
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
import typer
import wandb

from transform import transform_dataset


load_dotenv()


def main(
    epochs: int = 1,
    learning_rate: float = 1e-4,
    warnup_ratio: float = 0.05,
    batch_size: int = 1, # TODO: Need to figure out the collator on this
    gradient_accumulation_steps: int = 32,
    scheduler: str = "linear",
    # r: int = 32,
    # lora_alpha_multiplier: int = 1,
) -> None:
    model_id = "sesame/csm-1b"
    device = 'cuda'
    name = f"csm-glimpsed-base-ft-{epochs}e-{learning_rate}lr-{batch_size}bs-{gradient_accumulation_steps}g-{scheduler}"

    processor: CsmProcessor = cast(CsmProcessor, AutoProcessor.from_pretrained("sesame/csm-1b"))
    model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)

    if not isinstance(model, CsmForConditionalGeneration):
        raise TypeError(f"Expected model to be of type CsmForConditionalGeneration, got {type(model)}")

    # https://huggingface.co/docs/transformers/main/en/model_doc/csm#training
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # https://huggingface.co/docs/transformers/main/en/model_doc/csm#training
    model.train()
    model.codec_model.eval()


    # VCTK Dataset
    # dataset = load_dataset(
    #     "audiofolder",
    #     data_dir="../dataset-redux/vctk/dataset",
    #     split="train",
    # )
    # use mic_id == mic2
    # dataset = dataset.filter(lambda x: x["mic_id"] == "mic2")


    # Suzy Dataset
    glimpsed = load_dataset(
        "audiofolder",
        data_dir="../dataset-redux/alt_dataset/glimpsed/normalized",
        split="train",
    )

    # defiant = load_dataset(
    #     "audiofolder",
    #     data_dir="../dataset-redux/dataset/defiant",
    #     split="train",
    # )

    # valk = load_dataset(
    #     "audiofolder",
    #     data_dir="../dataset-redux/dataset/valk",
    #     split="train",
    # )

    # Combine the datasets
    # dataset = concatenate_datasets([glimpsed, defiant, valk])
    dataset = glimpsed

    # filter out nisqa_mos	nisqa_noisiness	nisqa_discontinuity	nisqa_coloration if < 3.5
    # file_name,text,length_ms,p808_mos,mos_sig,mos_bak,mos_ovr,nisqa_mos,nisqa_noisiness,nisqa_discontinuity,nisqa_coloration,nisqa_loudness,ce,cu,pc,pq,sr_score,sr_prediction
    dataset = dataset.filter(lambda x: x["nisqa_mos"] >= 3.5)
    dataset = dataset.filter(lambda x: x["nisqa_noisiness"] >= 3.8)
    dataset = dataset.filter(lambda x: x["nisqa_discontinuity"] >= 3.5)
    dataset = dataset.filter(lambda x: x["nisqa_coloration"] >= 3.0)

    # filter out any samples that have a sr_prediction of "False"
    dataset = dataset.filter(lambda x: x["sr_prediction"] == True)


    # Filter anything larger than 30 seconds and smaller than 1 second
    dataset = dataset.filter(lambda x: len(x["audio"]["array"]) / x["audio"]["sampling_rate"] <= 30.0)
    dataset = dataset.filter(lambda x: len(x["audio"]["array"]) / x["audio"]["sampling_rate"] >= 1.2)


    # Finally, transform the dataset
    dataset: Dataset = transform_dataset(
        dataset=dataset,
        processor=processor,
    )

    print(f"########## Dataset size: {len(dataset)} ##########")


    wandb.init(
        project="csm-glimpsed-base-ft",
        name=name,
        config={
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum_steps": gradient_accumulation_steps,
            "warmup_ratio": warnup_ratio,
            "lr_scheduler_type": scheduler,
        }
    )

    # Get rid of the processor to save memory
    del processor

    # Empty the torch cache and synchronize CUDA
    import torch

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            warmup_ratio=warnup_ratio,
            # bf16=True,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            logging_steps=1,
            save_steps=500,
            # optim="adamw_8bit",

            # https://huggingface.co/docs/bitsandbytes/v0.43.0/en/optimizers#paged-optimizers
            optim="paged_adamw_8bit",
            weight_decay=0.01, # Turn this on if overfitting
            lr_scheduler_type=scheduler,
            # max_grad_norm=1.0,
            seed=3407,
            output_dir=name,
            report_to="wandb",
        ),
    )

    trainer.train()


if __name__ == "__main__":
    typer.run(main)
