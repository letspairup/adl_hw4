from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision as tv
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoProcessor,
    AutoModel,
    AutoTokenizer,
    CLIPVisionModel,
    CLIPImageProcessor,
    Trainer,
    TrainingArguments,
)

from .data import CaptionDataset, MultiChoiceQADataset

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load(model_name: str = "clip_model"):
    model_path = Path(__file__).parent / model_name

    vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = AutoModel.from_pretrained("EleutherAI/gpt-neox-20b")

    clip = CLIP(vision_encoder, text_encoder).to(device)
    clip = PeftModel.from_pretrained(clip, model_path, local_files_only=True)
    clip.model.eval()
    return clip


def clip_data_collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_length = max(f["input_ids"].shape[0] for f in features)

    def pad_tensor(tensor, pad_value):
        return torch.cat([tensor, torch.full((max_length - tensor.shape[0],), pad_value, dtype=tensor.dtype)])

    input_ids = torch.stack([pad_tensor(f["input_ids"], pad_value=processor.tokenizer.eos_token_id) for f in features])
    attention_mask = torch.stack([pad_tensor(f["attention_mask"], pad_value=0) for f in features])
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    labels = torch.stack([pad_tensor(f["labels"], pad_value=-100) for f in features])

    return {
        "input_ids": input_ids.long(),
        "attention_mask": attention_mask.long(),
        "pixel_values": pixel_values.float(),
        "labels": labels.long(),
    }


class CaptionDatasetForTraining(Dataset):
    def __init__(self, dataset: CaptionDataset, processor: AutoProcessor):
        self.dataset = dataset
        self.image_processor = tv.transforms.Compose([
            tv.transforms.Resize(192),
            tv.transforms.RandomResizedCrop(192, scale=(0.5, 1.0)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        pixel_values = self.image_processor(image)
        text = item["caption"] + self.processor.tokenizer.eos_token
        text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        input_ids = text_inputs["input_ids"].squeeze(0).long()
        attention_mask = text_inputs["attention_mask"].squeeze(0)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
        }


class CLIP(nn.Module):
    def __init__(self, vit_model, text_model):
        super().__init__()
        self.vision_model = vit_model
        self.image_proj = nn.Linear(vit_model.config.hidden_size, 256)

        self.text_model = text_model
        #self.text_proj = nn.Linear(text_model.config.hidden_size, 256)
        self.text_proj = nn.Linear(getattr(self.text_model, "hidden_size", 4096), 256)

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    def forward(self, pixel_values, input_ids, attention_mask):
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs.pooler_output
        image_embeds = self.image_proj(image_embeds)

        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = text_outputs.last_hidden_state[:, 0, :]
        text_embeds = self.text_proj(text_embeds)

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        logits = torch.matmul(image_embeds, text_embeds.T)
        return logits


def compute_clip_loss(similarity_logits: torch.Tensor) -> torch.Tensor:
    batch_size = similarity_logits.shape[0]
    labels = torch.arange(batch_size).to(similarity_logits.device)
    loss_i2t = nn.functional.cross_entropy(similarity_logits, labels)
    loss_t2i = nn.functional.cross_entropy(similarity_logits.T, labels)
    return (loss_i2t + loss_t2i) / 2


def get_target_modules_for_lora(model: nn.Module) -> list[str]:
    return [name for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and ("vision_model" in name or "text_model" in name)]


def train(
        data_dir: Path | None = None,
        output_dir: str = "clip",
        num_train_epochs: float = 1,
        per_device_train_batch_size: int = 1024,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 5e-4,
        num_workers: int = 16,
):

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    output_dir = Path(__file__).parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = AutoModel.from_pretrained("EleutherAI/gpt-neox-20b")
    model = CLIP(vision_encoder, text_encoder).to(device)

    model.set_trainable_parameters = lambda: None  # dummy placeholder

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=get_target_modules_for_lora(model),
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model = model.bfloat16()
    model.print_trainable_parameters()
    model.train()

    train_dataset = CaptionDataset("train", data_dir)
    train_dataset = CaptionDatasetForTraining(train_dataset, processor)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        bf16=True if device == "cuda" else False,
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        label_names=["labels"],
        dataloader_num_workers=num_workers,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=clip_data_collator,
        compute_loss_func=compute_clip_loss,
    )

    trainer.train()
    trainer.save_model(output_dir)
    writer.close()
    return model, processor


def test(ckpt_path: str, val_dataset: str = "valid_grader"):
    import tqdm
    testset = MultiChoiceQADataset(val_dataset)
    clip = load(ckpt_path).model.to(device)
    image_processor = tv.transforms.Compose([
        tv.transforms.Resize(192),
        tv.transforms.CenterCrop(192),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    correct_count = 0
    for pair in tqdm.tqdm(testset):
        image = Image.open(pair["image_path"]).convert("RGB")
        pixel_values = image_processor(image).unsqueeze(0).to(device).bfloat16()
        text_inputs = processor(
            text=[s + processor.tokenizer.eos_token for s in pair["candidates"]],
            return_tensors="pt", padding=True, truncation=True
        )
        input_ids = text_inputs["input_ids"].long().to(device)
        attention_mask = text_inputs["attention_mask"].to(device)
        logits = clip(pixel_values, input_ids, attention_mask)
        prediction = torch.argmax(logits, dim=-1).item()
        if prediction == pair["correct_index"]:
            correct_count += 1
    print(f"Accuracy: {correct_count / len(testset):.4f}")


def main():
    from fire import Fire
    Fire({"train": train, "test": test})


if __name__ == "__main__":
    main()
