from pathlib import Path
from typing import Any
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
import torchvision as tv
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import gc
from transformers import (
    AutoProcessor,
    AutoModel,
    AutoTokenizer,
    CLIPVisionModel,
    Trainer,
    TrainingArguments,
)

from .data import CaptionDataset, MultiChoiceQADataset

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from safetensors.torch import load_file

from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoModel, CLIPVisionModel
from peft import LoraConfig, TaskType, get_peft_model

def load(model_name: str = "clip") -> nn.Module:
    model_path = Path(__file__).parent / model_name

    checkpoints = sorted([d for d in model_path.glob("checkpoint-*") if d.is_dir()])
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint directories found under {model_path}")
    latest_ckpt = checkpoints[-1]

    vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
    text_encoder = AutoModel.from_pretrained("EleutherAI/gpt-neo-125M")
    clip = CLIP(vision_encoder, text_encoder)

    clip = get_peft_model(clip, LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=True,
        r=8,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=get_target_modules_for_lora(clip),
        bias="none",
    ))

    adapter_path = latest_ckpt / "adapter_model.safetensors"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Missing adapter model at: {adapter_path}")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = load_file(str(adapter_path), device=device_str)
    clip.load_state_dict(state_dict, strict=False)

    clip.eval()
    return clip.to(torch.device(device_str))

def clip_data_collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_length = max(f["input_ids"].shape[0] for f in features)
    def pad_tensor(tensor, pad_value):
        return torch.cat([tensor, torch.full((max_length - tensor.shape[0],), pad_value, dtype=tensor.dtype)])
    input_ids = torch.stack([pad_tensor(f["input_ids"], processor.tokenizer.eos_token_id) for f in features])
    attention_mask = torch.stack([pad_tensor(f["attention_mask"], 0) for f in features])
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    labels = torch.stack([pad_tensor(f["labels"], -100) for f in features])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "labels": labels,
    }

class CaptionDatasetForTraining(Dataset):
    def __init__(self, dataset: CaptionDataset, processor: AutoProcessor):
        self.dataset = dataset
        self.processor = processor
        self.image_processor = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        pixel_values = self.image_processor(image)
        text = item["caption"] + self.processor.tokenizer.eos_token
        text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "labels": text_inputs["input_ids"].squeeze(0),
        }

class CLIP(nn.Module):
    def __init__(self, vision_encoder: nn.Module, text_encoder: nn.Module, proj_dim: int = 256, temperature: float = 0.07):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

        vision_hidden_size = vision_encoder.config.hidden_size
        text_hidden_size = text_encoder.config.hidden_size

        self.image_proj = nn.Linear(vision_hidden_size, proj_dim)
        self.text_proj = nn.Linear(text_hidden_size, proj_dim)

        # Learnable log temperature
        self.logit_scale = nn.Parameter(torch.tensor(temperature).log())

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode image and return pooled features of shape (B, hidden_size)"""
        output = self.vision_encoder(pixel_values)
        pooled = output.last_hidden_state.mean(dim=1)  # Average pooling
        return pooled

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text and extract first EOS token hidden state"""
        eos_token_id = self.tokenizer.eos_token_id
        output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = output.last_hidden_state  # shape: (B, seq_len, hidden_size)

        # Find first EOS token position in each sequence
        eos_positions = (input_ids == eos_token_id).float()
        index = eos_positions.argmax(dim=1)  # shape: (B,)

        # Gather the hidden state at the EOS position
        batch_size, hidden_size = input_ids.size(0), last_hidden.size(2)
        idx = index.view(-1, 1, 1).expand(-1, 1, hidden_size)  # shape: (B, 1, H)
        eos_hidden = last_hidden.gather(1, idx).squeeze(1)     # shape: (B, H)

        return eos_hidden

    def forward(self, **kwargs):
        pixel_values = kwargs["pixel_values"]
        input_ids = kwargs["input_ids"]
        attention_mask = kwargs.get("attention_mask", None)
        labels = kwargs.get("labels", None)

        image_feat = self.encode_image(pixel_values)
        text_feat = self.encode_text(input_ids, attention_mask)

        image_embeds = self.image_proj(image_feat)
        text_embeds = self.text_proj(text_feat)

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        logits = torch.matmul(image_embeds, text_embeds.T) * self.logit_scale.exp()
        return image_embeds, text_embeds, logits


def compute_clip_loss(
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        labels: torch.Tensor,
        num_items_in_batch: int | None = None,
) -> torch.Tensor:
    _, _, logits = outputs
    batch_size = logits.size(0)
    target = torch.arange(batch_size, device=logits.device)

    # Image-to-text and text-to-image loss
    loss_i2t = nn.functional.cross_entropy(logits, target)       # row-wise
    loss_t2i = nn.functional.cross_entropy(logits.T, target)     # col-wise

    return (loss_i2t + loss_t2i) / 2

def get_target_modules_for_lora(model: nn.Module) -> list[str]:
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear) and ("vision_model" in name or "text_model" in name)]

def train(data_dir: Path | None = None, output_dir: str = "clip", num_train_epochs: float = 1, per_device_train_batch_size: int = 8, gradient_accumulation_steps: int = 1, learning_rate: float = 5e-4, num_workers: int = 4):

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    output_dir = Path(__file__).parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=output_dir / "tensorboard")

    vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16",torch_dtype=torch.float16)
    text_encoder = AutoModel.from_pretrained("EleutherAI/gpt-neox-20b",torch_dtype=torch.float16)

    model = CLIP(vision_encoder, text_encoder)

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

    print("Moving model to GPU...")
    model = model.train().to(device)

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
        fp16=True,
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
    clip = load(ckpt_path)

    image_processor = tv.transforms.Compose([
        tv.transforms.Resize(224),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    correct = 0
    for pair in tqdm.tqdm(testset):
        image = Image.open(pair["image_path"]).convert("RGB")
        pixel_values = image_processor(image).unsqueeze(0).to(device)

        text_inputs = processor(
            text=[s + processor.tokenizer.eos_token for s in pair["candidates"]],
            return_tensors="pt", padding=True, truncation=True
        )
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        # ✅ FIXED: Call with keyword arguments
        _, _, logits = clip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        if logits.argmax(dim=-1).item() == pair["correct_index"]:
            correct += 1


    print(f"Accuracy: {correct / len(testset):.4f}")


def main():
    from fire import Fire
    Fire({"train": train, "test": test})

if __name__ == "__main__":
    main()
