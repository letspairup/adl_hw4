from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, Trainer, TrainingArguments

from .base_vlm import BaseVLM
from .data import CaptionDataset, MultiChoiceQADataset

# Processor to tokenize captions and append EOS
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load(model_name: str = "clip_model"):
    from peft import PeftModel

    model_path = Path(__file__).parent / model_name

    vlm = BaseVLM()
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model

    clip = CLIP(vision_encoder, text_encoder)
    clip = PeftModel.from_pretrained(clip, model_path).to(device)
    clip.model.load_pretrained(model_path)
    clip.model.eval()

    return clip


def clip_data_collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_length = max(f["input_ids"].shape[0] for f in features)
    def pad_tensor(tensor, pad_value):
        return torch.cat([tensor, torch.full((max_length - tensor.shape[0],), pad_value, dtype=tensor.dtype)])

    input_ids = torch.stack([pad_tensor(f["input_ids"], processor.tokenizer.eos_token_id) for f in features])
    attention_mask = torch.stack([pad_tensor(f["attention_mask"], 0) for f in features])
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    labels = torch.stack([pad_tensor(f["labels"], -100) for f in features])

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
            tv.transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        img = Image.open(item["image_path"]).convert("RGB")
        pixel_values = self.image_processor(img)

        text = item["caption"] + self.processor.tokenizer.eos_token
        txt_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        input_ids = txt_inputs["input_ids"].squeeze(0).long()
        attention_mask = txt_inputs["attention_mask"].squeeze(0)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
        }


class CLIP(nn.Module):
    def __init__(self, vision_encoder: nn.Module, text_encoder: nn.Module, proj_dim: int = 64, temperature: float = 0.07):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

        vision_hidden_dim = vision_encoder.config.hidden_size
        text_hidden_dim = text_encoder.config.hidden_size

        self.image_proj = nn.Linear(vision_hidden_dim, proj_dim)
        self.text_proj = nn.Linear(text_hidden_dim, proj_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / temperature))

    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None, **kwargs):
        v_out = self.vision_encoder(pixel_values=pixel_values)
        if getattr(v_out, "pooler_output", None) is not None:
            v_feat = v_out.pooler_output
        elif getattr(v_out, "image_embeds", None) is not None:
            v_feat = v_out.image_embeds
        else:
            v_feat = v_out.last_hidden_state[:, 0, :]

        t_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        if getattr(t_out, "pooler_output", None) is not None:
            t_feat = t_out.pooler_output
        else:
            hidden = t_out.last_hidden_state
            if attention_mask is None:
                t_feat = hidden[:, -1, :]
            else:
                lengths = attention_mask.sum(dim=1) - 1
                gather_idx = lengths.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
                t_feat = hidden.gather(1, gather_idx).squeeze(1)

        img_emb = F.normalize(self.image_proj(v_feat), dim=-1)
        txt_emb = F.normalize(self.text_proj(t_feat), dim=-1)

        return img_emb, txt_emb, self.logit_scale

    def save_pretrained(self, save_directory: str, **kwargs):
        state = {n: p.data for n, p in self.named_parameters() if "vision_encoder." not in n and "text_encoder." not in n}
        torch.save(state, Path(save_directory) / "additional_weights.pt")

    def load_pretrained(self, load_directory: str, **kwargs):
        path = Path(load_directory) / "additional_weights.pt"
        if path.exists():
            state = torch.load(path, map_location="cpu")
            for n, p in self.named_parameters():
                if n in state and "vision_encoder." not in n and "text_encoder." not in n:
                    p.data = state[n]

    def set_trainable_parameters(self):
        for n, p in self.named_parameters():
            if "vision_encoder." in n or "text_encoder." in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

    def gradient_checkpointing_enable(self, **kwargs):
        self.vision_encoder.gradient_checkpointing_enable(**kwargs)
        self.text_encoder.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self):
        def hook(_module, _inp, out): out.requires_grad_(True)
        if hasattr(self.vision_encoder, "embeddings"):
            self.vision_encoder.embeddings.register_forward_hook(hook)
        if hasattr(self.text_encoder, "get_input_embeddings"):
            self.text_encoder.get_input_embeddings().register_forward_hook(hook)


def compute_clip_loss(outputs, labels, num_items_in_batch=None):
    image_embeds, text_embeds, logit_scale = outputs
    logits_i = (image_embeds @ text_embeds.T) * torch.exp(logit_scale)
    logits_t = logits_i.T

    bsz = logits_i.size(0)
    target = torch.arange(bsz, device=logits_i.device)
    return 0.5 * (F.cross_entropy(logits_i, target) + F.cross_entropy(logits_t, target))


def get_target_modules_for_lora(model: nn.Module) -> list[str]:
    return [n for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and ("vision_encoder" in n or "text_encoder" in n) and "projection" not in n]



def train(
        data_dir: Path | None = None,
        output_dir: str = "clip",
        num_train_epochs: float = 1,
        per_device_train_batch_size: int = 1024,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 5e-4,
        num_workers: int = 16,
):
    vlm = BaseVLM()

    output_dir = Path(__file__).parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TensorBoard writer
    tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Initialize model and processor
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model
    model = CLIP(vision_encoder, text_encoder).to(device).bfloat16()
    model.set_trainable_parameters()

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.0,
        # target_modules="all-linear",
        target_modules=get_target_modules_for_lora(model),
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.to(device)
    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # load dataset
    train_dataset = CaptionDataset("train", data_dir)
    train_dataset = CaptionDatasetForTraining(train_dataset, processor)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
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

    # save model
    trainer.save_model(output_dir)
    model.model.save_pretrained(output_dir)

    writer.close()

    return model, processor


def demo_train():
    train(
        train_dataset_name="train_demo",
        output_dir="demo_clip",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        num_workers=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-8,
    )


def test(ckpt_path: str, val_dataset: str = "valid_grader"):
    import tqdm

    testset = MultiChoiceQADataset(val_dataset)

    clip = load(ckpt_path)
    clip = clip.model.to(device)

    image_processor = tv.transforms.Compose(
        [
            tv.transforms.Resize(192),
            tv.transforms.CenterCrop(192),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    correct_count = 0
    total_count = 0

    for pair in tqdm.tqdm(testset):
        image = Image.open(pair["image_path"]).convert("RGB")
        pixel_values = image_processor(image).unsqueeze(0).to(device).bfloat16()
        text_inputs = processor(
            text=[s + processor.tokenizer.eos_token for s in pair["candidates"]],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = text_inputs["input_ids"].long().to(device)
        attention_mask = text_inputs["attention_mask"].to(device)
        vision_feature, text_feature, _ = clip(pixel_values, input_ids, attention_mask)
        prediction = torch.matmul(vision_feature, text_feature.T).argmax(dim=-1)
        if prediction == pair["correct_index"]:
            correct_count += 1
        total_count += 1

    print(f"Accuracy: {correct_count / total_count}")


def main():
    from fire import Fire

    Fire({"train": train, "test": test})


if __name__ == "__main__":
    main()
