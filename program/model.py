"""
DriveGPT 模型定义
包含模型类和训练工具
"""

# ========== 标准库 ==========
import os

# ========== PyTorch ==========
import torch

# ========== Transformers ==========
from transformers import (
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
)

# ========== PEFT (LoRA) ==========
from peft import LoraConfig, get_peft_model, TaskType

# ========== 配置和数据集 ==========
from config import DriveConfig
from dataset import DriveDataCollator


# ==================== 模型 ====================
class DriveGPTModel(torch.nn.Module):
    """DriveGPT 模型 - Qwen3-VL 图片版本"""
    
    def __init__(self, config: DriveConfig):
        super().__init__()
        self.config = config
        
        self.model = AutoModelForImageTextToText.from_pretrained(
            config.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        self.model.config.use_cache = False

        self._setup_lora(config)
        self._unshare_lm_head_weights()
    
    def _unshare_lm_head_weights(self):
        """断开 lm_head 和 embed_tokens 的权重共享"""
        try:
            if hasattr(self.model, 'base_model'):
                base_model = self.model.base_model.model
            else:
                base_model = self.model
            
            if hasattr(base_model, 'lm_head') and hasattr(base_model.model, 'embed_tokens'):
                lm_head_weight = base_model.lm_head.weight
                embed_weight = base_model.model.embed_tokens.weight
                if lm_head_weight.data_ptr() == embed_weight.data_ptr():
                    print("检测到共享权重，正在断开...")
                    base_model.lm_head.weight = torch.nn.Parameter(lm_head_weight.clone())
                    print("✓ 已断开 lm_head 权重共享")
        except Exception as e:
            print(f"断开权重共享时出错: {e}")
    
    def _setup_lora(self, config: DriveConfig):
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def forward(self, input_ids, attention_mask, labels=None, pixel_values=None, image_grid_thw=None, **kwargs):
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
        if pixel_values is not None and pixel_values.numel() > 0:
            model_inputs['pixel_values'] = pixel_values
        if image_grid_thw is not None and image_grid_thw.numel() > 0:
            model_inputs['image_grid_thw'] = image_grid_thw
        
        outputs = self.model(**model_inputs)
        return {'loss': outputs.loss, 'logits': outputs.logits}


def create_trainer(model, processor, train_dataset, config: DriveConfig):
    """创建训练器"""
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        save_strategy="no",
        fp16=config.fp16,
        bf16=config.bf16,
        dataloader_num_workers=config.dataloader_num_workers,
        remove_unused_columns=config.remove_unused_columns,
        gradient_checkpointing=config.gradient_checkpointing,
        report_to=config.report_to,
        dataloader_pin_memory=config.dataloader_pin_memory,
    )

    # 在保存前断开权重共享
    model._unshare_lm_head_weights()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DriveDataCollator(processor),
    )
    
    return trainer
