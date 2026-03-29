"""
DriveGPT 配置文件
包含所有训练参数和路径配置
"""

from dataclasses import dataclass


# ==================== 配置类 ====================
@dataclass
class DriveConfig:
    """DriveGPT 训练配置类"""

    # ========== 路径配置 ==========
    output_dir: str = r"E:\DriveGPT\output"
    model_name: str = r"E:\DriveGPT\model\Qwen3-VL-2B-Instruct"
    image_root: str = r"E:\DriveGPT\image"
    json_path: str = r"E:\DriveGPT\data\BDD_X_training_label_zh.json"

    # ========== 训练参数 ==========
    num_epochs: int = 1           # 可选: 1, 2, 3, 5 (训练轮数)
    batch_size: int = 2           # 可选: 1, 2, 4, 8 (每批次样本数，根据显存调整)
    gradient_accumulation_steps: int = 4  # 可选: 1, 2, 4, 8 (梯度累积步数，模拟大batch)
    learning_rate: float = 5e-5   # 可选: 1e-5, 2e-5, 5e-5, 1e-4 (学习率)
    warmup_ratio: float = 0.1     # 可选: 0.03, 0.05, 0.1, 0.2 (预热比例)
    max_length: int = 1024        # 可选: 512, 1024, 2048 (最大序列长度)

    # ========== LoRA参数 ==========
    lora_r: int = 128             # 可选: 8, 16, 32, 64, 128, 256 (LoRA秩，越大容量越大)
    lora_alpha: int = 256         # 可选: 16, 32, 64, 128, 256, 512 (LoRA缩放，通常=2*r)
    lora_dropout: float = 0.05    # 可选: 0.0, 0.05, 0.1, 0.2 (Dropout率)

    # ========== Trainer参数 ==========
    weight_decay: float = 0.01           # 可选: 0.0, 0.01, 0.1 (权重衰减)
    lr_scheduler_type: str = "cosine"    # 可选: "linear", "cosine", "constant" (学习率调度器)
    fp16: bool = False                   # 可选: True, False (是否使用FP16)
    bf16: bool = True                    # 可选: True, False (是否使用BF16)
    dataloader_num_workers: int = 4      # 可选: 0, 2, 4, 8 (数据加载进程数)
    remove_unused_columns: bool = False  # 可选: True, False (是否移除未使用列)
    gradient_checkpointing: bool = False # 可选: True, False (是否使用梯度检查点)
    report_to: str = "none"              # 可选: "none", "wandb", "tensorboard" (报告目标)
    dataloader_pin_memory: bool = True   # 可选: True, False (是否使用pin_memory)
