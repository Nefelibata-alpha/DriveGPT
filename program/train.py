"""
DriveGPT 训练脚本
主入口程序
"""

# ========== 标准库 ==========
import os

# ========== Transformers ==========
from transformers import AutoProcessor

# ========== 配置、数据集、模型 ==========
from config import DriveConfig
from dataset import BDDXDataset
from model import DriveGPTModel, create_trainer


def main():
    """主函数"""
    config = DriveConfig()

    print(f"{'='*60}")
    print("DriveGPT with Qwen3-VL-Instruct")
    print(f"Batch size: {config.batch_size}")
    print(f"LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"{'='*60}")
    print("\n✓ 从头开始训练")
    
    # 加载 processor
    print("\n加载 Processor...")
    processor = AutoProcessor.from_pretrained(
        config.model_name,
        trust_remote_code=True
    )
    
    # 创建数据集
    print("创建数据集...")
    train_dataset = BDDXDataset(config, processor)

    # 创建模型
    print("\n创建模型...")
    model = DriveGPTModel(config)
    
    # 创建训练器
    print("创建训练器...")
    trainer = create_trainer(model, processor, train_dataset, config)
    
    # 训练
    print(f"\n{'='*60}")
    print("开始训练")
    print(f"{'='*60}\n")
    
    trainer.train()
    
    # 保存最终模型
    print("\n保存最终模型...")
    os.makedirs(config.output_dir, exist_ok=True)
    model._unshare_lm_head_weights()
    model.model.save_pretrained(config.output_dir)
    processor.save_pretrained(config.output_dir)
    print(f"✓ 模型已保存到: {config.output_dir}")
    
    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
