"""
DriveGPT with Qwen3-VL-Instruct (图片数据集版本)
适配 BDD-X 格式：{id}_{idx}.png 的 8 帧图片

模块说明：
- config.py: 配置类 DriveConfig
- dataset.py: 数据集类 BDDXDataset 和数据整理器 DriveDataCollator
- model.py: 模型类 DriveGPTModel 和训练工具
- train.py: 训练入口主程序

使用方式：
    python train.py
"""

# 导出主要组件，方便其他脚本导入
from config import DriveConfig
from dataset import BDDXDataset, DriveDataCollator
from model import DriveGPT4Model, create_trainer

__all__ = [
    'DriveConfig',
    'BDDXDataset',
    'DriveDataCollator',
    'DriveGPTModel',
    'create_trainer',
]
