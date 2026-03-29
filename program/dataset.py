"""
BDD-X 数据集类
处理图片序列和对话数据
"""

# ========== 标准库 ==========
import json
import re
from pathlib import Path
from typing import List, Dict

# ========== 第三方库 ==========
from PIL import Image

# ========== PyTorch ==========
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# ========== Transformers ==========
from transformers import AutoProcessor

# ========== 配置 ==========
from config import DriveConfig


# ==================== 数据集 ====================
class BDDXDataset(Dataset):
    """BDD-X 数据集 - 图片版本 (8帧图片序列)"""
    
    def __init__(self, config: DriveConfig, processor: AutoProcessor):
        with open(config.json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.image_root = Path(config.image_root)
        self.processor = processor
    
    def _get_frame_paths(self, sample: Dict) -> List[str]:
        """获取样本的8帧图片路径"""
        sample_id = sample.get('id', '')
        idx_list = [0, 3, 7, 11, 15, 19, 23, 27]
        return [
            str(self.image_root / f"{sample_id}_{idx}.png")
            for idx in idx_list
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict:
        sample = self.data[idx]
        frame_paths = self._get_frame_paths(sample)
        messages = self._build_conversation(sample)
        images = [Image.open(p).convert('RGB') for p in frame_paths]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
            padding=True,
        )

        # 处理 batch 维度
        processed_inputs = {}
        for k, v in inputs.items():
            if k in ['pixel_values', 'image_grid_thw', 'pixel_values_videos', 'video_grid_thw']:
                processed_inputs[k] = v
            elif v.dim() >= 1 and v.shape[0] == 1:
                processed_inputs[k] = v.squeeze(0)
            else:
                processed_inputs[k] = v
        inputs = processed_inputs

        inputs['labels'] = inputs['input_ids'].clone()
        return inputs
    
    def _build_conversation(self, sample: Dict) -> List[Dict]:

        conversations = sample.get('conversations', [])
        messages = []
        
        # 遍历所有问答对
        for i in range(0, len(conversations) - 1, 2):
            question = conversations[i].get('value', '')
            answer = conversations[i + 1].get('value', '')
            
            # 移除 <视频> 标记
            question = question.replace('\n<视频>', '').replace('<视频>', '')
            
            # 第一轮包含8张图片，后续轮次只有文本
            if i == 0:
                content = [{"type": "image"} for _ in range(8)]
                content.append({"type": "text", "text": question})
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "user", "content": [{"type": "text", "text": question}]})

            messages.append({"role": "assistant", "content": answer})
        
        return messages


# ==================== 数据整理器 ====================
class DriveDataCollator:
    """数据整理器 - 支持多图输入"""

    def __init__(self, processor: AutoProcessor):
        self.processor = processor
        self.processor.tokenizer.padding_side = 'left'

    def __call__(self, batch: List[Dict]) -> Dict:
        if not batch:
            return {}

        result = {
            'input_ids': pad_sequence(
                [item['input_ids'] for item in batch],
                batch_first=True,
                padding_value=self.processor.tokenizer.pad_token_id
            ),
            'attention_mask': pad_sequence(
                [item['attention_mask'] for item in batch],
                batch_first=True,
                padding_value=0
            ),
            'labels': pad_sequence(
                [item['labels'] for item in batch],
                batch_first=True,
                padding_value=-100
            ),
        }

        # 视觉特征 - 合并 batch 和图片维度
        if 'pixel_values' in batch[0]:
            pixel_values = torch.stack([item['pixel_values'] for item in batch])
            # [batch, num_images, C, H, W] -> [batch*num_images, C, H, W]
            if pixel_values.dim() == 5:
                batch_size, num_images = pixel_values.shape[:2]
                result['pixel_values'] = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])
            else:
                result['pixel_values'] = pixel_values

        if 'image_grid_thw' in batch[0]:
            image_grid_thw = torch.stack([item['image_grid_thw'] for item in batch]).long()
            # [batch, num_images, 3] -> [batch*num_images, 3]
            if image_grid_thw.dim() == 3:
                batch_size, num_images = image_grid_thw.shape[:2]
                result['image_grid_thw'] = image_grid_thw.view(batch_size * num_images, 3)
            else:
                result['image_grid_thw'] = image_grid_thw

        return result
