#!/usr/bin/env python3
"""
DriveGPT 模型测试程序
简单对比模型预测和标准答案
"""

import os
import json
import torch
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from PIL import Image

from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel


@dataclass
class EvalConfig:
    """测试配置"""
    base_model_path: str = r"E:\DriveGPT\model\Qwen3-VL-2B-Instruct"
    adapter_path: str = r"E:\DriveGPT\output"
    test_json_path: str = r"E:\DriveGPT\data\BDD_X_testing_label_zh.json"
    image_root: str = r"E:\DriveGPT\image\BDD_X_imgs_select"
    num_samples: int = 5  # 测试样本数
    num_frames: int = 8
    max_new_tokens: int = 100


def load_model(config: EvalConfig):
    """加载模型"""
    print("=" * 60)
    print("加载模型...")
    print("=" * 60)
    
    print("\n[1/2] 加载基础模型...")
    model = AutoModelForImageTextToText.from_pretrained(
        config.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print("✓ 基础模型加载成功")
    
    print("\n[2/2] 加载 LoRA 适配器...")
    model = PeftModel.from_pretrained(model, config.adapter_path)
    print("✓ 适配器加载成功")
    
    processor = AutoProcessor.from_pretrained(config.base_model_path, trust_remote_code=True)
    
    model.eval()
    return model, processor


def load_samples(config: EvalConfig):
    """加载测试样本"""
    with open(config.test_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 过滤有效样本
    valid_samples = []
    for sample in data:
        sample_id = sample.get('id', '')
        all_exist = True
        for idx in sample.get('idx_list', [0, 3, 7, 11, 15, 19, 23, 27]):
            img_path = Path(config.image_root) / f"{sample_id}_{idx}.png"
            if not img_path.exists():
                all_exist = False
                break
        if all_exist:
            valid_samples.append(sample)
    
    # 随机选择
    if len(valid_samples) > config.num_samples:
        valid_samples = random.sample(valid_samples, config.num_samples)
    
    return valid_samples


def get_qa_pairs(sample: dict) -> list:
    """获取所有问答对"""
    conversations = sample.get('conversations', [])
    qa_pairs = []
    
    for i in range(0, len(conversations) - 1, 2):
        question = conversations[i].get('value', '')
        answer = conversations[i + 1].get('value', '')
        # 移除 <视频> 标记
        question = question.replace('\n<视频>', '').replace('<视频>', '')
        qa_pairs.append({'question': question, 'answer': answer})
    
    return qa_pairs


def predict_qa(model, processor, images: list, question: str, config: EvalConfig):
    """预测单个问题"""
    messages = [
        {"role": "user", "content": [{"type": "image"} for _ in range(len(images))] + [{"type": "text", "text": question}]}
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=config.max_new_tokens)
    
    response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    
    return response


def main():
    config = EvalConfig()
    random.seed(42)
    
    # 加载模型
    model, processor = load_model(config)
    
    # 加载样本
    print("\n" + "=" * 60)
    print("加载测试样本...")
    samples = load_samples(config)
    print(f"✓ 加载 {len(samples)} 个样本")
    print("=" * 60)
    
    # 测试每个样本
    for i, sample in enumerate(samples, 1):
        sample_id = sample.get('id', '')
        
        print(f"\n{'='*60}")
        print(f"样本 {i}/{len(samples)}: {sample_id}")
        print(f"{'='*60}")
        
        try:
            # 加载图片
            frame_paths = []
            for idx in sample.get('idx_list', [0, 3, 7, 11, 15, 19, 23, 27]):
                img_path = Path(config.image_root) / f"{sample_id}_{idx}.png"
                if img_path.exists():
                    frame_paths.append(str(img_path))
            
            if len(frame_paths) > config.num_frames:
                step = len(frame_paths) // config.num_frames
                indices = [i * step for i in range(config.num_frames)]
                frame_paths = [frame_paths[i] for i in indices]
            
            images = [Image.open(p).convert('RGB') for p in frame_paths]
            
            # 获取所有问答对
            qa_pairs = get_qa_pairs(sample)
            
            # 逐个问题进行预测
            for j, qa in enumerate(qa_pairs, 1):
                question = qa['question']
                ground_truth = qa['answer']
                
                print(f"\n  问题 {j}: {question[:50]}...")
                
                prediction = predict_qa(model, processor, images, question, config)
                
                print(f"  【预测】 {prediction}")
                print(f"  【标准】 {ground_truth}")
        
        except Exception as e:
            print(f"\n预测失败: {e}")
    
    print(f"\n{'='*60}")
    print("测试完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
