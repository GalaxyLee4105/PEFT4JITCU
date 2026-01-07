# 让模型“只对 response（output）部分计算 loss”，而 prompt / padding 不参与训练
import torch
import codecs
import random
import json
from transformers import AutoTokenizer

class Dataset(torch.utils.data.Dataset):
    """
    每条数据包含：
    - instruction：代码 + 旧注释（作为上下文）
    - output：更新后的新注释（模型真正要学的部分）
    核心设计思想：prompt 只作为“条件输入”、loss 只计算 response（output）部分
    """
    def __init__(
        self,
        file_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        shuffle: bool = False,
        load_range: tuple[int, int] | None = None,  # 只加载部分数据
        add_eos_to_response: bool = True,           # 是否在 response 后加 eos
    ):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_eos_to_response = add_eos_to_response

        print(f"Loading JITCU dataset from: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                # 跳过空行
                line = line.strip()
                if not line:
                    continue
                
                # 解析 JSON，这里丢弃三类样本：JSON 语法错误、缺少 "instruction" / "output"和非 UTF-8 / 非法字符
                try:
                    item = json.loads(line)
                    instruction = item["instruction"].strip()   # instruction：模型输入（上下文）
                    response = item["output"].strip()           # response：模型要生成的内容
                except Exception as e:
                    print(f"Error parsing line {line_idx}: {e}")
                    continue

                prompt = instruction.rstrip() + "\n\nUpdated Comment:"  # 1. 构造完整的 prompt（不加 eos），可以根据需要调整 prompt 模板，这里保持简单
                full_text = prompt + " " + response     # 2. 拼接完整文本：full_text = prompt + response

                if self.add_eos_to_response:
                    full_text += tokenizer.eos_token

                # 先 tokenize 一次，用于检查长度，长度过滤（如果 prompt + response > max_length，整条样本直接被丢弃） 
                full_tokenized = tokenizer(
                    full_text,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                if full_tokenized.input_ids.size(1) > max_length:
                    continue  # 跳过过长样本（可改成截断）

                # 3. 只 tokenize prompt 部分，不加特殊符号，用于计算 labels 掩码位置
                prompt_tokenized = tokenizer(
                    prompt,
                    add_special_tokens=False,  # 不加 bos/eos
                    return_tensors="pt"
                )
                prompt_len = prompt_tokenized.input_ids.size(1)     # prompt 对应的 token 数量， prompt_len 决定 loss 从哪里开始算

                # 4. 最终 tokenize（带 padding）
                tokenized = tokenizer(      # 先对 full_text（prompt + " " + response + eos）进行完整 tokenize，并 padding 到 max_length
                    full_text, 
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt"
                )

                input_ids = tokenized["input_ids"][0]   # 形状: [max_length]，→ 此时 input_ids 包含：[prompt tokens] + [response tokens] + [eos] + [padding tokens]
                attention_mask = tokenized["attention_mask"][0]

                # 5. 构造 labels：只对 response 部分计算损失
                labels = input_ids.clone()
                # 最终只有 response tokens 会被计算 loss
                labels[:prompt_len] = -100          # prompt 部分不计算 loss
                labels[attention_mask == 0] = -100  #  padding 部分也不计算 loss
                
                # 有效标签检查（会被丢弃的情况：response 为空、response 被截断光、response token 数 = 0）
                valid_label_count = (labels != -100).sum().item()
                if valid_label_count <= 0:
                    print(f"Warning: Sample at line {line_idx} has NO valid labels! "
                          f"Prompt len: {prompt_len}, Total len: {tokenized['attention_mask'][0].sum()}")
                    continue # 跳过这个没有有效标签的样本，防止训练报错
                # ---------------------
                
                self.data.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                })

                if load_range and len(self.data) >= load_range[1]:
                    break

        if shuffle:
            random.seed(42)
            random.shuffle(self.data)

        if load_range:
            self.data = self.data[load_range[0]:]

        print(f"{file_path} total size after filtering: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def custom_collate(batch):
    """
    collate 的作用：
    - 把 list[dict] → dict[tensor]
    - 用于 DataLoader 组 batch
    """
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }