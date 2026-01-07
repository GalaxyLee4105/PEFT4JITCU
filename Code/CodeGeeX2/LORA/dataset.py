import torch
import codecs
import random
import json
from transformers import AutoTokenizer

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        shuffle: bool = False,
        load_range: tuple[int, int] | None = None,  
        add_eos_to_response: bool = True,           
    ):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_eos_to_response = add_eos_to_response

        print(f"Loading JITCU dataset from: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                    instruction = item["instruction"].strip()   
                    response = item["output"].strip()           
                except Exception as e:
                    print(f"Error parsing line {line_idx}: {e}")
                    continue

                prompt = instruction.rstrip() + "\n\nUpdated Comment:" 
                full_text = prompt + " " + response

                if self.add_eos_to_response:
                    full_text += tokenizer.eos_token

                full_tokenized = tokenizer(
                    full_text,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                if full_tokenized.input_ids.size(1) > max_length:
                    continue 

                prompt_tokenized = tokenizer(
                    prompt,
                    add_special_tokens=False, 
                    return_tensors="pt"
                )
                prompt_len = prompt_tokenized.input_ids.size(1)     

                tokenized = tokenizer(     
                    full_text, 
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt"
                )

                input_ids = tokenized["input_ids"][0]  
                attention_mask = tokenized["attention_mask"][0]


                labels = input_ids.clone()
                labels[:prompt_len] = -100          
                labels[attention_mask == 0] = -100  
                

                valid_label_count = (labels != -100).sum().item()
                if valid_label_count <= 0:
                    print(f"Warning: Sample at line {line_idx} has NO valid labels! "
                          f"Prompt len: {prompt_len}, Total len: {tokenized['attention_mask'][0].sum()}")
                    continue 

                
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
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }