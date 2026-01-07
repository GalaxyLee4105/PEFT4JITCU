import json
import os

# --- 1. 配置路径和文件 ---
INPUT_DIR = "ACL20" 
SPLITS = ["train", "valid", "test"] # 对应 train_cleaned.json, valid_cleaned.json, test_cleaned.json

# 定义输出文件名后缀和对应的模板名称
FORMATS = {
    "NEW": "_new.jsonl", 
    "DIFF": "_diff.jsonl",
    "ND": "_nd.jsonl"
}

# --- 2. 指令模板定义 ---
# 基础任务指令
BASE_INSTRUCTION = "As a code documentation specialist, update the comment to match the implemented code changes."

# 模板 1: old_comment + old_code + new_code 
TEMPLATE_NEW = BASE_INSTRUCTION + """

Previous comment:
{old_comment}

Previous code:
```java
{old_code}
```

Current code:
```java
{new_code}
```
"""

# 模板 2: old_comment + old_code + span_diff_code 
TEMPLATE_DIFF = BASE_INSTRUCTION + """

Previous comment:
{old_comment}

Previous code:
```java
{old_code}
```

Code modifications:
{span_diff_code}
"""

# 模板 3: old_comment + old_code + new_code + span_diff_code 
TEMPLATE_ND = BASE_INSTRUCTION + """

Previous comment:
{old_comment}

Previous code:
```java
{old_code}
```

Current code:
```java
{new_code}
```

Code modifications:
{span_diff_code}
"""

# 将模板映射到格式名称以便循环处理
TEMPLATES = {
    "NEW": TEMPLATE_NEW,
    "DIFF": TEMPLATE_DIFF,
    "ND": TEMPLATE_ND
}

def load_raw_data(filepath):
    """
    按行读取，确保能够处理 JSON Lines 格式。
    """
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"❌ 警告: 文件 {filepath} 中第 {i+1} 行有无效的 JSON 行。跳过。错误: {e}")
        return data
    except FileNotFoundError:
        print(f"❌ 错误: 找不到输入文件 {filepath}。请检查路径。")
        return []


def create_jsonl_record(template_str, record):
    """
    根据模板和原始记录，创建标准的 {"instruction": ..., "output": ...} 字典。
    注意：使用最新的 JSON 示例中的键名（如 old_comment, span_diff_code）。
    """
    
    # 准备用于 .format() 的参数
    format_args = {
        "old_comment": record.get("old_comment", "").strip(),
        "old_code": record.get("old_code", "").strip(),
        "new_code": record.get("new_code", "").strip(),
        "span_diff_code": record.get("span_diff_code", "").strip() 
    }
    
    # 填充 instruction
    instruction_content = template_str.format(**format_args).strip()

    # 填充 output
    output_content = record.get("new_comment", "").strip()

    return {
        "instruction": instruction_content,
        "output": output_content
    }

def write_jsonl_file(filepath, data_list):
    """将字典列表（每条记录）写入 JSONL 文件（每行一个 JSON 对象）。"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for record in data_list:
                json_line = json.dumps(record, ensure_ascii=False)                  # 序列化为 JSON 字符串
                f.write(json_line + '\n')                   # 写入文件，并添加换行符
        print(f"✅ 成功生成文件: {filepath}，共包含 {len(data_list)} 条记录。")
    except Exception as e:
        print(f"❌ 写入文件 {filepath} 时发生错误: {e}")

def main():
    """主函数：加载所有分片数据，并为每个分片生成三种格式的 JSONL 文件。"""
    
    # 循环处理 'train', 'valid', 'test'
    for split in SPLITS:
        input_filename = f"{split}_cleaned.json"
        input_filepath = os.path.join(INPUT_DIR, input_filename)
        
        print(f"\n--- 正在处理 {input_filepath} ---")
        
        raw_data = load_raw_data(input_filepath)

        if not raw_data:
            print(f"跳过 {split} (无数据)。")
            continue

        # 为 NEW, DIFF, ND 三种格式生成文件
        for format_name, template in TEMPLATES.items():
            # 1. 构建输出文件名 (例如: train_new.jsonl)
            output_filename = f"{split}{FORMATS[format_name]}"
            
            # 2. 构建输出完整路径 (将文件放入 INPUT_DIR, 例如: ACL20/train_new.jsonl)
            output_filepath = os.path.join(INPUT_DIR, output_filename)
            
            # --- 转换数据 ---
            formatted_data = [create_jsonl_record(template, record) for record in raw_data]
            
            # --- 写入文件 ---
            # 使用新的输出路径 output_filepath
            write_jsonl_file(output_filepath, formatted_data)
    
    print("\n--- 所有转换完成。---")

if __name__ == "__main__":
    main()