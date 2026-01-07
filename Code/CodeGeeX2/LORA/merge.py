from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str, default="zai-org/codegeex2-6b")   
    parser.add_argument("--peft_model_path", type=str, default="/")    
    parser.add_argument("--merged_output_dir", type=str, default=None)   

    return parser.parse_args()

def main():
    args = get_args()

    print(f"Loading base model: {args.base_model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        trust_remote_code=True, 
        # return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto" 
    )

    print(f"Loading PEFT adapter from: {args.peft_model_path}")
    model = PeftModel.from_pretrained(base_model, args.peft_model_path)

    print("Merging weights... This may take a few minutes.")
    model = model.merge_and_unload()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path, trust_remote_code=True)

    # if args.push_to_hub:
    #     print(f"Saving to hub ...")
    #     model.push_to_hub(f"{args.base_model_name_or_path}-merged", use_temp_dir=False, private=True)
    #     tokenizer.push_to_hub(f"{args.base_model_name_or_path}-merged", use_temp_dir=False, private=True)
    # else:
    #     model.save_pretrained(f"{args.base_model_name_or_path}-merged")
    #     tokenizer.save_pretrained(f"{args.base_model_name_or_path}-merged")
    #     print(f"Model saved to {args.base_model_name_or_path}-merged")
    save_path = args.merged_output_dir if args.merged_output_dir else f"{args.peft_model_path}_merged"
    
    print(f"Saving merged model to: {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

    import glob
    import shutil
    for py_file in glob.glob(os.path.join(args.base_model_name_or_path, "*.py")):
        shutil.copy(py_file, save_path)
        print(f"Copied {py_file} to {save_path}")
    
if __name__ == "__main__" :
    main()