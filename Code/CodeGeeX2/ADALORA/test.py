import os
import json
import torch
import logging
import argparse
import re
import sys
import nltk
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import eval_utils


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class EvalSample:
    def __init__(self, old_comment, new_comment):
        self.old_comment = old_comment  # Source
        self.new_comment = new_comment  # Reference

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model_name_or_path", type=str, default="bigcode/large-model")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Merged model path")
    
    parser.add_argument("--test_filename", type=str, required=True, help="Path to test_new.jsonl")
    parser.add_argument("--orig_file", type=str, required=True, help="Path to source (old) comments file, e.g., test.src")
    parser.add_argument("--ref_file", type=str, required=True, help="Path to reference (new) comments file, e.g., test.ref")
    parser.add_argument("--pred_file", type=str, required=True, help="Path to predicted comments file, e.g., test.pred")
    parser.add_argument("--test_output", type=str, required=True, help="Directory for metrics results")
    
    parser.add_argument("--max_source_length", type=int, default=2048)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--output_size", type=int, default=1)
    parser.add_argument("--test_batch_size", type=int, default=1)
    
    return parser.parse_args()

def extract_old_comment(instruction_text):
    try:
        pattern = r"Previous comment:\s*\n(.*?)\s*\n\nPrevious code:"
        match = re.search(pattern, instruction_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    except Exception:
        return ""

def run_evaluation(test_data_objects, ref_strs, pred_strs, ref_tokens_list, hyp_tokens_list, orig_path, ref_path, pred_path, output_dir):
    results = {}
    logger.info("Computing metrics...")

    # Metric 1: Accuracy (Exact Match)
    try:
        score = eval_utils.compute_accuracy(ref_strs, pred_strs)
        results['Accuracy'] = score
        logger.info(f"Accuracy: {score}")
    except Exception as e:
        logger.warning(f"Accuracy 指标未能评估: {e}")

    # Metric 2: BLEU-4 (Corpus Level)
    try:
        score = eval_utils.compute_bleu(ref_tokens_list, hyp_tokens_list)
        results['BLEU-4'] = score
        logger.info(f"BLEU-4: {score}")
    except Exception as e:
        logger.warning(f"BLEU-4 指标未能评估: {e}")

    # Metric 3: Sentence BLEU (Average)
    try:
        sentence_scores = []
        for ref, hyp in zip(ref_tokens_list, hyp_tokens_list):
            s_score = eval_utils.compute_sentence_bleu(ref[0], hyp) 
            sentence_scores.append(s_score)
        avg_score = sum(sentence_scores) / len(sentence_scores) * 100 if sentence_scores else 0
        results['Sentence-BLEU'] = avg_score
        logger.info(f"Sentence-BLEU: {avg_score}")
    except Exception as e:
        logger.warning(f"Sentence-BLEU 指标未能评估: {e}")

    # Metric 4 & 5: METEOR (Sentence & Corpus)
    try:
        score = eval_utils.compute_meteor(ref_tokens_list, hyp_tokens_list)
        results['METEOR'] = score
        logger.info(f"METEOR: {score}")
    except Exception as e:
        logger.warning(f"METEOR 指标未能评估 (check pycocoevalcap or jdk): {e}")

    # Metric 6: Unchanged Rate
    try:
        score = eval_utils.compute_unchanged(test_data_objects, hyp_tokens_list)
        results['Unchanged-Rate'] = score
        logger.info(f"Unchanged-Rate: {score}")
    except Exception as e:
        logger.warning(f"Unchanged-Rate 指标未能评估: {e}")

    # Metric 7: SARI
    try:
        score = eval_utils.compute_sari(test_data_objects, hyp_tokens_list)
        results['SARI'] = score
        logger.info(f"SARI: {score}")
    except Exception as e:
        logger.warning(f"SARI 指标未能评估 (check SARI.py): {e}")

    # Metric 8: GLEU
    try:
        score = eval_utils.compute_gleu(test_data_objects, orig_path, ref_path, pred_path)
        results['GLEU'] = score
        logger.info(f"GLEU: {score}")
    except Exception as e:
        logger.warning(f"GLEU 指标未能评估 (Require python2.7 and gleu scripts): {e}")

 
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"All metrics saved to {metrics_path}")


def main():
    args = get_args()
    logger.info(args)
    

    logger.info(f"Loading Tokenizer and Model from {args.model_name_or_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    logger.info(f"Reading test file: {args.test_filename}")
    samples = []
    with open(args.test_filename, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                item = json.loads(line)
                instruction = item.get('instruction', "")
                output = item.get('output', "").strip()
                old_comment = extract_old_comment(instruction)
                
                samples.append({
                    'instruction': instruction,
                    'output': output,
                    'old_comment': old_comment
                })
            except json.JSONDecodeError:
                continue
    

    logger.info(f"Starting generation for {len(samples)} samples...")
    predictions = []
    PROMPT_SUFFIX = "\n\nUpdated Comment:"
    
    for i, sample in tqdm(enumerate(samples), total=len(samples), desc="Generating"):

        instruction = sample['instruction']
        input_text = instruction.rstrip() + PROMPT_SUFFIX
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=args.max_source_length).to(model.device)
        
        if inputs.input_ids.size(1) >= args.max_source_length:        
            pred_text = ""
        else:
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs.input_ids,
                    max_new_tokens=args.max_target_length,
                    num_beams=args.beam_size,
                    num_return_sequences=1,
                    early_stopping=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            input_len = inputs.input_ids.shape[1]       
            new_tokens = generated_ids[0][input_len:]   
            pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            if "</s>" in pred_text:
                pred_text = pred_text.split("</s>")[0].strip()

        predictions.append(pred_text)

        if i < 5 or i % 50 == 0:
            logger.info("\n" + "="*60)
            logger.info(f"Sample {i}/{len(samples)} Generation Debug")
            logger.info(f"Input tokens: {input_len}")
            logger.info(f"Output tokens: {len(new_tokens)}")
            if len(new_tokens) > 0:
                logger.info(f"New token IDs (first 10): {new_tokens[:10].tolist()}")
            logger.info(f"[Sample {i}] Prediction: {pred_text}")
            logger.info("="*60)


    os.makedirs(args.test_output, exist_ok=True) 

    orig_path = args.orig_file   
    ref_path = args.ref_file
    pred_path = args.pred_file


    test_data_objects = []        
    ref_strs = []                 
    pred_strs = []                 
    ref_tokens_list = []          
    hyp_tokens_list = []

    logger.info("Saving results and preparing evaluation data...")

    with open(orig_path, 'w', encoding='utf-8') as f_src, \
         open(ref_path, 'w', encoding='utf-8') as f_ref, \
         open(pred_path, 'w', encoding='utf-8') as f_pred:
        
        for samp, pred in zip(samples, predictions):
            old_c = samp['old_comment']
            new_c = samp['output']

            f_src.write(old_c.replace('\n', ' ') + '\n')          
            f_ref.write(new_c.replace('\n', ' ') + '\n')
            f_pred.write(pred.replace('\n', ' ') + '\n')

            test_data_objects.append(EvalSample(old_comment=old_c, new_comment=new_c))  
            
            ref_strs.append(new_c)           
            pred_strs.append(pred)
            
            ref_t = nltk.word_tokenize(new_c)          
            hyp_t = nltk.word_tokenize(pred)
            
            ref_tokens_list.append([ref_t])
            hyp_tokens_list.append(hyp_t)


    run_evaluation(
        test_data_objects=test_data_objects,
        ref_strs=ref_strs,
        pred_strs=pred_strs,
        ref_tokens_list=ref_tokens_list,
        hyp_tokens_list=hyp_tokens_list,
        orig_path=args.orig_file,
        ref_path=args.ref_file, 
        pred_path=args.pred_file,
        output_dir=args.test_output
    )

    logger.info("Evaluation finished.")

if __name__ == "__main__":
    main()