import difflib
import logging
import os
import numpy as np
from typing import List, NamedTuple
import subprocess 
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  
from pycocoevalcap.meteor.meteor import Meteor                          
from SARI import SARIsent                                           

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--orig_file", type=str, required=True, help="Directory or path for source (old) comments")
    parser.add_argument("--ref_file", type=str, required=True, help="Directory or path for reference (new) comments")
    parser.add_argument("--pred_file", type=str, required=True, help="Directory or path for predicted comments")
    parser.add_argument("--test_output", type=str, required=True, help="Directory for metrics results")
    
    return parser.parse_args()

# Accuracy：（Exact Match, EM）
def compute_accuracy(reference_strings, predicted_strings): 
    assert(len(reference_strings) == len(predicted_strings))
    correct = 0.0
    for i in range(len(reference_strings)):
        if reference_strings[i] == predicted_strings[i]:
            correct += 1 
    return 100 * correct/float(len(reference_strings))

# BLEU-4（Corpus Level）
def compute_bleu(references, hypotheses):
    """
    references: List[List[List[str]]]：
    hypotheses: List[List[str]]：
    """
    bleu_4_sentence_scores = []
    for ref, hyp in zip(references, hypotheses):
        bleu_4_sentence_scores.append(sentence_bleu(ref, hyp,
            smoothing_function=SmoothingFunction().method2))
    return 100*sum(bleu_4_sentence_scores)/float(len(bleu_4_sentence_scores))

# BLEU-4
def compute_sentence_bleu(ref, hyp):
    return sentence_bleu(ref, hyp, smoothing_function=SmoothingFunction().method2)

# METEOR
def compute_sentence_meteor(reference_list, sentences):
    """
    reference_list: List[List[List[str]]]  
    sentences: List[List[str]]            
    """
    preds = dict()
    refs = dict()

    for i in range(len(sentences)):
        preds[i] = [' '.join([s for s in sentences[i]])]
        refs[i] = [' '.join(l) for l in reference_list[i]]

    final_scores = dict()

    scorers = [
        (Meteor(),"METEOR")
    ]

    for scorer, method in scorers:
        score, scores = scorer.compute_score(refs, preds)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                final_scores[m] = scs
        else:
            final_scores[method] = scores

    meteor_scores = final_scores["METEOR"]
    return meteor_scores

# METEOR
def compute_meteor(reference_list, sentences):
    meteor_scores = compute_sentence_meteor(reference_list, sentences)
    return 100 * sum(meteor_scores)/len(meteor_scores)

# unchanged rate
def compute_unchanged(test_data, predictions):
    source_sentences = [ex.old_comment for ex in test_data]
    predicted_sentences = [' '.join(p) for p in predictions]
    unchanged = 0

    for source, predicted in zip(source_sentences, predicted_sentences):
        if source == predicted:
            unchanged += 1
    
    return 100*(unchanged)/len(test_data)

# SARI
def compute_sari(test_data, predictions):
    source_sentences = [ex.old_comment for ex in test_data]
    target_sentences = [[ex.new_comment] for ex in test_data]
    predicted_sentences = [' '.join(p) for p in predictions]

    inp = zip(source_sentences, target_sentences, predicted_sentences)
    scores = []

    for source, target, predicted in inp:
        scores.append(SARIsent(source, predicted, target))
    
    return 100*sum(scores)/float(len(scores))

# GLEU
def compute_gleu(test_data, orig_file, ref_file, pred_file):
    py27_bin = "/opt/conda/envs/py27/bin/python"
    command = '{} gleu/scripts/compute_gleu -s {} -r {} -o {} -d'.format(py27_bin, orig_file, ref_file, pred_file)
    output = subprocess.check_output(command.split())

    # 解析 GLEU 输出
    output_lines = [l.strip() for l in output.decode("utf-8").split('\n') if len(l.strip()) > 0]
    l = 0
    while l < len(output_lines):
        if output_lines[l][0] == '0':
            break
        l += 1

    scores = np.zeros(len(test_data), dtype=np.float32)
    while l < len(test_data):
        terms = output_lines[l].split()
        idx = int(terms[0])
        val = float(terms[1])
        scores[idx] = val
        l += 1
    scores = np.ndarray.tolist(scores)
    return 100*sum(scores)/float(len(scores))