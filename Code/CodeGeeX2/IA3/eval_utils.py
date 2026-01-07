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


def compute_accuracy(reference_strings, predicted_strings): 
    assert(len(reference_strings) == len(predicted_strings))
    correct = 0.0
    for i in range(len(reference_strings)):
        if reference_strings[i] == predicted_strings[i]:
            correct += 1 
    return 100 * correct/float(len(reference_strings))


def compute_bleu(references, hypotheses):
    """
    references: List[List[List[str]]]：每个样本可能有多个 reference，每个 reference 是 token list
    hypotheses: List[List[str]]：每个预测是 token list
    """
    bleu_4_sentence_scores = []
    for ref, hyp in zip(references, hypotheses):
        bleu_4_sentence_scores.append(sentence_bleu(ref, hyp,
            smoothing_function=SmoothingFunction().method2))
    return 100*sum(bleu_4_sentence_scores)/float(len(bleu_4_sentence_scores))


def compute_sentence_bleu(ref, hyp):
    return sentence_bleu(ref, hyp, smoothing_function=SmoothingFunction().method2)


def compute_sentence_meteor(reference_list, sentences):
    """
    reference_list: List[List[List[str]]]  多参考
    sentences: List[List[str]]            预测结果
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


def compute_meteor(reference_list, sentences):
    meteor_scores = compute_sentence_meteor(reference_list, sentences)
    return 100 * sum(meteor_scores)/len(meteor_scores)

def compute_unchanged(test_data, predictions):
    source_sentences = [ex.old_comment for ex in test_data]
    predicted_sentences = [' '.join(p) for p in predictions]
    unchanged = 0

    for source, predicted in zip(source_sentences, predicted_sentences):
        if source == predicted:
            unchanged += 1
    
    return 100*(unchanged)/len(test_data)


def compute_sari(test_data, predictions):
    source_sentences = [ex.old_comment for ex in test_data]
    target_sentences = [[ex.new_comment] for ex in test_data]
    predicted_sentences = [' '.join(p) for p in predictions]

    inp = zip(source_sentences, target_sentences, predicted_sentences)
    scores = []

    for source, target, predicted in inp:
        scores.append(SARIsent(source, predicted, target))
    
    return 100*sum(scores)/float(len(scores))

def compute_gleu(test_data, orig_file, ref_file, pred_file):
    command = 'python2.7 gleu/scripts/compute_gleu -s {} -r {} -o {} -d'.format(orig_full_path, ref_full_path,pred_full_path)
    output = subprocess.check_output(command.split())


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