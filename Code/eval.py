import pickle
import collections
import sys
import numpy as np
import pandas as pd
sys.path.append('./pycocoevalcap')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from rouge import Rouge as R
from pycocoevalcap.meteor.meteor import Meteor


def evaluate(cands_list,ref_list):
    my_scorers = [
        (Meteor(), "METEOR")
    ] 

    hypo={}
    ref={}
    final_scores = {}
    for vid, temp in enumerate(cands_list):
        hypo[vid] = [temp]
        ref[vid] = [refs_list[vid]]
    for scorer, method in my_scorers:
        score,_ = scorer.compute_score(ref, hypo)

        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

def rouge1(cands, refs):
    '''
    f:F1值  p：查准率  R：召回率
    '''
    R1 = []
    R2 = []
    RL = []
    rouge = R()
    for i, cand in enumerate(cands):
        rouge_score = rouge.get_scores(cand, refs[i])
        R1.append(rouge_score[0]["rouge-1"]['f'])
        R2.append(rouge_score[0]["rouge-2"]['f'])
        RL.append(rouge_score[0]["rouge-l"]['f'])
    R1 = np.mean(np.array(R1)) * 100
    R2 = np.mean(np.array(R2)) * 100
    RL = np.mean(np.array(RL)) * 100

    return R1, R2, RL



if __name__ == '__main__':
    cand_file='hypothesis.txt'
    ref_file='references.txt'

    with open(ref_file, encoding='utf-8') as f:
        refs_list = [x.strip() for x in f.readlines()]  
    with open(cand_file,encoding='utf-8') as f:
        cands_list = [x.strip() for x in f.readlines()]

    ME_final_score=evaluate(cands_list, refs_list)

    R1, R2, RL=rouge1(cands_list, refs_list)

    print('METEOR SCORE:',ME_final_scoreE)
    print('rouge01:',R1)
    print('rouge-2:',R2)
    print('rouge-3:',R3)