import pickle
import time
import json

def read_fenci_json(filename):
    for line in open(filename, 'r', encoding='utf-8-sig'):
        g = json.loads(line)
        Gsen_vertex_features = g['Gsen_vertex_features']
        Gcpt_vertex_features = g['concepts']
        Gsen_adj_martix = g['Gsen_adj_martix']
        Gcpt_adj_martix = 1
        summary = g['summary']
        yield Gsen_vertex_features, Gcpt_vertex_features, Gsen_adj_martix, Gcpt_adj_martix,  summary
        
def get_prob_idx(prob_file, idx_file):
    with open(prob_file, 'rb', 0) as file1, open(idx_file, 'rb', 0) as file2:
        problist = pickle.load(file1)
        idxlist = pickle.load(file2)
    return [problist, idxlist]
