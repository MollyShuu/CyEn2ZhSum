'''
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
=================================================
@Project -> File   ：GraphSum -> translate
@Author ：MollyShuu
@Date   ：2021/5/26 15:02
@IDE    ：PyCharm
==================================================
'''
# -*- coding: utf-8 -*-
import logging

import torch
import argparse
from infer import beam_search, cross_beam_search
from Utils import get_device, parseopt, calculate_bleu, get_prob_idx
from dataset.util_base import build_dataset
from gensim.models import KeyedVectors
from layers.newmodel import GraphTransformer
from layers.newmodel3 import GraphTransformer as GCT

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
opt = parseopt.parse_translate_args()
device = get_device()

opt = argparse.Namespace(**translate_topic_example)
print('==============opt==================')
print(opt)


def zhchar(str):
    english = 'abcdefghijklmnopqrstuvwxyz0123456789'
    unk = '<unk>'
    output = []
    buffer = ''
    for s in str:
        if s in english or s in english.upper():
            buffer += s
        elif s in unk or s in unk.upper():
            buffer += s
        else:
            if buffer: output.append(buffer)
            buffer = ''
            if s != ' ':
                output.append(s)
    if buffer: output.append(buffer)
    return ' '.join(output)



def cross_translate(dataset, fields, model):
    already, hypothesis, references = 0, [], []

    for batch in dataset:
        if opt.tf:
            scores = model(batch.Gsens, batch.Gsens_adjs, batch.Gcpts, batch.dec_inputs, batch.probs, batch.idxes)
            _, predictions = scores.topk(k=1, dim=-1)
        else:
            predictions = cross_beam_search(opt, model, batch.Gsens, batch.Gsens_adjs, batch.Gcpts, batch.probs,
                                            batch.idxes, fields)

        hypothesis += [fields["tgt"].decode(p) for p in predictions]
        already += len(predictions)
        logging.info("Translated: %7d/%7d" % (already, dataset.num_examples))
        references += [fields["tgt"].decode(t) for t in batch.dec_inputs]
        with open(opt.output, "w+", encoding="UTF-8") as out_file:
            out_file.write("\n".join(hypothesis))
            out_file.write("\n")
        with open(opt.ref, "w+", encoding="UTF-8") as out_file:
            out_file.write("\n".join(references))
            out_file.write("\n")

    # Please comment out the following 3 lines，if it is an English summary.
    for i, line in enumerate(hypothesis):
        hypothesis[i] = zhchar(line)
        references[i] = zhchar(references[i])

    if opt.bleu:
        bleu = calculate_bleu(hypothesis, references)
        logging.info("BLEU: %3.2f" % bleu)
        print("BLEU: %3.2f" % bleu)

    origin = sorted(zip(hypothesis, dataset.seed), key=lambda t: t[1])
    hypothesis = [h for h, _ in origin]
    origin = sorted(zip(references, dataset.seed), key=lambda t: t[1])
    references = [h for h, _ in origin]
    with open(opt.output, "w", encoding="UTF-8") as out_file:
        out_file.write("\n".join(hypothesis))
        out_file.write("\n")
    with open(opt.ref, "w", encoding="UTF-8") as out_file:
        out_file.write("\n".join(references))
        out_file.write("\n")

    logging.info("Translation finished. ")


def cross_main():
    logging.info("Build dataset...")
    prob_and_idx = get_prob_idx(opt.prob_idx_path[0], opt.prob_idx_path[1])
    logging.info('-----------load_vocab--------------')
    vocab_src = False
    vocab_tgt = False

    dataset, weights = build_dataset(opt, [opt.input, opt.truth], opt.vocab, device, prob_and_idx, vocab_src, vocab_tgt,
                                     train=False)

    fields = dataset.fields
    pad_ids = {"src": fields["src"].pad_id, "tgt": fields["tgt"].pad_id}
    vocab_sizes = {"src": len(fields["src"].vocab), "tgt": len(fields["tgt"].vocab)}

    # load checkpoint from model_path
    logging.info("Load checkpoint from %s." % opt.model_path)
    checkpoint = torch.load(opt.model_path, map_location=lambda storage, loc: storage)

    logging.info("Build model...")
    model = GCT.load_model(checkpoint["opt"], pad_ids, vocab_sizes, weights, device,
                           checkpoint["model"]).eval()

    logging.info("Start translation...")
    with torch.set_grad_enabled(False):
        cross_translate(dataset, fields, model)


def get_model_parameters():
    logging.info("Build dataset...")
    prob_and_idx = get_prob_idx(opt.prob_idx_path[0], opt.prob_idx_path[1])
    logging.info('-----------load_vocab--------------')
    vocab_src = False
    vocab_tgt = False

    dataset, weights = build_dataset(opt, [opt.input, opt.truth], opt.vocab, device, prob_and_idx, vocab_src, vocab_tgt,
                                     train=False)

    fields = dataset.fields
    pad_ids = {"src": fields["src"].pad_id, "tgt": fields["tgt"].pad_id}
    vocab_sizes = {"src": len(fields["src"].vocab), "tgt": len(fields["tgt"].vocab)}

    # load checkpoint from model_path
    logging.info("Load checkpoint from %s." % opt.model_path)
    checkpoint = torch.load(opt.model_path, map_location=lambda storage, loc: storage)

    logging.info("Build model...")
    model = GCT.load_model(checkpoint["opt"], pad_ids, vocab_sizes, weights, device,
                           checkpoint["model"]).eval()
    print(model)
    print(model.state_dict())
    for name, param in model.named_parameters():
        print(name, ' : ', param.size(), param.requires_grad)


if __name__ == '__main__':
    cross_main()
