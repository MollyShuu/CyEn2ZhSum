'''
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
=================================================
@Project -> File   ：GraphSum -> trainGranpSum
@Author ：MollyShuu
@Date   ：2021/5/17 10:35
@IDE    ：PyCharm
==================================================
'''
import os
import logging
import datetime
import torch
import torch.cuda
from layers.newmodel3 import GraphTransformer
from loss import LabelSmoothingLoss, WarmAdam
from infer import beam_search
from Utils import get_device, parseopt, printing_opt, Saver, calculate_bleu, get_prob_idx
from dataset.util_base import build_dataset
import argparse
import pickle

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
device = get_device()
logging.info("DEVICE:" + str(device))
opt = parseopt.parse_train_args()
print('==============opt==================')
print(opt)

logging.info("\n" + printing_opt(opt))

saver = Saver(opt)


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


def valid(model, criterion, valid_dataset, step):
    model.eval()
    total_loss, total = 0.0, 0

    hypothesis, references = [], []
    for batch in valid_dataset:
        scores = model(batch.Gsens, batch.Gsens_adjs, batch.Gcpts, batch.dec_inputs, batch.probs, batch.idxes)
        loss = criterion(scores, batch.dec_inputs)
        total_loss += loss.data
        total += 1
        if opt.tf:
            _, predictions = scores.topk(k=1, dim=-1)
        else:
            predictions = beam_search(opt, model, batch.src, valid_dataset.fields)

        hypothesis += [valid_dataset.fields['tgt'].decode(p) for p in predictions]
        references += [valid_dataset.fields['tgt'].decode(t) for t in batch.dec_inputs]
    for i, line in enumerate(hypothesis):
        hypothesis[i] = zhchar(line)
        references[i] = zhchar(references[i])
    with open(opt.output, "w", encoding="UTF-8") as out_file:
        out_file.write("\n".join(hypothesis))
        out_file.write("\n")
    with open(opt.ref, "w", encoding="UTF-8") as out_file:
        out_file.write("\n".join(references))
        out_file.write("\n")
    # bleu = 0
    bleu = calculate_bleu(hypothesis, references)
    logging.info("Valid loss: %.2f\tValid BLEU: %3.2f" % (total_loss / total, bleu))
    checkpoint = {"model": model.state_dict(), "opt": opt}
    saver.save(checkpoint, step, bleu, total_loss / total)
    del checkpoint
    # print('hypothesis.txt', hypothesis)
    # print('references', references)


def train(model, criterion, optimizer, train_dataset, valid_dataset):
    total_loss = 0.0
    model_path = opt.model_path
    model.zero_grad()
    for i, batch in enumerate(train_dataset):  
        scores = model(batch.Gsens, batch.Gsens_adjs, batch.Gcpts, batch.dec_inputs, batch.probs,
                       batch.idxes)
        loss = criterion(scores, batch.dec_inputs)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        loss.backward()
        total_loss = total_loss + loss.item()  # loss.data

        if (i + 1) % opt.grad_accum == 0:
            optimizer.step()
            model.zero_grad()

            if optimizer.n_step % opt.report_every == 0:
                mean_loss = total_loss / opt.report_every / opt.grad_accum
                logging.info("step: %7d\t loss: %7f" % (optimizer.n_step, mean_loss))
                with open(os.path.join(saver.model_path, "train_log"), "a", encoding="UTF-8") as log:
                    log.write("%s\t step: %6d\t loss: %.2f\n" % (datetime.datetime.now(), optimizer.n_step, mean_loss))
                total_loss = 0.0

            if optimizer.n_step % opt.save_every == 0:
                with torch.no_grad():
                    valid(model, criterion, valid_dataset, optimizer.n_step)
                model.train()
        del loss


if __name__ == '__main__':
    logging.info("Build dataset...")
    prob_and_idx = get_prob_idx(opt.prob_idx_path[0], opt.prob_idx_path[1])
    vocab_src = False
    vocab_tgt = False
    logging.info('-----------valid_dataset--------------')
    valid_dataset, weights = build_dataset(opt, opt.valid, opt.vocab, device, prob_and_idx, vocab_src, vocab_tgt,
                                           train=False)
    logging.info('-----------train_dataset--------------')
    train_dataset, _ = build_dataset(opt, opt.train, opt.vocab, device, prob_and_idx, vocab_src, vocab_tgt,
                                     train=True)

    fields = valid_dataset.fields = train_dataset.fields  
    del vocab_tgt, vocab_src, prob_and_idx
    logging.info("Build model...")
    pad_ids = {"src": fields["src"].pad_id, "tgt": fields["tgt"].pad_id}
    vocab_sizes = {"src": len(fields["src"].vocab), "tgt": len(fields["tgt"].vocab)}
    logging.info('vocab_sizes:' + str(vocab_sizes))

    model = GraphTransformer.load_model(opt, pad_ids, vocab_sizes, weights, device)

    criterion = LabelSmoothingLoss(opt.label_smoothing, vocab_sizes["tgt"], pad_ids["tgt"]).to(device)
    n_step = int(opt.train_from.split("-")[-1]) if opt.train_from else 1
    optimizer = WarmAdam(model.parameters(), opt.lr, opt.embedding_dim, opt.warm_up, n_step)

    logging.info("start training...")
    train(model, criterion, optimizer, train_dataset, valid_dataset)
