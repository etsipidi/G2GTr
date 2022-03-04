# -*- coding: utf-8 -*-

from parser.metric import Metric

import torch
import torch.nn as nn
import numpy as np
import numpy
from tqdm import tqdm
import torch.nn.functional as F


class Model(object):

    def __init__(self, vocab, parser, state_class, config, num_labels):
        super(Model, self).__init__()
        self.vocab = vocab
        self.parser = parser
        self.num_labels = num_labels
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.state_class = state_class

    def train(self, loader):
        self.parser.train()
        pbar = tqdm(total= len(loader))

        for ccc,(words, tags, masks, actions, mask_actions, rels) in enumerate(loader):

            states = [self.state_class(mask, tags.device,self.vocab.bert_index,self.config.input_graph)
                      for mask in masks]
            s_arc,s_rel = self.parser(words, tags, masks, states, actions, rels)


            if self.config.use_two_opts:
                self.optimizer_nonbert.zero_grad()
                self.optimizer_bert.zero_grad()
            else:
                self.optimizer.zero_grad()

            ## leftarc and rightarc have dependencies, so we filter swap/ reduce and shift
            mask_rels = ((actions != 3).long() * (actions != 2).long() * mask_actions.long()).bool()

            actions = actions[mask_actions]
            s_arc = s_arc[mask_actions]

            rels = rels[mask_rels]
            s_rel = s_rel[mask_rels]

            loss = self.get_loss(s_arc,actions,s_rel,rels)
            loss.backward()
            ## optimization step
            if self.config.use_two_opts:
                self.optimizer_nonbert.step()
                self.optimizer_bert.step()
                self.scheduler_nonbert.step()
                self.scheduler_bert.step()
            else:
                self.optimizer.step()
                self.scheduler.step()
            del states,words,tags,masks,mask_actions,actions,rels,s_rel,s_arc,mask_rels

            pbar.update(1)

    @torch.no_grad()
    def evaluate(self, loader, punct=False):
        self.parser.eval()
        metric = Metric()
        pbar = tqdm(total=len(loader))

        for words, tags, masks,heads,rels,mask_heads in loader:
            states = [self.state_class(mask, tags.device,self.vocab.bert_index,self.config.input_graph)
                      for mask in masks]

            states = self.parser(words, tags, masks,states)

            pred_heads = []
            pred_rels = []
            for state in states:
                pred_heads.append([h[0] for h in state.head][1:])
                pred_rels.append([h[1] for h in state.head][1:])
            pred_heads = [item for sublist in pred_heads for item in sublist]
            pred_rels = [item for sublist in pred_rels for item in sublist]

            pred_heads = torch.tensor(pred_heads).to(heads.device)
            pred_rels = torch.tensor(pred_rels).to(heads.device)

            heads = heads[mask_heads]
            rels = rels[mask_heads]
            pbar.update(1)
            metric(pred_heads, pred_rels, heads, rels)
            del states

        return metric

    @torch.no_grad()
    def predict(self, loader):
        self.parser.eval()
        metric = Metric()
        pbar = tqdm(total=len(loader))
        all_arcs, all_rels = [], []
        for words, tags, masks,heads,rels,mask_heads in loader:
            states = [self.state_class(mask, tags.device, self.vocab.bert_index,self.config.input_graph)
                      for mask in masks]
            states = self.parser(words, tags, masks, states)

            pred_heads = []
            pred_rels = []
            for state in states:
                pred_heads.append([h[0] for h in state.head][1:])
                pred_rels.append([h[1] for h in state.head][1:])

            pred_heads = [item for sublist in pred_heads for item in sublist]
            pred_rels = [item for sublist in pred_rels for item in sublist]

            pred_heads = torch.tensor(pred_heads).to(heads.device)
            pred_rels = torch.tensor(pred_rels).to(heads.device)

            heads = heads[mask_heads]
            rels = rels[mask_heads]

            metric(pred_heads, pred_rels, heads, rels)

            lens = masks.sum(1).tolist()

            all_arcs.extend(torch.split(pred_heads, lens))
            all_rels.extend(torch.split(pred_rels, lens))
            pbar.update(1)
        all_arcs = [seq.tolist() for seq in all_arcs]
        all_rels = [self.vocab.id2rel(seq) for seq in all_rels]

        return all_arcs, all_rels, metric

    def get_loss(self, s_arc, actions, s_rel, rels):
        arc_loss = self.criterion(s_arc, actions)
        rel_loss = self.criterion(s_rel, rels)
        loss = arc_loss + rel_loss
        return loss