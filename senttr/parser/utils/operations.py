# -*- coding: utf-8 -*-

import torch

class ParserOperations(object):
    def __init__(self):
        pass

    def get_vocab_transitions(self):
        raise NotImplementedError()

    @classmethod
    def read_seq(self, in_file, vocab):
        raise NotImplementedError()

    def state_update(self, state, act, rel=None):
        raise NotImplementedError()

    def state_legal_act(self, state):
        raise NotImplementedError()


class ArcStandardSwapOps(ParserOperations):
    def __init__(self):
        super(ArcStandardSwapOps, self).__init__()
        self.vocab_transitions = ['L', 'R', 'S', 'H']
        self.act_dict = {0:"LEFTARC", 1:"RIGHTARC", 2:"SHIFT", 3:"SWAP"}

    def get_vocab_transitions(self):
        return self.vocab_transitions

    @classmethod
    def read_seq(self, in_file, vocab):
        lines = []
        with open(in_file, 'r') as f:
            for line in f:
                lines.append(line)
        for i in range(len(lines)):
            lines[i] = lines[i].strip().split()
        gold_seq, arcs, seq = [], [], []
        max_read = 0
        for line in lines:
            if len(line) == 0:
                gold_seq.append({'act':seq, 'rel':arcs})
                max_read += 1
                arcs, seq = [], []
            elif len(line) == 3:
                assert line[0] == 'Shift'
                seq.append(2)
                arcs.append(0)
            elif len(line) == 1:
                assert line[0] == 'Swap'
                seq.append(3)
                arcs.append(0)
            elif len(line) == 2:
                if line[0].startswith('R'):
                    assert line[0] == 'Right-Arc'
                    seq.append(1)
                    arcs.append(vocab.rel2id( line[1] ))
                elif line[0].startswith('L'):
                    assert line[0] == 'Left-Arc'
                    seq.append(0)
                    arcs.append(vocab.rel2id( line[1] ))
        return gold_seq

    def state_update(self, state, act, rel=None):
        act = self.act_dict[act.item()]
        if not state.finished():
            if act == "SHIFT":
                state.stack = [state.buf[0]] + state.stack
                state.buf = state.buf[1:]
                state.tok_buffer = torch.roll(state.tok_buffer,-1,dims=0).clone()
                state.tok_stack = torch.roll(state.tok_stack,1,dims=0).clone()
                state.tok_stack[0] = state.tok_buffer[-1].clone()
            elif act == "LEFTARC":
                state.head[state.stack[1]] = [state.stack[0], rel.item()]
                if state.input_graph:
                    state.graph[state.convert[state.stack[0]],state.convert[state.stack[1]]] = 1
                    state.graph[state.convert[state.stack[1]],state.convert[state.stack[0]]] = 2
                    state.label[state.convert[state.stack[1]]] = rel
                state.stack = [state.stack[0]] + state.stack[2:]
                state.tok_stack = torch.cat(
                    (state.tok_stack[0].unsqueeze(0),torch.roll(state.tok_stack[1:],-1,dims=0))).clone()
            elif act == "RIGHTARC":
                state.head[state.stack[0]] = [state.stack[1], rel.item()]
                if state.input_graph:
                    state.graph[state.convert[state.stack[1]],state.convert[state.stack[0]]] = 1
                    state.graph[state.convert[state.stack[0]],state.convert[state.stack[1]]] = 2
                    state.label[state.convert[state.stack[0]]] = rel
                state.stack = state.stack[1:]
                state.tok_stack = torch.roll(state.tok_stack,-1,dims=0).clone()
            elif act == "SWAP":
                state.buf = [state.stack[1]] + state.buf
                state.stack = [state.stack[0]] + state.stack[2:]
                state.tok_stack = torch.cat(
                    (state.tok_stack[0].unsqueeze(0), torch.roll(state.tok_stack[1:], -1, dims=0))).clone()
                state.tok_buffer = torch.roll(state.tok_buffer, 1, dims=0).clone()
                state.tok_buffer[0] = state.tok_stack[-1]

    def state_legal_act(self, state):
        t = [0,0,0,0]
        if len(state.stack) >= 2 and state.stack[1] != 0:
            t[0] = 1
        if len(state.stack) >= 2 and state.stack[0] != 0:
            t[1] = 1
        if len(state.buf) > 0:
            t[2] = 1
        if len(state.stack) >= 2 and 0 < state.stack[1] < state.stack[0]:
            t[3] = 1
        return t


class ArcStandardOps(ParserOperations):
    def __init__(self):
        super(ArcStandardOps, self).__init__()
        self.vocab_transitions = ['L', 'R', 'S']
        self.act_dict = {0:"LEFTARC", 1:"RIGHTARC", 2:"SHIFT"}

    def get_vocab_transitions(self):
        return self.vocab_transitions

    @classmethod
    def read_seq(self, in_file, vocab):
        lines = []
        with open(in_file, 'r') as f:
            for line in f:
                lines.append(line)
        for i in range(len(lines)):
            lines[i] = lines[i].strip().split()
        gold_seq, arcs, seq = [], [], []
        max_read = 0
        for line in lines:
            if len(line) == 0:
                gold_seq.append({'act':seq, 'rel':arcs})
                max_read += 1
                arcs, seq = [], []
            elif line[0].startswith('S'):
                assert line[0] == 'Shift'
                seq.append(2)
                arcs.append(0)
            elif line[0].startswith('R'):
                assert line[0] == 'Right-Arc'
                seq.append(1)
                arcs.append(vocab.rel2id( line[1] ))
            elif line[0].startswith('L'):
                assert line[0] == 'Left-Arc'
                seq.append(0)
                arcs.append(vocab.rel2id( line[1] ))
        return gold_seq

    def state_update(self, state, act, rel=None):
        act = self.act_dict[act.item()]
        if not state.finished():
            if act == "SHIFT":
                state.stack = [state.buf[0]] + state.stack
                state.buf = state.buf[1:]
                state.tok_buffer = torch.roll(state.tok_buffer,-1,dims=0).clone()
                state.tok_stack = torch.roll(state.tok_stack,1,dims=0).clone()
                state.tok_stack[0] = state.tok_buffer[-1].clone()
            elif act == "LEFTARC":
                state.head[state.stack[1]] = [state.stack[0], rel.item()]
                if state.input_graph:
                    state.graph[state.convert[state.stack[0]],state.convert[state.stack[1]]] = 1
                    state.graph[state.convert[state.stack[1]],state.convert[state.stack[0]]] = 2
                    state.label[state.convert[state.stack[1]]] = rel
                state.stack = [state.stack[0]] + state.stack[2:]
                state.tok_stack = torch.cat(
                    (state.tok_stack[0].unsqueeze(0),torch.roll(state.tok_stack[1:],-1,dims=0))).clone()
            elif act == "RIGHTARC":
                state.head[state.stack[0]] = [state.stack[1], rel.item()]
                if state.input_graph:
                    state.graph[state.convert[state.stack[1]],state.convert[state.stack[0]]] = 1
                    state.graph[state.convert[state.stack[0]],state.convert[state.stack[1]]] = 2
                    state.label[state.convert[state.stack[0]]] = rel
                state.stack = state.stack[1:]
                state.tok_stack = torch.roll(state.tok_stack,-1,dims=0).clone()

    def state_legal_act(self, state):
        t = [0,0,0]
        if len(state.stack) >= 2 and state.stack[1] != 0:
            t[0] = 1
        if len(state.stack) >= 2 and state.stack[0] != 0:
            t[1] = 1
        if len(state.buf) > 0:
            t[2] = 1
        return t


class ArcEagerOps(ParserOperations):
    def __init__(self):
        super(ArcEagerOps, self).__init__()
        self.vocab_transitions = ['LA', 'RA', 'SH', 'RE']
        self.act_dict = {0:"LEFTARC", 1:"RIGHTARC", 2:"SHIFT", 3:"REDUCE"}

    def get_vocab_transitions(self):
        return self.vocab_transitions

    @classmethod
    def read_seq(in_file, vocab):
        lines = []
        with open(in_file, 'r') as f:
            for line in f:
                lines.append(line)
        for i in range(len(lines)):
            lines[i] = lines[i].strip().split()
        gold_seq, arcs, seq = [], [], []
        max_read = 0
        for line in lines:
            if len(line) == 0:
                gold_seq.append({'act':seq, 'rel':arcs})
                max_read += 1
                arcs, seq = [], []
            elif line[0].startswith('Sh'):
                assert line[0] == 'Shift'
                seq.append(2)
                arcs.append(0)
            elif line[0].startswith('Re'):
                assert line[0] == 'Reduce'
                seq.append(3)
                arcs.append(0)
            elif line[0].startswith('Ri'):
                assert line[0] == 'Right-Arc'
                seq.append(1)
                arcs.append(vocab.rel2id( line[1] ))
            elif line[0].startswith('Le'):
                assert line[0] == 'Left-Arc'
                seq.append(0)
                arcs.append(vocab.rel2id( line[1] ))
        return gold_seq

    def state_update(self, state, act, rel=None):
        act = self.act_dict[act.item()]
        if not state.finished():
            if len(state.buf) == 0:
                act = "REDUCE"
            if act == "SHIFT":
                state.stack = [state.buf[0]] + state.stack
                state.buf = state.buf[1:]
                state.tok_buffer = torch.roll(state.tok_buffer,-1,dims=0).clone()
                state.tok_stack = torch.roll(state.tok_stack,1,dims=0).clone()
                state.tok_stack[0] = state.tok_buffer[-1].clone()
            elif act == "LEFTARC":
                state.head[state.stack[0]] = [state.buf[0], rel.item()]
                if state.input_graph:
                    state.graph[state.convert[state.buf[0]],state.convert[state.stack[0]]] = 1
                    state.graph[state.convert[state.stack[0]],state.convert[state.buf[0]]] = 2
                    state.label[state.convert[state.stack[0]]] = rel
                state.stack = state.stack[1:]
                state.tok_stack = torch.roll(state.tok_stack,-1,dims=0).clone()
            elif act == "RIGHTARC":
                state.head[state.buf[0]] = [state.stack[0], rel.item()]
                if state.input_graph:
                    state.graph[state.convert[state.stack[0]],state.convert[state.buf[0]]] = 1
                    state.graph[state.convert[state.buf[0]],state.convert[state.stack[0]]] = 2
                    state.label[state.convert[state.buf[0]]] = rel
                state.stack = [state.buf[0]] + state.stack
                state.buf = state.buf[1:]
                state.tok_buffer = torch.roll(state.tok_buffer,-1,dims=0).clone()
                state.tok_stack = torch.roll(state.tok_stack,1,dims=0).clone()
                state.tok_stack[0] = state.tok_buffer[-1].clone()
            elif act == "REDUCE":
                state.stack = state.stack[1:]
                state.tok_stack = torch.roll(state.tok_stack,-1,dims=0).clone()


    def state_legal_act(self, state):
        t = [0,0,0,0]
        if len(state.buf) > 0:
            t[2] = 1 # shift
            if len(state.stack) >= 2:
                t[1] = 1 # right-arc
                if state.stack[0] != 0:
                    t[0] = 1 # left-arc
        if len(state.stack) >= 2:
            t[3] = 1 # reduce
        return t

