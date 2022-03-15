# -*- coding: utf-8 -*-

import torch

class ParserOperations(object):
    def __init__(self):
        pass

    def get_vocab_transitions(self):
        raise NotImplementedError()

    def get_state_class(self):
        raise NotImplementedError()

    @classmethod
    def read_seq(self, in_file, vocab):
        raise NotImplementedError()


class ArcStandardSwapOps(ParserOperations):
    def __init__(self):
        super(ArcStandardSwapOps, self).__init__()
        self.vocab_transitions = ['L', 'R', 'S', 'H']

    def get_vocab_transitions(self):
        return self.vocab_transitions

    def get_state_class(self):
        return ArcStandardSwapState

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


class ArcStandardOps(ParserOperations):
    def __init__(self):
        super(ArcStandardOps, self).__init__()
        self.vocab_transitions = ['L', 'R', 'S']

    def get_vocab_transitions(self):
        return self.vocab_transitions

    def get_state_class(self):
        return ArcStandardState

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


class ArcEagerOps(ParserOperations):
    def __init__(self):
        super(ArcEagerOps, self).__init__()
        self.vocab_transitions = ['LA', 'RA', 'SH', 'RE']

    def get_vocab_transitions(self):
        return self.vocab_transitions

    def get_state_class(self):
        return ArcEagerState

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


class EisnerOps(ParserOperations):
    def __init__(self):
        super(EisnerOps, self).__init__()
        self.vocab_transitions = ['L', 'R', 'S', 'T']

    def get_vocab_transitions(self):
        return self.vocab_transitions

    def get_state_class(self):
        return EisnerState

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
            elif line[0].startswith('T'):
                assert line[0] == 'Trans'
                seq.append(3)
                arcs.append(0)
        return gold_seq


## this class keeps parser state information
class State(object):
    def __init__(self):
        pass

    # build partially constructed graph
    def build_graph(self,mask,device,bert_label):
        raise NotImplementedError()

    def get_graph(self):
        raise NotImplementedError()

    #required features for graph output mechanism (exist classifier)
    def feature(self):
        raise NotImplementedError()

    # required features for graph output mechanism (relation classifier)
    def feature_label(self):
        raise NotImplementedError()

    # update state
    def update(self,act,rel=None):
        raise NotImplementedError()

    # legal actions at evaluation time
    def legal_act(self):
        raise NotImplementedError()

    # check whether the dependency tree is completed or not.
    def finished(self):
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError()


class ArcStandardSwapState(State):
    def __init__(self, mask, device, bert_label=None, input_graph=False):
        super(ArcStandardSwapState, self).__init__()
        self.tok_buffer = mask.nonzero().squeeze(1)
        self.tok_stack = torch.zeros(len(self.tok_buffer)+1).long().to(device)
        self.tok_stack[0] = 1
        self.buf = [i+1 for i in range(len(self.tok_buffer))]
        self.stack = [0]
        self.head = [[-1, -1] for _ in range(len(self.tok_buffer)+1)]
        self.dict = {0:"LEFTARC", 1:"RIGHTARC" ,2:"SHIFT", 3:"SWAP"}
        self.graph,self.label,self.convert = self.build_graph(mask,device,bert_label)
        self.input_graph=input_graph

    def build_graph(self,mask,device,bert_label):
        graph = torch.zeros((len(mask),len(mask))).long().to(device)
        label = torch.ones(len(mask) * bert_label).long().to(device)
        offset = self.tok_buffer.clone()
        convert = {0:1}
        convert.update({i+1:off.item() for i,off in enumerate(offset)})
        convert.update({len(convert):len(mask)})
        for i in range(len(offset)-1):
            graph[offset[i],offset[i]+1:offset[i+1]] = 1
            graph[offset[i]+1:offset[i+1],offset[i]] = 2
        label[offset] = 0
        label[:2] = 0
        del offset
        return graph,label,convert

    def get_graph(self):
        return self.graph,self.label

    def feature(self):
        return torch.cat((self.tok_stack[1].unsqueeze(0),self.tok_stack[0].unsqueeze(0)
                          ,self.tok_buffer[0].unsqueeze(0)))

    def feature_label(self):
        return torch.cat((self.tok_stack[1].unsqueeze(0),self.tok_stack[0].unsqueeze(0)))

    def update(self,act,rel=None):
        act = self.dict[act.item()]
        if not self.finished():
            if act == "SHIFT":
                self.stack = [self.buf[0]] + self.stack
                self.buf = self.buf[1:]
                self.tok_buffer = torch.roll(self.tok_buffer,-1,dims=0).clone()
                self.tok_stack = torch.roll(self.tok_stack,1,dims=0).clone()
                self.tok_stack[0] = self.tok_buffer[-1].clone()
            elif act == "LEFTARC":
                self.head[self.stack[1]] = [self.stack[0], rel.item()]
                if self.input_graph:
                    self.graph[self.convert[self.stack[0]],self.convert[self.stack[1]]] = 1
                    self.graph[self.convert[self.stack[1]],self.convert[self.stack[0]]] = 2
                    self.label[self.convert[self.stack[1]]] = rel
                self.stack = [self.stack[0]] + self.stack[2:]
                self.tok_stack = torch.cat(
                    (self.tok_stack[0].unsqueeze(0),torch.roll(self.tok_stack[1:],-1,dims=0))).clone()
            elif act == "RIGHTARC":
                self.head[self.stack[0]] = [self.stack[1], rel.item()]
                if self.input_graph:
                    self.graph[self.convert[self.stack[1]],self.convert[self.stack[0]]] = 1
                    self.graph[self.convert[self.stack[0]],self.convert[self.stack[1]]] = 2
                    self.label[self.convert[self.stack[0]]] = rel
                self.stack = self.stack[1:]
                self.tok_stack = torch.roll(self.tok_stack,-1,dims=0).clone()
            elif act == "SWAP":
                self.buf = [self.stack[1]] + self.buf
                self.stack = [self.stack[0]] + self.stack[2:]
                self.tok_stack = torch.cat(
                    (self.tok_stack[0].unsqueeze(0), torch.roll(self.tok_stack[1:], -1, dims=0))).clone()
                self.tok_buffer = torch.roll(self.tok_buffer, 1, dims=0).clone()
                self.tok_buffer[0] = self.tok_stack[-1]

    def legal_act(self):
        t = [0,0,0,0]
        if len(self.stack) >= 2 and self.stack[1] != 0:
            t[0] = 1
        if len(self.stack) >= 2 and self.stack[0] != 0:
            t[1] = 1
        if len(self.buf) > 0:
            t[2] = 1
        if len(self.stack) >= 2 and 0 < self.stack[1] < self.stack[0]:
            t[3] = 1
        return t

    def finished(self):
        return len(self.stack) == 1 and len(self.buf) == 0

    def __repr__(self):
        return "State:\nConvert:{}\n Graph:{}\n,Label:{}\nHead:{}\n".\
            format(self.convert,self.graph,self.label,self.head)


class ArcStandardState(State):
    def __init__(self, mask, device, bert_label=None, input_graph=False):
        super(ArcStandardState, self).__init__()
        self.tok_buffer = mask.nonzero().squeeze(1)
        self.tok_stack = torch.zeros(len(self.tok_buffer)+1).long().to(device)
        self.tok_stack[0] = 1
        self.buf = [i+1 for i in range(len(self.tok_buffer))]
        self.stack = [0]
        self.head = [[-1, -1] for _ in range(len(self.tok_buffer)+1)]
        self.dict = {0:"LEFTARC", 1:"RIGHTARC" ,2:"SHIFT"}
        self.graph,self.label,self.convert = self.build_graph(mask,device,bert_label)
        self.input_graph=input_graph

    def build_graph(self,mask,device,bert_label):
        graph = torch.zeros((len(mask),len(mask))).long().to(device)
        label = torch.ones(len(mask) * bert_label).long().to(device)
        offset = self.tok_buffer.clone()
        convert = {0:1}
        convert.update({i+1:off.item() for i,off in enumerate(offset)})
        convert.update({len(convert):len(mask)})
        for i in range(len(offset)-1):
            graph[offset[i],offset[i]+1:offset[i+1]] = 1
            graph[offset[i]+1:offset[i+1],offset[i]] = 2
        label[offset] = 0
        label[:2] = 0
        del offset
        return graph,label,convert

    def get_graph(self):
        return self.graph,self.label

    def feature(self):
        return torch.cat((self.tok_stack[1].unsqueeze(0),self.tok_stack[0].unsqueeze(0)
                          ,self.tok_buffer[0].unsqueeze(0)))

    def feature_label(self):
        return torch.cat((self.tok_stack[1].unsqueeze(0),self.tok_stack[0].unsqueeze(0)))

    def update(self,act,rel=None):
        act = self.dict[act.item()]
        if not self.finished():
            if act == "SHIFT":
                self.stack = [self.buf[0]] + self.stack
                self.buf = self.buf[1:]
                self.tok_buffer = torch.roll(self.tok_buffer,-1,dims=0).clone()
                self.tok_stack = torch.roll(self.tok_stack,1,dims=0).clone()
                self.tok_stack[0] = self.tok_buffer[-1].clone()
            elif act == "LEFTARC":
                self.head[self.stack[1]] = [self.stack[0], rel.item()]
                if self.input_graph:
                    self.graph[self.convert[self.stack[0]],self.convert[self.stack[1]]] = 1
                    self.graph[self.convert[self.stack[1]],self.convert[self.stack[0]]] = 2
                    self.label[self.convert[self.stack[1]]] = rel
                self.stack = [self.stack[0]] + self.stack[2:]
                self.tok_stack = torch.cat(
                    (self.tok_stack[0].unsqueeze(0),torch.roll(self.tok_stack[1:],-1,dims=0))).clone()
            elif act == "RIGHTARC":
                self.head[self.stack[0]] = [self.stack[1], rel.item()]
                if self.input_graph:
                    self.graph[self.convert[self.stack[1]],self.convert[self.stack[0]]] = 1
                    self.graph[self.convert[self.stack[0]],self.convert[self.stack[1]]] = 2
                    self.label[self.convert[self.stack[0]]] = rel
                self.stack = self.stack[1:]
                self.tok_stack = torch.roll(self.tok_stack,-1,dims=0).clone()

    def legal_act(self):
        t = [0,0,0]
        if len(self.stack) >= 2 and self.stack[1] != 0:
            t[0] = 1
        if len(self.stack) >= 2 and self.stack[0] != 0:
            t[1] = 1
        if len(self.buf) > 0:
            t[2] = 1
        return t

    def finished(self):
        return len(self.stack) == 1 and len(self.buf) == 0

    def __repr__(self):
        return "State:\nConvert:{}\n Graph:{}\n,Label:{}\nHead:{}\n".\
            format(self.convert,self.graph,self.label,self.head)

class EisnerState(State):
    def __init__(self, mask, device, bert_label=None, input_graph=False):
        super(EisnerState, self).__init__()
        self.tok_buffer = mask.nonzero().squeeze(1)
        self.tok_stack = torch.zeros(len(self.tok_buffer)+1).long().to(device)
        self.tok_stack[0] = 1
        self.buf = [i+1 for i in range(len(self.tok_buffer))]
        self.stack = [0]
        self.head = [[-1, -1] for _ in range(len(self.tok_buffer)+1)]
        self.dict = {0:"LEFTARC", 1:"RIGHTARC" ,2:"SHIFT", 3:"TRANS"}
        self.graph,self.label,self.convert = self.build_graph(mask,device,bert_label)
        self.input_graph=input_graph

    def build_graph(self,mask,device,bert_label):
        graph = torch.zeros((len(mask),len(mask))).long().to(device)
        label = torch.ones(len(mask) * bert_label).long().to(device)
        offset = self.tok_buffer.clone()
        convert = {0:1}
        convert.update({i+1:off.item() for i,off in enumerate(offset)})
        convert.update({len(convert):len(mask)})
        for i in range(len(offset)-1):
            graph[offset[i],offset[i]+1:offset[i+1]] = 1
            graph[offset[i]+1:offset[i+1],offset[i]] = 2
        label[offset] = 0
        label[:2] = 0
        del offset
        return graph,label,convert

    def get_graph(self):
        return self.graph,self.label

    def feature(self):
        return torch.cat((self.tok_stack[1].unsqueeze(0),self.tok_stack[0].unsqueeze(0)
                          ,self.tok_buffer[0].unsqueeze(0)))

    def feature_label(self):
        return torch.cat((self.tok_stack[1].unsqueeze(0),self.tok_stack[0].unsqueeze(0)))

    def update(self,act,rel=None):
        act = self.dict[act.item()]
        print(act)
        if not self.finished():
            if act == "SHIFT":
                self.stack = ['X_'+str(self.buf[0])] + self.stack
                self.buf = self.buf[1:]
                self.tok_buffer = torch.roll(self.tok_buffer,-1,dims=0).clone()
                self.tok_stack = torch.roll(self.tok_stack,1,dims=0).clone()
                self.tok_stack[0] = self.tok_buffer[-1].clone()
            elif act == "LEFTARC":
                self.head[int(self.stack[1].strip('Y_'))] = [int(self.stack[0].strip('X_')), rel.item()]
                if self.input_graph:
                    self.graph[self.convert[int(self.stack[0].strip('X_'))],self.convert[int(self.stack[1].strip('Y_'))]] = 1
                    self.graph[self.convert[int(self.stack[1].strip('Y_'))],self.convert[int(self.stack[0].strip('X_'))]] = 2
                    self.label[self.convert[int(self.stack[1].strip('Y_'))]] = rel
                self.stack = [self.stack[0]] + self.stack[2:]
                self.tok_stack = torch.cat(
                    (self.tok_stack[0].unsqueeze(0),torch.roll(self.tok_stack[1:],-1,dims=0))).clone()
            elif act == "RIGHTARC":
                if self.stack[1] == 0:
                    a, b = int(self.stack[0].strip('Y_')), self.stack[1]
                else:
                    a, b = int(self.stack[0].strip('Y_')), int(self.stack[1].strip('Y_'))
                self.head[a] = [b, rel.item()]
                if self.input_graph:
                    self.graph[self.convert[b],self.convert[a]] = 1
                    self.graph[self.convert[a],self.convert[b]] = 2
                    self.label[self.convert[a]] = rel
                self.stack = self.stack[1:]
                self.tok_stack = torch.roll(self.tok_stack,-1,dims=0).clone()
            elif act == "TRANS":
                self.stack = ["Y_"+self.stack[0].strip("X_")] + self.stack[1:]

    def legal_act(self):
        # TODO: add checks for X and Y for left and right arc - is the order right?
        print(self.stack)
        t = [0,0,0,0]
        if len(self.stack) >= 2 and self.stack[1] != 0:
            t[0] = 1
        if len(self.stack) >= 2 and self.stack[0] != 0 and self.stack[0][:2] == 'X_':
            t[1] = 1
        if len(self.buf) > 0 and self.stack[0][:2] == 'Y_':
            t[2] = 1
        if len(self.stack) >= 1 and self.stack[0] != 0 and self.stack[0][:2] == 'X_':
            t[3] = 1
        return t

    def finished(self):
        return len(self.stack) == 1 and len(self.buf) == 0

    def __repr__(self):
        return "State:\nConvert:{}\n Graph:{}\n,Label:{}\nHead:{}\n".\
            format(self.convert,self.graph,self.label,self.head)

class ArcEagerState(State):
    def __init__(self, mask, device, bert_label=None, input_graph=False):
        super(ArcEagerState, self).__init__()
        self.tok_buffer = mask.nonzero().squeeze(1)
        self.tok_stack = torch.zeros(len(self.tok_buffer)+1).long().to(device)
        self.tok_stack[0] = 1
        self.buf = [i+1 for i in range(len(self.tok_buffer))]
        self.stack = [0]
        self.head = [[-1, -1] for _ in range(len(self.tok_buffer)+1)]
        self.dict = {0:"LEFTARC", 1:"RIGHTARC" ,2:"SHIFT", 3:"REDUCE"}
        self.graph,self.label,self.convert = self.build_graph(mask,device,bert_label)
        self.input_graph=input_graph

    def build_graph(self,mask,device,bert_label):
        graph = torch.zeros((len(mask),len(mask))).long().to(device)
        label = torch.ones(len(mask) * bert_label).long().to(device)
        offset = self.tok_buffer.clone()
        convert = {0:1}
        convert.update({i+1:off.item() for i,off in enumerate(offset)})
        convert.update({len(convert):len(mask)})
        for i in range(len(offset)-1):
            graph[offset[i],offset[i]+1:offset[i+1]] = 1
            graph[offset[i]+1:offset[i+1],offset[i]] = 2
        label[offset] = 0
        label[:2] = 0
        del offset
        return graph,label,convert

    def get_graph(self):
        return self.graph,self.label

    def feature(self):
        return torch.cat((self.tok_stack[1].unsqueeze(0),self.tok_stack[0].unsqueeze(0)
                          ,self.tok_buffer[0].unsqueeze(0)))

    def feature_label(self):
        return torch.cat((self.tok_stack[0].unsqueeze(0),self.tok_buffer[0].unsqueeze(0)))

    def update(self,act,rel=None):
        act = self.dict[act.item()]
        if not self.finished():
            if len(self.buf) == 0:
                act = "REDUCE"
            if act == "SHIFT":
                self.stack = [self.buf[0]] + self.stack
                self.buf = self.buf[1:]
                self.tok_buffer = torch.roll(self.tok_buffer,-1,dims=0).clone()
                self.tok_stack = torch.roll(self.tok_stack,1,dims=0).clone()
                self.tok_stack[0] = self.tok_buffer[-1].clone()
            elif act == "LEFTARC":
                self.head[self.stack[0]] = [self.buf[0], rel.item()]
                if self.input_graph:
                    self.graph[self.convert[self.buf[0]],self.convert[self.stack[0]]] = 1
                    self.graph[self.convert[self.stack[0]],self.convert[self.buf[0]]] = 2
                    self.label[self.convert[self.stack[0]]] = rel
                self.stack = self.stack[1:]
                self.tok_stack = torch.roll(self.tok_stack,-1,dims=0).clone()
            elif act == "RIGHTARC":
                self.head[self.buf[0]] = [self.stack[0], rel.item()]
                if self.input_graph:
                    self.graph[self.convert[self.stack[0]],self.convert[self.buf[0]]] = 1
                    self.graph[self.convert[self.buf[0]],self.convert[self.stack[0]]] = 2
                    self.label[self.convert[self.buf[0]]] = rel
                self.stack = [self.buf[0]] + self.stack
                self.buf = self.buf[1:]
                self.tok_buffer = torch.roll(self.tok_buffer,-1,dims=0).clone()
                self.tok_stack = torch.roll(self.tok_stack,1,dims=0).clone()
                self.tok_stack[0] = self.tok_buffer[-1].clone()
            elif act == "REDUCE":
                self.stack = self.stack[1:]
                self.tok_stack = torch.roll(self.tok_stack,-1,dims=0).clone()

    def legal_act(self):
        t = [0,0,0,0]
        if len(self.buf) > 0:
            t[2] = 1 # shift
            if len(self.stack) >= 2:
                t[1] = 1 # right-arc
                if self.stack[0] != 0:
                    t[0] = 1 # left-arc
        if len(self.stack) >= 2:
            t[3] = 1 # reduce
        return t

    def finished(self):
        return len(self.stack) == 1 and len(self.buf) == 0

    def __repr__(self):
        return "State:\nConvert:{}\n Graph:{}\n,Label:{}\nHead:{}\n".\
            format(self.convert,self.graph,self.label,self.head)
