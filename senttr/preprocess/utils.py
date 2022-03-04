import logging
import os.path as op
from smart_open import smart_open
import pickle
from parserstate import ParserState, SwapParserState
from transition import ArcSwift, ArcEagerReduce, ArcEagerShift, ArcStandard, ArcHybrid, ArcStandardSwap
import numpy as np

transition_dims = ['action', 'n', 'rel', 'pos', 'fpos']
transition_pos = {v:i for i, v in enumerate(transition_dims)}
floatX = np.float32

def transsys_lookup(k):
    lookup = {"ASw": ArcSwift,
              "AER": ArcEagerReduce,
              "AES": ArcEagerShift,
              "ASd": ArcStandard,
              "AH" : ArcHybrid,
              "ASWAP": ArcStandardSwap,}
    return lookup[k]

def read_mappings(mappings_file, transsys, log=None):
    i = 0
    res = dict()
    res2 = dict()
    with smart_open(mappings_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("::"):
                currentkey = line[2:]
                res[currentkey] = dict()
                res2[currentkey] = []
                i = 0
            else:
                res[currentkey][line] = i
                res2[currentkey] += [line]
                i += 1

    res['action'] = {k: i for i, k in enumerate(transsys.actions_list())}
    res2['action'] = transsys.actions_list()

    return res, res2

def read_gold_parserstates(fin, transsys, fpos=False):
    def processlines(lines):
        arcs = [dict() for i in range(len(lines)+1)]

        pos = ["" for i in range(len(lines)+1)]
        fpos = ["" for i in range(len(lines)+1)]

        for i, line in enumerate(lines):
            pos[i+1] = line[3] # fine-grained
            fpos[i+1] = line[4]
            parent = int(line[6])
            relation = line[7]
            arcs[parent][i+1] = transsys.mappings['rel'][relation]

        if transsys == "ASWAP":
            res = [SwapParserState(["<ROOT>"] + lines, transsys=transsys, goldrels=arcs), pos]
        else:
            res = [ParserState(["<ROOT>"] + lines, transsys=transsys, goldrels=arcs), pos]
        if fpos:
            res += [fpos]
        else:
            res == [None]
        return res
    res = []

    lines = []
    line = fin.readline()
    while line:
        line = line.strip().split()

        if len(line) == 0:
            res += [processlines(lines)]
            lines = []
        else:
            lines += [line]

        line = fin.readline()

    if len(lines) > 0:
        res += [processlines(lines)]

    return res

def write_gold_trans(tpl, fout):
    state, pos, fpos = tpl
    transsys = state.transsys
    while len(state.transitionset()) > 0:
        t = transsys.goldtransition(state)

        fout.write("%s\n" % transsys.trans_to_str(t, state, pos, fpos))

        transsys.advance(state, t)

    fout.write("\n")
