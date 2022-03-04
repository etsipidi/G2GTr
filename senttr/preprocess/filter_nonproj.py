import sys

filein = sys.argv[1]
fileout = sys.argv[2]

def processlines(lines, fout):
    projective = True

    # find decendents
    words = ['ROOT']
    for line in lines:
        words += [line[1]]

    children = [[] for i in range(len(words))]
    for i, line in enumerate(lines):
        try:
            parent = int(line[6])
            relation = line[7]
            children[parent] += [(relation, i+1)]
        except Exception:
            print(line)

    decendents = [set([child[1] for child in children[i]]) for i in range(len(words))]

    change = True
    while change:
        change = False
        for i in range(len(decendents)):
            update = []
            for d in decendents[i]:
                for d1 in decendents[d]:
                    if d1 not in decendents[i]:
                        update += [d1]
            if len(update) > 0:
                decendents[i].update(update)
                change = True

    for i, node in enumerate(children):
        for child in node:
            childid = child[1]
            for j in range(min(childid, i)+1, max(childid, i)):
                if j not in decendents[i]:
                    projective = False
                    return 1

    if projective:
        fout.write(("%s\n\n" % ("\n".join(["\t".join(line) for line in lines]))))

    return 0

count = 0
nonproj = 0
lines = []

fout = open(fileout, 'w')
with open(filein, 'r') as fin:
    line = fin.readline()
    while line:
        if line.startswith('#'):
            line = fin.readline()
            continue
        line = line.strip().split()
        if len(line) > 0 and '-' in line[0]:
            line = fin.readline()
            continue

        if len(line) == 0:
            count += 1
            nonproj += processlines(lines, fout)
            lines = []
        else:
            lines += [line]

        line = fin.readline()

    if len(lines) > 0:
        processlines(lines)

fout.close()

print("%d trees processed, %d non-projective trees filtered out" % (count, nonproj))
