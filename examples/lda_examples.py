import sys, os
root_dir = os.path.join(os.path.dirname(os.path.realpath('__file__')), '..')
sys.path.append(root_dir)
import dmr
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def init_lda(datfilepath, K=5, alpha=0.1, beta=0.01):
    corpus = dmr.Corpus.read(datfilepath)
    voca = dmr.Vocabulary()
    docs = voca.read_corpus(corpus)
    lda = dmr.LDA(K, alpha, beta, docs, voca.size())
    return corpus, voca, docs, lda

datfilepath = os.path.join(root_dir, 'dat', 'LDA.doc.dat')

# learning
corpus, voca, docs, lda = init_lda(datfilepath)
lda.learning(iteration=100, voca=voca)

# word probability of each topic
wdist = lda.word_dist_with_voca(voca)
for k in wdist:
    print("TOPIC", k)
    print("\t".join([w for w in wdist[k]]))
    print("\t".join(["%0.2f" % wdist[k][w] for w in wdist[k]]))

print()

# topic probability of each document
tdist = lda.topicdist()
for first_letter in ["a", "b", "c", "d", "e"]:
    for doc, td in zip(corpus, tdist):
        if doc[0].startswith(first_letter):
            print("DOC", "Words: ", doc, "Max topic: ", np.argmax(td), "Max prob.: ", np.max(td))