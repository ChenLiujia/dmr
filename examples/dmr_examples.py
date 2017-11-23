
import sys, os
root_dir = os.path.join(os.path.dirname(os.path.realpath('__file__')), '..')
sys.path.append(root_dir)
import dmr
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def init_dmr(docfilepath, vecfilepath, K=5, sigma=1.0, beta=0.01):
    corpus = dmr.Corpus.read(docfilepath)
    vecs = dmr.Corpus.read(vecfilepath, dtype=float)
    vecs = np.array([[v for v in vec] for vec in vecs], dtype=np.float32)
    voca = dmr.Vocabulary()
    docs = voca.read_corpus(corpus)
    lda = dmr.DMR(K, sigma, beta, docs, vecs, voca.size())
    return corpus, voca, docs, vecs, lda

docfilepath = os.path.join(root_dir, 'dat', 'DMR.doc.dat')
vecfilepath = os.path.join(root_dir, 'dat', 'DMR.vec.dat')

# learning
corpus, voca, docs, vecs, lda = init_dmr(docfilepath, vecfilepath)
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
    for doc, vec, td in zip(corpus, vecs, tdist):
        if doc[0].startswith(first_letter):
            print("DOC", "Words: ", doc, "Max topic: ", np.argmax(td), "Max prob.: ", np.max(td))
            print("ALPHA", np.dot(vec, lda.Lambda.T))