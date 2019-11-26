import numpy as np
from parse import parse
from vqr import ResidualPQ
from vqp import PQ
import os


dim, Ks, r = 4, 64, 4



def quantize(filename):
    id, dataset, model, I, O, W, H = parse("{:d}_{}_{}_conv_I{:d}_O{:d}_W{:d}_H{:d}.txt", filename)
    X = np.genfromtxt("activations/{}".format(filename), dtype=np.float32)
    print(X.shape)


    M = I // dim
    rq = ResidualPQ([PQ(M=M, Ks=Ks) for _ in range(r)])
    rq.fit(X, iter=20, save_codebook=True, save_dir='activations',
                  dataset_name=filename+".M{}_Ks{}_r{}_dim{}".format(M, Ks, r, dim))


files = os.listdir('activations')
for filename in files:
    if filename.endswith(".txt"):
        print(filename)
        quantize(filename)