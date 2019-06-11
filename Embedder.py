#!/usr/bin/env python

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os, sys, csv, time
import numpy as np
import pylab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from MulticoreTSNE import MulticoreTSNE as TSNE

from pathos.multiprocessing import ProcessingPool as Pool
from kpal.klib import Profile

import pandas as pd

import umap
from Bio import SeqIO
from io import StringIO
import sys

import textwrap as _textwrap
import argparse


t1 = time.time()

################################################################################
#                                FUNCTION ZOO                                  #
################################################################################

def main(args):
    seqs = args.s

    label_param = args.l

    k_len = args.k

    freqs = args.kf

    threads = args.nt

    perp = args.t

    neighbors = args.u

    dimensionality = args.d

    labels = None

    ### If desired, compute kmer embedding; if necessary, modify labels to fit
    label_flag = False
    if seqs != None:
        if label_param == None:
            kmer_array = convert_seqs(seqs, k_len, threads)
        else:
            label_flag = True
            kmer_array, labels = convert_seqs(seqs, k_len, threads, label_param)
            labels = [int(x) for x in labels]
    #Save your stuff to output files
        np.savetxt('Embedder_' + str(k_len) + 'mer_freqs.csv', kmer_array, delimiter=',')
        if label_flag == True:
            np.savetxt('Embedder_Modified_Labels.csv', labels, delimiter=',')

    if perp == None and neighbors == None:
        print("No dimensionality reduction requested; frequency vector construction complete.")
        print("Process completed in " + str(time.time() - t1) + " seconds.")
        sys.exit()

    if perp != None:
        #OPT: Load kmer Embedding and labels
        if labels is None:
            #If labels is a string, means it's a path to a file
            if label_param.split('.')[-1] == 'npy':
                try:
                    labels = np.load(label_param)
                except:
                    print('Something went wrong loading your .npy labels. Please specify full path.')
            if label_param.split('.')[-1] == 'txt' or label_param.split('.')[-1] == 'csv':
                try:
                    labels = np.loadtxt(label_param, delimiter=',')
                except:
                    print('Something went wrong loading your .csv/.txt labels. Please modify code to include proper delimiter and/or specify full path.')


        if freqs != None:
            if freqs.split('.')[-1] == 'npy':
                try:
                    kmer_array = np.load(freqs)
                except:
                    print('Something went wrong loading your .npy embedding. Please specify full path.')
            if freqs.split('.')[-1] == 'txt' or freqs.split('.')[-1] == 'csv':
                try:
                    kmer_array = np.loadtxt(freqs, delimiter=',')
                except:
                    print('Something went wrong loading your .csv/.txt embedding. Please modify code to include proper delimiter and/or specify full path.')

        #Do your tSNE

        # tsne_embed(dims, perplexity, threads, kmer_arr):

        tsne_embedding = tsne_embed(dimensionality, perp, threads, kmer_array)

        #Save your tSNE
        np.save('Embedder_TSNE_p' + str(perp) + '.npy', tsne_embedding)

        #Plot
        if dimensionality == 2:
            plot_meta_2D(tsne_embedding, 't-SNE', labels)
        if dimensionality == 3:
            plot_meta_3D(tsne_embedding, 't-SNE', labels)
        if dimensionality > 3:
            print("Invalid dimensionality for plotting. Sorry.")

    if neighbors != None:

        if labels is None:
            #Load Labels
            if label_param.split('.')[-1] == 'npy':
                try:
                    labels = np.load(label_param)
                except:
                    print('Something went wrong loading your .npy labels. Please specify full path.')
            if label_param.split('.')[-1] == 'txt' or label_param.split('.')[-1] == 'csv':
                try:
                    labels = np.loadtxt(label_param, delimiter=',')
                except:
                    print('Something went wrong loading your .csv/.txt labels. Please modify code to include proper delimiter and/or specify full path.')

        #OPT: Load kmer embedding

        if freqs != None:
            if freqs.split('.')[-1] == 'npy':
                try:
                    kmer_array = np.load(freqs)
                except:
                    print('Something went wrong loading your .npy embedding. Please specify full path.')
            if freqs.split('.')[-1] == 'txt' or freqs.split('.')[-1] == 'csv':
                try:
                    kmer_array = np.loadtxt(freqs, delimiter=',')
                except:
                    print('Something went wrong loading your .csv/.txt embedding. Please modify code to include proper delimiter and/or specify full path.')
        #Do your UMAP

        ### umap_embed(neighbors, dims, metric, kmer_arr):

        umap_embedding = umap_embed(neighbors, dimensionality, 'canberra', kmer_array)
        #Save your UMAP
        np.save('Embedder_UMAP_n' + str(neighbors) + '.npy', umap_embedding)
        #Plot
        if dimensionality == 2:
            plot_meta_2D(umap_embedding, 'UMAP', labels)
        if dimensionality == 3:
            plot_meta_3D(umap_embedding, 'UMAP', labels)
        else:
            print("Invalid dimensionality for plotting. Sorry.")

#### calc_kmer_freqs and chunk_sequence are from Patrick West's EukRep:
#### https://github.com/patrickwest/EukRep

def calc_kmer_freqs(split_seqs, kmer_size):
    '''
    Use kpal to calculate kmer frequencies for split sequences
    '''

    kmer_freqs = []




    def kmer_singleton(seq, kmer_size):
        if type(seq) != str:
            seq = str(seq)
        seqlist = [seq]

        ktable = Profile.from_sequences(seqlist, kmer_size, name=None)

        if len(seq) < 3000:
            if ktable.total >= len(seq)/2:
                ktable.counts = [count/ktable.total for count in ktable.counts]
                return ktable.counts

        else:
            if ktable.total >= 1500:
                ktable.counts = [count/ktable.total for count in ktable.counts]
                return ktable.counts
    """
    for seq in split_seqs:
        temp_list = []

        #for some reason this kmer counter function only works on iterable(str) type objects.
        temp_list.append(str(seq))
        ktable = Profile.from_sequences(temp_list, kmer_size, name=None)

        #skip sequences with a lot of Ns/characters besides A|T|C|G
        if len(str(seq)) < 3000:
            if ktable.total >= len(str(seq))/2:
                ktable.counts = [count/ktable.total for count in ktable.counts]
                kmer_freqs.append(ktable.counts)
        else:
            if ktable.total >= 1500:
                ktable.counts = [count/ktable.total for count in ktable.counts]
                kmer_freqs.append(ktable.counts)
    """

    kmer_freqs = list(map(lambda x: kmer_singleton(x, kmer_size), split_seqs))

    return kmer_freqs

def chunk_sequence(sequence, min_size, max_size):
    '''
    Cut sequences longer than 5kb into 5kb chunks and exclude trailing sequences
    if shorter than user specified min_length
    '''

    split_seqs = []
    while True:
        chunk = sequence.read(max_size)
        if len(chunk) >= min_size:
            split_seqs.append(chunk)
        else:
            break

    return split_seqs

def plot_meta_3D(embedding, method, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    cmspec = plt.get_cmap("nipy_spectral")

    ax.scatter(embedding[:,0], \
              embedding[:,1], \
              embedding[:,2], \
              c=labels, cmap=cmspec)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    pylab.title("3D Embedding: " + str(method))
    pylab.show()
    pylab.savefig(str(method) + '_Embedding.png')
    return

def umap_embed(neighbors, dims, metric, kmer_arr):
    u_embedding = umap.UMAP(n_neighbors=neighbors,
                          min_dist=0.3,
                          n_components=dims,
                          metric=metric).fit_transform(kmer_arr)
    return u_embedding

def tsne_embed(dims, perplexity, threads, kmer_arr):
    tsne = TSNE(n_jobs=threads, n_components=dims, perplexity=perplexity)
    return tsne.fit_transform(kmer_arr)

def plot_meta_2D(embedding, method, labels):
    colors = [int(i % (max(labels))) for i in labels]
    cmspec = plt.get_cmap("nipy_spectral")
    pylab.scatter(embedding[:,0], embedding[:,1],c=colors,cmap=cmspec)
    pylab.title(str(method) + "2D Plot")
    pylab.show()
    pylab.savefig(str(method) + '_Embedding.png')
    return

def train_and_score(em, test_size, kernel):
    X_train, X_test, y_train, y_test = train_test_split(em, contig_labels, test_size=test_size)
    Uclf = svm.SVC(kernel=kernel, C=1).fit(X_train, y_train)
    return Uclf.score(X_test,y_test)

def run_nonsense(neighbors):
    u_euc_scores = []
    u_mah_scores = []
    u_cor_scores = []
    tsne_scores = []

    for i in range(5):
        euc_embed = umap_embed(neighbors, 'euclidean')
        u_euc_scores.append(train_and_score(euc_embed, 0.4, 'rbf'))

        mah_embed = umap_embed(neighbors, 'mahalanobis')
        u_mah_scores.append(train_and_score(mah_embed, 0.4, 'rbf'))

        cor_embed = umap_embed(neighbors, 'correlation')
        u_cor_scores.append(train_and_score(cor_embed, 0.4, 'rbf'))

        tsne_embedding = tsne_embed_fun()
        tsne_scores.append(train_and_score(tsne_embedding, 0.4, 'rbf'))
        print("loop " + str(i) + " complete.")
        return u_euc_scores, u_mah_scores, u_cor_scores, tsne_scores


def u_run_nonsense(metrics):
    scores = []
    for metric in metrics:
        print('---------------------------------------------------------------')
        print('                       ' + metric + '                          ')
        print('---------------------------------------------------------------')
        scores.append([])
        for i in range(9):
            u_embedding = umap_embed_3(10+10*i, metric)
            plot_meta_3(u_embedding, 'UMAP', metric + ' (n=' + str(10+10*i)+')')
            score = train_and_score(u_embedding, 0.4, 'rbf')
            scores[-1].append(score)
            print(score)
        print(metric + " average classification score: ", np.mean(scores[-1]))
    return scores

def convert_seqs(seqfile, k, threads, labels=None):
    #Returns an array of kmer frequency vectors (OPTIONAL: new label list based on original to compensate for chunking)

    kmer_list = []
    if labels != None:
        new_labels = []

        ext = labels.split('.')[-1]
        if ext == 'npy':
            contig_labels = np.load(labels)
        elif ext == 'csv' or ext == 'txt':
            contig_labels = np.loadtxt(labels, delimiter=',')

    max_size = 5000
    min_size = 1000



    """

    for index, record in enumerate(SeqIO.parse(seqfile, "fasta")):
        s = StringIO(str(record.seq))
        split_seqs = []
        kmer_freqs = []

        #split sequence into 5kb max_size chunks
        split_seqs = chunk_sequence(s, min_size, max_size)

        #Calculate kmer frequences for each chunk
        kmer_freqs = calc_kmer_freqs(split_seqs, k)

        kmer_list.append(kmer_freqs)
    """

    #All of what you see below (up to and including the kmer_list declaration) is an alternative to the block comment code written above;
    #Hopefully this successfully parallelizes the kmer frequency vector creation step. Let's see

    p = Pool(threads)

    def process_contig(rec, min_size, max_size, k):
        s = StringIO(str(rec.seq))

        split_seqs = chunk_sequence(s, min_size, max_size)

        return calc_kmer_freqs(split_seqs, k)

    kmer_list = list(p.map(lambda x: process_contig(x, min_size, max_size, k), SeqIO.parse(seqfile, 'fasta')))


    #kmer array comes out in a weird nested list; fix its shape
    flatter_list = []

    for index, contig in enumerate(kmer_list):
        for i, kmer_freq in enumerate(contig):
            #Append correct number of corresponding labels to y
            if labels != None:
                new_labels.append(contig_labels[index])

            flatter_list.append(kmer_freq)

    if labels != None:
        return np.array(flatter_list), np.array(new_labels)
    else:
        return np.array(flatter_list)

def Parse_Args(args=None):
    class LineWrapRawTextHelpFormatter(argparse.RawDescriptionHelpFormatter):
        def _split_lines(self, text, width):
            text = self._whitespace_matcher.sub(' ', text).strip()
            return _textwrap.wrap(text, width)


    parser=argparse.ArgumentParser(description='Two functions:\n \
                        1. Take a set of sequences (-s) and project into a kmer frequency array (-k). See code for method details.\n \
                        2. Take a given kmer embedding (-ke) and reduce its dimensionality with either t-SNE (-t [perplexity]) or UMAP (-u [n_neighbors]).\n\n \
                        In order to perform both of these tasks in sequential order, simply provide values for all parameters except (-ke).\n \
                        In addition, please specify the dimensionality of your projected space (2 or 3 recommended; >4 will not plot) as well as your labels file (-l).',\
                        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-s', metavar='sequences', default=None, type=str, help="Contig/scaffolds file (FASTA FORMAT ONLY)")

    parser.add_argument('-l', metavar='labels', default=None, type=str, help="Labels file (npy or txt).")

    parser.add_argument('-k', metavar='kmer length', default=None, type=int, help="Any number (hopefully above 3 and less than 9, unless you have a HUGE computer)")

    parser.add_argument('-kf', metavar='kmer freqs', default=None, type=str, help="Path to kmer frequency array file (.npy or .csv)")

    parser.add_argument('-t', metavar='tSNE (perplexity)', default=None, type=int, help="Enables barnes-hut t-SNE. Please also provide thread num to use. (-tn)")

    parser.add_argument('-u', metavar='UMAP (n_neighbors)', default=None, type=int, help="Enables UMAP (Universal Manifold Approximation and Projection).")

    parser.add_argument('-d', metavar='Dimensionality', default=None, type=int, help="Dimensions to reduce embedding to with t-SNE or UMAP.")

    parser.add_argument('-nt', metavar='Threads', default=None, type=int, help="Number of threads to use for multicore TSNE.")

    return parser.parse_args()

if __name__ == "__main__":
    args = Parse_Args(sys.argv[1:])
    main(args)
