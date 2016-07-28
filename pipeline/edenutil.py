'''Imports'''

# essentials

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# read_data imports

import json
import operator
import itertools

# preprocess_data imports

from sklearn.feature_extraction.text import TfidfVectorizer
import cPickle as pickle
from datetime import datetime
import time

# cluster_data imports

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics.pairwise import cosine_similarity

# evaluate imports


'''Library functions'''

# read_data: stitch together data from various feeds


def read_data(fn):
    data = []
    for f in fn.split(','):
        print f
        with open("../datasets/raw-data/" + str(f) + ".json", 'rb') as infile:
            data.extend(json.load(infile))

    print "[EDEN I/O -- read_data] Data length: ", len(data)
    print "[EDEN I/O -- read_data] Data type: ", type(data)

    return json.dumps(data)

# preprocess_data: create vector space model of content


def preprocess_data(data):
    # stories = [doc["_source"]["content"] for doc in data]
    # ids = [doc["_id"] for doc in data]

    def format_entities(norm_ent):
        ents = []
        for ent in norm_ent:
            try:
                ents.append(ent['surface-form'])
            except:
                continue
        return " ".join(ents)

    d = [{"ids": doc["_id"],
          "first-published": doc["_source"]["first-published"],
          "title": doc["_source"]["title"],
          "summary": doc["_source"]["title"],
          "content": doc["_source"]["content"],
          "entities": format_entities(doc["_source"]["normalised-entities"])} for doc in data]

    df_story = pd.DataFrame(d)
    df_story['first-published-epoch'] = df_story['first-published'].apply(
        lambda x: int(datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ").strftime("%s")))
    df_story = df_story.sort_values(by='first-published-epoch')
    df_story = df_story.reset_index()

    vect = TfidfVectorizer(use_idf=True, norm='l2', sublinear_tf=True)
    vsm = vect.fit_transform(df_story['content'].values)
    vsm_arr = vsm.toarray()

    print "[EDEN I/O -- preprocess_data] VSM shape: ", vsm.shape
    print "[EDEN I/O -- preprocess_data] VSM type: ", type(vsm)

    df_story['vsm'] = [r for r in vsm_arr]

    return df_story

# cluster_data: cluster stories based on similarity


# reconsturct vsm: reconstruct np.ndarray from pd.Series
def recon_vsm(vsm_series):
    rows = vsm_series.shape[0]
    cols = vsm_series[0].shape[0]
    print "[EDEN I/O -- recon_vsm] (rows,cols): ", rows, cols
    vsm = np.zeros((rows, cols))

    for i, r in enumerate(vsm_series):
        vsm[i] = r

    # print "[EDEN I/O -- recon_vsm] vsm shape: ", vsm.shape
    # print "[EDEN I/O -- recon_vsm] vsm type: ", type(vsm)
    # print "[EDEN I/O -- recon_vsm] vsm[0] type: ", type(vsm[0])

    return vsm

# GAC Model


def mycosine(x1, x2):
    x1 = x1.reshape(1, -1)
    x2 = x2.reshape(1, -1)
    ans = 1 - cosine_similarity(x1, x2)
    return max(ans[0][0], 0)

# GAC utilities
def get_linkage_matrix(vsm):
    Z = linkage(vsm, method='average', metric=mycosine)
    c, coph_dists = cophenet(Z, pdist(vsm))
    print "cophenet test: ", c
    return Z

def get_cluster_size(vsm, Z):
    n = vsm.shape[0]
    for i, z in enumerate(Z):
        #         print i, z
        if z[2] > 0.8:
            print i, ": Min similarity reached"
            print Z[i]
            n_clus = n - i
            break
        if n * 0.5 == i:
            print i, ": Max reduction reached"
            print Z[i]
            n_clus = n - i
            break

    return n_clus

def get_cluster_order(Z):
    cluster_order_dict = {}
    cluster_order_list = []
    for i, z in enumerate(Z):
        # create pairs based on order
        cluster_order_dict[z[0]] = 2 * i
        cluster_order_dict[z[1]] = 2 * i + 1
        # add docs to list
        cluster_order_list.extend([z[0], z[1]])
    
    return [cluster_order_dict, cluster_order_list]


class GAC:

    def __init__(self, b = 10.0, p = 0.5, s = 0.2, t = 100, re = 5):
        self.labels_ = []
        self.b = b
        self.p = p
        self.s = s
        self.t = t
        self.re = re

    # cluster
    def fit(self, df_story):

        # basic GAC within sub_def
        # recon vsm
        vsm = recon_vsm(df_story['vsm'])
        # 1. linkage
        Z = get_linkage_matrix(vsm)
        n_clus = get_cluster_size(vsm, Z)
        print "[EDEN I/O -- cluster_data] n_clus: ", n_clus
        # 2. clusters
        clusters = fcluster(Z, n_clus, criterion='maxclust')
        df_story['clusters'] = clusters
        # 3. average centroids and clusters
        # get clustering order
        # [cluster_order_dict, cluster_order_list] = get_cluster_order(Z)

        # optional : fancy dendrogram
        ddata = fancy_dendrogram(
            Z,
            truncate_mode='lastp',
            p=58,
            leaf_rotation=90.,
            leaf_font_size=12.,
            show_contracted=True,
            annotate_above=10,
            max_d=0.62337  # useful in small plots so annotations don't overlap
        )

        print "[EDEN I/O -- cluster_data] clusters: ", clusters

        self.labels_ = clusters


    # creation dictionary and lists of ordering of clustering
    # from linkage matrix
    # anatomy of z: z[0] - first doc, z[1] - second doc
    # returns dict: key - doc, value - order

def basic_GAC(df_story):
    Z = get_linkage_matrix(recon_vsm(df_story['vsm']))
    n_clus = get_cluster_size(vsm, Z)
    clusters = fcluster(Z, n_clus, criterion='maxclust')
    df_story['cluster'] = clusters
    [cluster_order_dict, cluster_order_list] = get_cluster_order(Z)
    print len(df_story)
    df_update = df_story.groupby('cluster').apply(build_centroids).reset_index(drop=True)
    print len(df_update)

class GACTemporal:

    def __init__(self, b = 10.0, p = 0.5, s = 0.2, t = 100, re = 5):
        self.labels_ = []
        self.b = b
        self.p = p
        self.s = s
        self.t = t
        self.re = re

    # cluster
    def fit(self, df_story):

        # basic GAC within sub_def
        # recon vsm
        vsm = recon_vsm(df_story['vsm'])
        # 1. linkage
        Z = get_linkage_matrix(vsm)
        n_clus = get_cluster_size(vsm, Z)
        print "[EDEN I/O -- cluster_data] n_clus: ", n_clus
        # 2. clusters
        clusters = fcluster(Z, n_clus, criterion='maxclust')
        df_story['clusters'] = clusters
        # 3. average centroids and clusters
        # get clustering order
        # [cluster_order_dict, cluster_order_list] = get_cluster_order(Z)

        # optional : fancy dendrogram
        ddata = fancy_dendrogram(
            Z,
            truncate_mode='lastp',
            p=58,
            leaf_rotation=90.,
            leaf_font_size=12.,
            show_contracted=True,
            annotate_above=10,
            max_d=0.62337  # useful in small plots so annotations don't overlap
        )

        print "[EDEN I/O -- cluster_data] clusters: ", clusters

        self.labels_ = clusters


    # creation dictionary and lists of ordering of clustering
    # from linkage matrix
    # anatomy of z: z[0] - first doc, z[1] - second doc
    # returns dict: key - doc, value - order


def algo_select(algo='kmeans', params={}):
    print "[EDEN I/O -- algo_select] algo: ", algo
    print "[EDEN I/O -- algo_select] params: ", params
    if algo == 'kmeans':
        return KMeans(**params)
    elif algo == 'dbscan':
        return DBSCAN(**params)
    elif algo == 'meanshift':
        return MeanShift(**params)
    elif algo == 'gac':
        return GAC(**params)
    elif algo == 'gactemporal':
        return GACTemporal(**params)


def cluster_data(df_story, algo='kmeans'):
    print "[EDEN I/O -- cluster_data] algo: ", algo
    # here vsm is dense array (test performance)
    # maybe add column for csr_matrix as well

    if algo == 'gac':
        b = max(math.ceil((400.0 / 15836.0) * df_story.shape[0]),10)
        params = {'b': b,
                  'p': 0.5,
                  's': 0.2,
                  't': 100,
                  're': 5
                  }
        model = algo_select(algo, params)
        model.fit(df_story)

    elif algo == 'meanshift':
        vsm = recon_vsm(df_story['vsm'])
        params = {'bandwidth': estimate_bandwidth(vsm, n_samples=200)}
        model = algo_select(algo, params)
        model.fit(vsm)
    else:
        vsm = recon_vsm(df_story['vsm'])
        model = algo_select(algo)
        model.fit(vsm)

    # print "[EDEN I/O -- cluster_data.py] plot: cluster counts"
    # plot_cluster_counts(model, "Cluster counts using algorithm: " + str(algo))

    # print "[EDEN I/O -- cluster_data] model: ", model

    return model

# plotting functions for clusters

# plot: Reduce dimensionality of data and plot with cluster colors


def reduce_and_plot_clusters(X, model, title=""):
    X_reduced = TruncatedSVD().fit_transform(X)
    X_embedded = TSNE(learning_rate=100).fit_transform(X_reduced)

    n_clus = len(set(model.labels_.tolist()))

    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], marker="x",
                c=model.labels_.tolist(), cmap=plt.cm.get_cmap("prism", n_clus))
    plt.colorbar(ticks=range(n_clus))
    plt.clim(-0.5, (n_clus - 0.5))
    plt.savefig('i/reduce_and_plot_clusters' +
                str(datetime.datetime.now()) + '.png')
    plt.show()


# plot: Draw a fancy dendrogram
# https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')

    plt.show()

    return ddata


def plot_cluster_counts(model, title=""):
    label_counts = {}
    for label in model.labels_:
        try:
            label_counts[label] += 1
        except:
            label_counts[label] = 0

    print label_counts
    print type(label_counts.keys()), " ", type(label_counts.values())

    plt.bar(label_counts.keys(), label_counts.values())
    plt.title(title)
    plt.xlabel('Cluster')
    plt.ylabel('Number of stories')
    plt.savefig('i/plot_cluster_counts' +
                str(datetime.now()) + '.png')
    plt.show()

# evaluate: evaluate performance of clustering data


def evaluate(fn, ids, model):
    df_label = get_df_label(ids, model)
    df_eval = get_df_eval(fn)
    event_cluster = match_event_cluster(df_label, df_eval)
    return score_event_cluster(df_label, df_eval, event_cluster)


def get_df_label(ids, model):
    clusters = model.labels_
    decisions = ['YES' for x in range(0, len(clusters))]
    df_label = pd.DataFrame(
        {'story': ids, 'cluster': clusters, 'decision': decisions})
    df_label['Score'] = 0
    df_label['Nt'] = 0
    return df_label


def get_df_eval(fn):
    eval_list = []
    for f in fn.split(','):
        print f
        eval_list.append(pd.read_csv("../datasets/eval/" + str(f) +
                                     ".txt", sep='\t', names=['story', 'event', 'decision']))

    df_eval = pd.concat(eval_list)
    return df_eval


def match_event_cluster(df_label, df_eval):
    match = []

    for name, group in df_label.groupby('cluster'):
        max_list = []
        for name_, group_ in df_eval.groupby('event'):
            max_list.append(
                (name_, pd.merge(group, group_, on='story', how='inner').shape[0]))
        max_event = max(max_list, key=operator.itemgetter(1))
        [max_event_id, max_event_freq] = max_event
        match.append((name, max_event_id, max_event_freq))

    event_cluster = {event: [0, 0] for event in list(df_eval.event.unique())}

    for m in match:
        if event_cluster[m[1]][1] < m[2]:
            # update freq, event
            event_cluster[m[1]][1] = m[2]
            event_cluster[m[1]][0] = m[0]

    return event_cluster


def score_event_cluster(df_label, df_eval, event_cluster):
    df_results = pd.DataFrame(
        columns=["a", "b", "c", "d", "miss", "f", "r", "p", "f1"])

    for i, event in enumerate(event_cluster):
        max_cluster = event_cluster[event][0]
        max_event = event

        print "Cluster: ", max_cluster, " Event: ", max_event

        # init floats
        a = 0.0
        b = 0.0
        c = 0.0
        d = 0.0

        # cluster df
        df_cluster = df_label[df_label['cluster'] == max_cluster]
        df_non_cluster = df_label[~df_label.story.isin(df_cluster.story)]

        df_event = df_eval[df_eval['event'] == max_event]
        df_non_event = df_label[~df_label.story.isin(df_event.story)]

        # contingency df
        df_event_cluster = pd.merge(
            df_event, df_cluster, on='story', how='inner')
        df_non_event_cluster = pd.merge(
            df_non_event, df_cluster, on='story', how='inner')
        df_event_non_cluster = pd.merge(
            df_event, df_non_cluster, on='story', how='inner')
        df_non_event_non_cluster = pd.merge(
            df_non_event, df_non_cluster, on='story', how='inner')

        # counts
        a += df_event_cluster.shape[0]
        b += df_non_event_cluster.shape[0]
        c += df_event_non_cluster.shape[0]
        d += df_non_event_non_cluster.shape[0]

        # set rates (otherwise undefined)
        try:
            miss = c / (a + c)
        except:
            miss = 0.0
        try:
            f = b / (b + d)
        except:
            f = 0.0
        try:
            r = a / (a + c)
        except:
            r = 0.0
        try:
            p = a / (a + b)
        except:
            p = 0.0
        try:
            f1 = 2 * r * p / (r + p)
        except:
            f1 = 0.0

        df_results.loc[i] = [a, b, c, d, miss, f, r, p, f1]

    macro_results = df_results[['miss', 'f', 'r', 'p', 'f1']].mean()

    print "[EDEN I/O -- evaluate] macro results: "

    print "- miss: ", macro_results['miss']
    print "- f (false alarm): ", macro_results['f']
    print "- r (recall): ", macro_results['r']
    print "- p (precision): ", macro_results['p']
    print "- f1: ", macro_results['f1']

    # micro rates

    [a_, b_, c_, d_] = df_results[['a', 'b', 'c', 'd']].sum()

    miss_ = c_ / (a_ + c_)
    f_ = b_ / (b_ + d_)
    r_ = a_ / (a_ + c_)
    p_ = a_ / (a_ + b_)
    f1_ = 2 * r_ * p_ / (r_ + p_)

    print "[EDEN I/O -- evaluate] micro results: "

    print "- miss: ", miss_
    print "- f (false alarm): ", f_
    print "- r (recall): ", r_
    print "- p (precision): ", p_
    print "- f1: ", f1_

    micro_results = pd.DataFrame(
        columns=["miss", "f", "r", "p", "f1"])
    micro_results.loc[0] = [miss_, f_, r_, p_, f1_]

    return [df_results, macro_results, micro_results.transpose()]
