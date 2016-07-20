'''Imports'''

# read_data imports

import json
import operator
import itertools

# preprocess_data imports

from sklearn.feature_extraction.text import TfidfVectorizer
import cPickle as pickle

# cluster_data imports

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import datetime

# evaluate imports

import pandas as pd


# fn = [35, 30, 6, 33, 23, 2, 1, 20, 29, 40]
# fn = [35]

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
    stories = [doc["_source"]["content"] for doc in data]
    ids = [doc["_id"] for doc in data]

    vect = TfidfVectorizer(use_idf=True)
    vsm = vect.fit_transform(stories)

    print "[EDEN I/O -- preprocess_data] VSM shape: ", vsm.shape
    print "[EDEN I/O -- preprocess_data] VSM type: ", type(vsm)

    return [ids, vsm]

# cluster_data: cluster stories based on similarity


def algo_select(algo):
    return {
        'kmeans': KMeans(),
    }.get(algo, KMeans())


def cluster_data(vsm, algo='kmeans'):

    model = algo_select(algo)
    model.fit(vsm)

    print "[EDEN I/O -- cluster_data] algo: ", algo

    # print "[EDEN I/O -- cluster_data.py] plot: cluster counts"
    # plot_cluster_counts(model, "Cluster counts using algorithm: " + str(algo))

    return model

# plotting functions for clusters


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
                str(datetime.datetime.now()) + '.png')
    plt.show()


def reduce_and_plot_clusters(X, model, title=""):
    X_reduced = TruncatedSVD().fit_transform(X)
    X_embedded = TSNE(learning_rate=100).fit_transform(X_reduced)

    n_clus = len(set(model.labels_.tolist()))

    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], marker="x",
                c=model.labels_.tolist(), cmap=plt.cm.get_cmap("jet", n_clus))
    plt.colorbar(ticks=range(n_clus))
    plt.clim(-0.5, (n_clus - 0.5))
    plt.savefig('i/reduce_and_plot_clusters' +
                str(datetime.datetime.now()) + '.png')
    plt.show()

# evaluate: evaluate performance of clustering data


def evaluate(fn, ids, model):
    df_label = get_df_label(ids, model)
    df_eval = get_df_eval(fn, model)
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


def get_df_eval(fn, model):
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
            f = b / (b + d)
            r = a / (a + c)
            p = a / (a + b)
            f1 = 2 * r * p / (r + p)
        except:
            a = None
            b = None
            c = None
            d = None
            miss = None
            f = None
            r = None
            f1 = None

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
