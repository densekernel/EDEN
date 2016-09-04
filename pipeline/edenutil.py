'''Imports'''

# essentials

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import collections
import ast

# read_data imports

import json
import operator
import itertools

# preprocess_data imports

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
import cPickle as pickle
from datetime import datetime
import time
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer

# cluster_data imports

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import Birch
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics.pairwise import cosine_similarity

# evaluate imports

# timing imports

import time

# graphical imports

from sklearn.manifold import TSNE


'''Library functions'''

# read_data: stitch together data from various feeds


def read_data(fn):
    data = []
    print "[EDEN I/O -- read_data] Files: "
    for f in fn.split(','):
        print "f: ", f
        with open("../datasets/raw-data/" + str(f) + ".json", 'rb') as infile:
            data.extend(json.load(infile))

    print "[EDEN I/O -- read_data] Data length: ", len(data)
    print "[EDEN I/O -- read_data] Data type: ", type(data)

    return json.dumps(data)

# preprocess_data: create vector space model of content


def preprocess_data(data, method):
    # stories = [doc["_source"]["content"] for doc in data]
    # ids = [doc["_id"] for doc in data]

    print "[EDEN I/O -- preprocess_data] Preprocessing data"

    def format_entities(norm_ent):
        ents = []
        for ent in norm_ent:
            try:
                ents.append(porter.stem(ent['surface-form'].lower()))
            except:
                continue
        return " ".join(ents)

    stop_words = set(stopwords.words('english'))
    porter = PorterStemmer()

    def nlp_prepro(doc, porter, stop_words):
        return " ".join([porter.stem(i.lower()) for i in wordpunct_tokenize(doc) if i.lower() not in stop_words])

    d = [{"id": doc["_id"],
          "first-published": doc["_source"]["first-published"],
          "title": doc["_source"]["title"],
          "summary": doc["_source"]["title"],
          "content": doc["_source"]["content"],
          "entities": format_entities(doc["_source"]["normalised-entities"]),
          "content_prepro": nlp_prepro(doc["_source"]["content"], porter, stop_words)} for doc in data]

    df_story = pd.DataFrame(d)
    df_story['first-published-epoch'] = df_story['first-published'].apply(
        lambda x: int(datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ").strftime("%s")))
    df_story = df_story.sort_values(by='first-published-epoch')
    df_story = df_story.reset_index()

    if method == 'ltc':
        vect = TfidfVectorizer(use_idf=True, norm='l2', sublinear_tf=True)
        vsm = vect.fit_transform(df_story['content_prepro'].values)
        vsm_arr = vsm.toarray()
        print "[EDEN I/O -- preprocess_data] VSM shape: ", vsm.shape
        print "[EDEN I/O -- preprocess_data] VSM type: ", type(vsm)
        df_story['vsm'] = [r for r in vsm_arr]
    elif method == 'ltc_ent':
        vect = TfidfVectorizer(use_idf=True, norm='l2', sublinear_tf=True)
        vsm = vect.fit_transform(df_story['entities'].values)
        vsm_arr = vsm.toarray()
        print "[EDEN I/O -- preprocess_data] VSM shape: ", vsm.shape
        print "[EDEN I/O -- preprocess_data] VSM type: ", type(vsm)
        df_story['vsm'] = [r for r in vsm_arr]
    elif method == 'word2vec':
        with open('../datasets/word2vec_signal/word2vec_signal.p', 'rb') as fin:
            word2vec_signal = pickle.load(fin)

        vecs = [word2vec_signal[id_] for id_ in df_story['id']]
        df_story['vsm'] = vecs
    elif method == 'LatentDirichlet':
        vect = CountVectorizer(max_df=0.90, min_df=2).fit_transform(
            df_story['content_prepro'].values)
        lda = LatentDirichletAllocation(n_topics=10, max_iter=5,
                                        learning_method='online', learning_offset=50.,
                                        random_state=0)
        vsm_arr = lda.fit_transform(vect, None)
        # print "[EDEN I/O -- preprocess_data] VSM shape: ", vsm.shape
        # print "[EDEN I/O -- preprocess_data] VSM type: ", type(vsm)
        df_story['vsm'] = [r for r in vsm_arr]

    return df_story

# cluster_data: cluster stories based on similarity


# reconsturct vsm: reconstruct np.ndarray from pd.Series
def recon_vsm(vsm_series):
    # print vsm_series.shape
    rows = vsm_series.shape[0]
    cols = vsm_series.iloc[0].shape[0]
    # print "[EDEN I/O -- recon_vsm] (rows,cols): ", rows, cols
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
    # print "[EDEN I/O -- get_linkage_matrix] cophenet: ", c
    return Z


def get_cluster_size(vsm, Z, p, s):
    # Z debug
    # print "Z"
    # print Z
    n = vsm.shape[0]
    # print "eval cluster"
    n_clus = 1.0
    max_d = s
    for i, z in enumerate(Z):
        # print i, z
        if (1 - z[2]) < s:
            # print i, ": Min similarity reached"
            # print Z[i]
            max_d = Z[i - 1][2]
            n_clus = n - i
            break
        if n * p == n - (i + 1):
            n_clus = n * p
            max_d = Z[i][2]
        if n * p > n - (i + 1):
            # print i, ": Max reduction reached"
            # print Z[i]
            max_d = Z[i - 1][2]
            n_clus = n - i
            break

    # print "[EDEN I/O get_cluster_size] (n_clus, max_d): ", n_clus, max_d

    return [n_clus, max_d]


def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el

# creation dictionary and lists of ordering of clustering
# from linkage matrix
# anatomy of z: z[0] - first doc, z[1] - second doc
# returns dict: key - doc, value - order


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

# build centroid


def build_centroids(group, cluster_order_dict):
    idx = group.index.values
    # print "idx", idx
    # print "cluster_order_dict", cluster_order_dict

    # print "name: ", group.name, "group len: ", len(group), "IDX: ", idx
    # if not singleton (incrementally average vectors and timestamps)
    if len(group) == 1:
        return group
    else:
        # get idx, order and associated (id, time, vsm vectors)
        order = [cluster_order_dict[i] for i in idx]
        idx_order = sorted(zip(idx, order), key=lambda x: x[1])
        idx_order = [list(el) for el in idx_order]
        id_vector = [group.loc[i[0]]['id'] for i in idx_order]
        time_vector = [group.loc[i[0]]['first-published-epoch']
                       for i in idx_order]
        vsm_vector = [group.loc[i[0]]['vsm'] for i in idx_order]

        # debug
        # print "BEFORE PROCESSING"
        # print "idx_order: ", idx_order
        # print "id_vector: ", id_vector
        # print "time_vector: ", time_vector
        # print "vsm_vector: ", vsm_vector

        # pair immediate pairs i, j, where j = i+1
        # and update vectors
        for i, el in enumerate(idx_order):
            if i + 1 < len(idx_order):
                j = i + 1
                if idx_order[i][1] == idx_order[j][1] - 1:
                    #                     print "pair found: ", idx_order[i][0], idx_order[j][0]
                    # update idx_order
                    idx_order[i][0] = [idx_order[i][0], idx_order[j][0]]
                    idx_order[i][1] = np.mean(
                        [idx_order[i][1], idx_order[j][1]])
                    idx_order.pop(j)
                    # update id_vector
#                     id_vector[i] = [id_vector[i], id_vector[j]]
#                     id_vector.pop(j)
                    # update time_vector
                    time_vector[i] = np.mean([time_vector[i], time_vector[j]])
                    time_vector.pop(j)
                    # update vsm_vector
                    vsm_vector[i] = np.mean(
                        np.array([vsm_vector[i], vsm_vector[j]]), axis=0)
                    vsm_vector.pop(j)

        i = 0
        j = 1
        # Merge remaining clusters in order
        # and update vectors
        while len(idx_order) > 1:
            #             print "merge pair: ", idx_order[i][0], idx_order[j][0]
            # update idx_order
            idx_order[i][0] = [idx_order[i][0], idx_order[j][0]]
            idx_order[i][1] = np.mean([idx_order[i][1], idx_order[j][1]])
            idx_order.pop(j)
            # update id_vector
#             id_vector[i] = [id_vector[i], id_vector[j]]
            # update time_vector
            time_vector[i] = np.mean([time_vector[i], time_vector[j]])
            time_vector.pop(j)

            vsm_vector[i] = np.mean(
                np.array([vsm_vector[i], vsm_vector[j]]), axis=0)
            vsm_vector.pop(j)

        # perform remaining averageing

        # print "AFTER PROCESSING"
        # print "id_vector: ", id_vector
        # print "idx_order: ", idx_order
        # print "time_vector: ", time_vector
        # print "vsm_vector: ", vsm_vector

        group_columns = group.columns
        df_group = pd.DataFrame(columns=group_columns)

        group_data = []
        for col in group_columns:
            if col == 'id':
                # id_flat = [item for sublist in id_vector for item in sublist]
                group_data.append(id_vector)
            elif col == 'first-published-epoch':
                group_data.append(time_vector[0])
            elif col == 'vsm':
                group_data.append(vsm_vector[0])
            elif col == 'cluster':
                group_data.append(group.name)
            else:
                group_data.append(list(group[col]))

        df_group.loc[0] = group_data
        return df_group


def basic_GAC(df_story, p, s):
    # experiment with incremental reset of index on each new sub story frame
    df_story = df_story.reset_index(drop=True)
    # print "BASIC GAC"
    vsm = recon_vsm(vsm_series=df_story['vsm'])
    Z = get_linkage_matrix(vsm)
    [n_clus, max_d] = get_cluster_size(vsm, Z, p=p, s=s)
    clusters = fcluster(Z, n_clus, criterion='maxclust')
    df_story['cluster'] = clusters
    [cluster_order_dict, cluster_order_list] = get_cluster_order(Z)
    # print "cluster_order_list", cluster_order_list
    # print len(df_story)
    df_update = df_story.groupby('cluster').apply(
        lambda group: build_centroids(group, cluster_order_dict)).reset_index(drop=True)
    # print len(df_update)
    # optional : fancy dendrogram
    # ddata = fancy_dendrogram(
    #     Z,
    #     truncate_mode='lastp',
    #     p=58,
    #     leaf_rotation=90.,
    #     leaf_font_size=12.,
    #     show_contracted=True,
    #     annotate_above=10,
    #     max_d=max_d  # useful in small plots so annotations don't overlap
    # )

    return df_update


class GAC:

    def __init__(self, b=10.0, p=0.5, s=0.8, t=100, re=5):
        self.labels_ = []
        self.b = b
        self.p = p
        self.s = s
        self.t = t
        self.re = re

    # cluster
    def fit(self, df_story):

        df_update = basic_GAC(df_story, p=self.p, s=self.s)
        clusters = construct_clusters(df_update, df_story)
        # print "[EDEN I/O -- cluster_data] clusters: ", clusters
        print "[EDEN I/O -- gac.fit] len(clusters): ", len(set(clusters))
        print "[EDEN I/O -- gac.fit] clusters: ", clusters

        self.labels_ = clusters


class GACTemporal:

    def __init__(self, b=10.0, p=0.5, s=0.8, t=100, re=5):
        self.labels_ = []
        self.b = b
        self.p = p
        self.s = s
        self.t = t
        self.re = re

    # cluster
    def fit(self, df_story):

        df_update = df_story

        # print len(df_story)
        i = 0
        # split dataframe according to b
        # max value (similarity)
        max_value = 1.0
        while len(df_update) > self.b and max_value > self.s:
            # drop partial labels
            # re-format df and calc max_value (similarity)
            if i > 0:
                df_update = df_update.drop('cluster', 1)
                # print "***** REPEAT RE-BUCKETING ******"
            # rebucket
            if i % self.re == 0 and i > 0:
                # print "no split"
                splits = [df_update]
            # split GAC
            else:
                # print "normal split"
                split_size = len(df_update) / self.b
                splits = np.array_split(df_update, split_size)
            # print "[EDEN I/O -- fit] split_size: ", split_size

            # apply GAC
            df_split_update = []

            for df_split in splits:
                df_split_update.append(
                    basic_GAC(df_story=df_split, p=self.p, s=self.s))

            df_update = pd.concat(df_split_update)

            i += 1

            sep_vsm = recon_vsm(df_update['vsm'])
            cos_sim = cosine_similarity(sep_vsm)
            mask = np.ones(cos_sim.shape, dtype=bool)
            np.fill_diagonal(mask, 0)
            max_value = cos_sim[mask].max()
            # print "max_value", max_value
            # if max_value < self.s:
            # print "min sim overall reached"

        clusters = construct_clusters(df_update, df_story)

        print "[EDEN I/O -- gactemporal.fit] len(clusters): ", len(set(clusters))
        print "[EDEN I/O -- gactemporal.fit] clusters: ", clusters

        self.labels_ = clusters


def construct_clusters(df_update, df_story):
    id_cluster = {}
    for i, row in df_update.iterrows():
        # print type(row['cluster'])
        if isinstance(row['id'], list):
            id_list = flatten(row['id'])
            for id_ in id_list:
                id_cluster[id_] = row['cluster']
        else:
            id_cluster[row['id']] = row['cluster']

    return [id_cluster[id_] for id_ in df_story['id'].values]


def algo_select(algo='kmeans', params={}):
    print "[EDEN I/O -- algo_select] algo: ", algo
    print "[EDEN I/O -- algo_select] params: ", params
    print "[EDEN I/O -- algo_select] type(params): ", type(params)
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
    elif algo == 'birch':
        return Birch(**params)


def cluster_data(df_story, algo='kmeans', params='{}'):
    print "[EDEN I/O -- cluster_data] algo: ", algo
    # here vsm is dense array (test performance)
    # maybe add column for csr_matrix as well

    start = time.time()

    params = ast.literal_eval(params)

    if algo in ['gac', 'gactemporal']:
        min_clus = 4.0
        try:
            b = params["b"]
        except:
            b = max(math.ceil((400.0 / 15836.0) * df_story.shape[0]), min_clus)

        params["b"] = b
        # params = {'b': b,
        #           'p': 0.5,
        #           's': 0.5,
        #           't': 100,
        #           're': 5
        #           }
        model = algo_select(algo, params)
        model.fit(df_story)

    elif algo == 'meanshift':
        vsm = recon_vsm(df_story['vsm'])
        params['bandwidth'] = estimate_bandwidth(vsm, n_samples=200)
        model = algo_select(algo, params)
        model.fit(vsm)
    else:
        vsm = recon_vsm(df_story['vsm'])
        model = algo_select(algo, params)
        model.fit(vsm)

    # print "[EDEN I/O -- cluster_data.py] plot: cluster counts"
    # plot_cluster_counts(model, "Cluster counts using algorithm: " + str(algo))

    # print "[EDEN I/O -- cluster_data] model: ", model

    end = time.time()
    print "[EDEN I/O -- cluster_data.py] Total elapsed time: ", end - start

    return model

# plotting functions for clusters

# plot: Reduce dimensionality of data and plot with cluster colors


def reduce_and_plot_clusters(X, model, title="", just_labelled=False, event_clusters=[]):

    if not type(model.labels_) is list:
        n_clus = len(set(model.labels_.tolist()))
        labels_ = model.labels_.tolist()
    else:
        labels_ = model.labels_
        n_clus = len(set(model.labels_))

    if just_labelled:
        idx = [i for i, label in enumerate(labels_) if label in event_clusters]
        print idx
        X = X[idx, :]
        labels_ = [labels_[i] for i in idx]
        label_dict = dict(zip(set(labels_), range(0, len(set(labels_)))))
        labels_ = [label_dict[label] for label in labels_]
        print labels_
        n_clus = len(event_clusters)
        print n_clus

    X_reduced = TruncatedSVD().fit_transform(X)
    X_embedded = TSNE(learning_rate=100).fit_transform(X_reduced)

    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], marker="x",
                c=labels_, cmap=plt.cm.get_cmap("jet", n_clus))
    plt.colorbar(ticks=min(range(500), range(n_clus)))
    plt.clim(-0.5, (n_clus - 0.5))
    plt.savefig('i/reduce_and_plot_clusters' +
                str(datetime.now()) + '.png')
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
            label_counts[label] = 1

    print label_counts
    # print type(label_counts.keys()), " ", type(label_counts.values())

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
    event_cluster = {}

    # populate matrix
    for i, (name, group) in enumerate(df_eval.groupby('event')):
        # print "i: ", i, "Event: ", name
        m = 0
        for j, (name_, group_) in enumerate(df_label.groupby('cluster')):
            # print "j: ", j, "Cluster: ", name_
            n = pd.merge(group, group_, on='story', how='inner').shape[0]
            if n > m:
                m = n
                event_cluster[name] = (name_, n)

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
        # try:
        #     miss = c / (a + c)
        # except:
        #     miss = 0.0
        #     continue
        # try:
        #     f = b / (b + d)
        # except:
        #     f = 0.0
        #     continue
        # try:
        #     r = a / (a + c)
        # except:
        #     r = 0.0
        #     continue
        # try:
        #     p = a / (a + b)
        # except:
        #     p = 0.0
        #     continue
        # try:
        #     f1 = (2 * r * p) / (r + p)
        # except:
        #     f1 = 0.0
        #     continue

        # try single block
        try:
            miss = c / (a + c)
            f = b / (b + d)
            r = a / (a + c)
            p = a / (a + b)
            f1 = (2 * r * p) / (r + p)
            df_results.loc[i] = [a, b, c, d, miss, f, r, p, f1]
        except:
            continue

        # df_results.loc[i] = [a, b, c, d, miss, f, r, p, f1]

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
    f1_ = (2 * r_ * p_) / (r_ + p_)

    print "[EDEN I/O -- evaluate] micro results: "

    print "- miss: ", miss_
    print "- f (false alarm): ", f_
    print "- r (recall): ", r_
    print "- p (precision): ", p_
    print "- f1: ", f1_

    micro_results = pd.DataFrame(
        columns=["miss", "f", "r", "p", "f1"])
    micro_results.loc[0] = [miss_, f_, r_, p_, f1_]

    return [df_results, macro_results, micro_results.mean()]

# cross_validate


def cross_validate(fn, df_story, algo='kmeans', params='{}', train='35'):
    print "[EDEN I/O -- cross_validate] algo: ", algo
    print "[EDEN I/O -- cross_validate] params: ", params
    # here vsm is dense array (test performance)
    # maybe add column for csr_matrix as well

    params = ast.literal_eval(params)

    params_grid = list(itertools.product(*params.values()))

    results_list = []

    for p_inst in params_grid:
        # print "p_inst", p_inst
        params_inst = {}
        p_list = list(p_inst)
        for i, key in enumerate(params):
            params_inst[key] = p_list[i]
        # print "params_inst", params_inst
        model = cluster_data(df_story, algo, params=str(params_inst))
        res = [str(params_inst)]
        res.extend(evaluate(fn, df_story['id'], model))
        # print "res", res
        results_list.append(res)

    max_macro_F1 = 0.0
    max_micro_F1 = 0.0
    max_harmonic_F1 = 0.0

    output = []

    for i, results in enumerate(results_list):
        output.append("Parameters:\n")
        output.append(results[0] + "\n")
        output.append("Cluster results:\n")
        output.append(results[1].to_string() + "\n")
        output.append("Macro results:\n")
        output.append(results[2].to_string() + "\n")
        output.append("Micro results:\n")
        output.append(results[3].to_string() + "\n")

        if max_macro_F1 < results[2]['f1']:
            max_macro_params = results[0]
            max_macro_F1 = results[2]['f1']
        if max_micro_F1 < results[3]['f1']:
            max_micro_params = results[0]
            max_micro_F1 = results[3]['f1']

    output.append("Max Micro F1:\n")
    output.append(str(max_micro_F1) + '\n')
    output.append(max_micro_params + '\n')
    output.append("Max Macro F1:\n")
    output.append(str(max_macro_F1) + '\n')
    output.append(max_macro_params + '\n')

    # print "[EDEN I/O -- cluster_data.py] plot: cluster counts"
    # plot_cluster_counts(model, "Cluster counts using algorithm: " + str(algo))

    # print "[EDEN I/O -- cluster_data] model: ", model

    return [max_micro_params, max_macro_params, output]

# anomaly_detection


def anomaly_detection(fn, df_story, model, threshold):
    print "[EDEN I/O -- anomaly_detection] threshold: ", threshold
    threshold = ast.literal_eval(threshold)
    # get data and ec pair
    df_update = df_story
    ids_ = df_story['id']
    ec = match_event_cluster(get_df_label(ids_, model), get_df_eval(fn))

    # loop through groups (characterise events)
    df_update['cluster'] = model.labels_
    g = df_update.groupby('cluster')
    ts = {}
    ts_w = {}
    for name, group in g:
        if len(group) <= threshold["t"]:
            continue
        print "[EDEN I/O -- anomaly_detection] name (group): ", name
        group[
            'first-published_dt'] = group['first-published'].apply(pd.to_datetime)
        group.index = group['first-published_dt']
        event_data = group.resample(threshold["tau"], how={
                                    'first-published_dt': 'count'})
        max_spike = event_data['first-published_dt'].max()
        ts[name] = max_spike
        # print "[EDEN I/O -- anomaly_detection] vsm: ", group['vsm']
        cos_sim_mat = cosine_similarity(recon_vsm(group['vsm']))
        cos_sim_avg = np.mean(
            cos_sim_mat[np.triu_indices(cos_sim_mat.shape[0])])
        print "[EDEN I/O -- anomaly_detection] cos_sim_avg: ", cos_sim_avg
        print "****"
        ts_w[name] = max_spike * cos_sim_avg

    # get outliers
    # if threshold[t]

    # test
    ts = ts_w

    ts_no1 = [t for t in ts.values() if t > threshold["t"]]
    iqr = np.subtract(*np.percentile(ts_no1, [75, 25]))
    q75, q25 = np.percentile(ts_no1, [75, 25])

    print "[EDEN I/O -- anomaly_detection] ts_no1: ", ts_no1

    print "[EDEN I/O -- anomaly_detection] iqr, q75, q25: ", iqr, q75, q25, "range:", q25 - threshold["k"] * iqr, q75 + threshold["k"] * iqr
    # outliers = [(k, v) for (k,v) in zip(ts.keys(), ts.values()) if ts[k] > q75+threshold["k"]*iqr or ts[k] < q25-threshold["k"]*iqr]
    # just positive spikes
    outliers = [(k, v) for (k, v) in zip(ts.keys(), ts.values())
                if ts[k] > q75 + threshold["k"] * iqr]

    # evaluation
    print "[EDEN I/O -- anomaly_detection] ts: ", ts
    print "[EDEN I/O -- anomaly_detection] outliers: ", outliers
    gold_clus = [i[0] for i in ec.values()]

    print "[EDEN I/O -- anomaly_detection] gold_clus: ", gold_clus
    rel = float(len([o[0] for o in outliers if o[0] in gold_clus]))
    p = rel / len(outliers)
    r = rel / len(gold_clus)
    print "[EDEN I/O -- anomaly_detection] p, r: ", p, r
    f1 = 2 * ((p * r) / (p + r))

    print "[EDEN I/O -- anomaly_detection] p, r, F1: ", p, r, f1

    return [p, r, f1, outliers, gold_clus]
