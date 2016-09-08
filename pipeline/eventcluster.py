import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics.pairwise import cosine_similarity
import collections

class GAC:

    def __init__(self, b=10.0, p=0.5, s=0.8, t=100, re=5):
        self.labels_ = []
        self.b = b
        self.p = p
        self.s = s
        self.t = t
        self.re = re

    def fit(self, df_story):

        df_update = basic_GAC(df_story, p=self.p, s=self.s)
        clusters = construct_clusters(df_update, df_story)
        self.labels_ = clusters

        # print "[EDEN I/O -- cluster_data] clusters: ", clusters
        # print "[EDEN I/O -- gac.fit] len(clusters): ", len(set(clusters))
        # print "[EDEN I/O -- gac.fit] clusters: ", clusters


class GACTemporal:

    def __init__(self, b=10.0, p=0.5, s=0.8, t=100, re=5):
        self.labels_ = []
        self.b = b
        self.p = p
        self.s = s
        self.t = t
        self.re = re

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


def recon_vsm(vsm_series):
    rows = vsm_series.shape[0]
    cols = vsm_series.iloc[0].shape[0]
    vsm = np.zeros((rows, cols))

    for i, r in enumerate(vsm_series):
        vsm[i] = r

    print "[EDEN I/O -- recon_vsm] (rows, cols, shape, type(vsm), type(vsm[0])): ", rows, cols, vsm.shape, type(vsm), type(vsm[0])

    return vsm

# mycosine: cosine similarity function


def mycosine(x1, x2):
    x1 = x1.reshape(1, -1)
    x2 = x2.reshape(1, -1)
    ans = 1 - cosine_similarity(x1, x2)
    return max(ans[0][0], 0)

# get_linkage_matrix: perform hierarchical clustering on VSM


def get_linkage_matrix(vsm):
    Z = linkage(vsm, method='average', metric=mycosine)
    c, coph_dists = cophenet(Z, pdist(vsm))
    # print "[EDEN I/O -- get_linkage_matrix] cophenet: ", c
    return Z

# get_cluster_size: find number of clusters at stopping criterion


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

# get_cluster_order: creation dictionary and lists of ordering of clustering
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

# build_centroids: incrementally average cluster members


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
