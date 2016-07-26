# how to run
# python eden.py ReadData --local-scheduler
# with all files
# python eden.py Evaluate --local-scheduler --fn '1,12,2,20,21,23,28,29,30,33,35,40,5,6'

import luigi

# edenutil import

import edenutil

# io imports

import json
import cPickle as pickle
from sklearn.externals import joblib

io_dir = "io/"

# list luigi parameter

# class Parameter(luigi.Parameter):
#     def parse(self, arguments):
#         return arguments.split(' ')


class ReadData(luigi.Task):
    fn = luigi.Parameter(default="35")

    def requires(self):
        return []

    def output(self):
        return luigi.LocalTarget(io_dir + "read_data/read_data_fn_{}.json".format(self.fn))

    def run(self):
        with self.output().open('w') as f:
            f.write(edenutil.read_data(self.fn))


class PreprocessData(luigi.Task):
    fn = luigi.Parameter(default='35')

    def requires(self):
        return [ReadData(fn=self.fn)]

    def output(self):
        return luigi.LocalTarget(io_dir + "preprocess_data/preprocess_data_fn_{}.p".format(self.fn))

    def run(self):
        with self.input()[0].open() as fin, self.output().open('wb') as fout:
            data = edenutil.preprocess_data(json.load(fin))
            pickle.dump(data, fout, pickle.HIGHEST_PROTOCOL)

class ClusterData(luigi.Task):
    fn = luigi.Parameter(default='35')
    algo = luigi.Parameter(default='kmeans')

    def requires(self):
        return [PreprocessData(fn=self.fn)]

    def output(self):
        return luigi.LocalTarget(io_dir + "cluster_data/cluster_data_fn_{}_algo_{}.p".format(self.fn, self.algo))

    def run(self):
        with self.input()[0].open() as fin, self.output().open('wb') as fout:
            model = edenutil.cluster_data(pickle.load(fin), self.algo)
            pickle.dump(model, fout, pickle.HIGHEST_PROTOCOL)

class Evaluate(luigi.Task):
    fn = luigi.Parameter(default='35')
    algo = luigi.Parameter(default='kmeans')

    def requires(self):
        return [PreprocessData(fn=self.fn), ClusterData(fn=self.fn, algo=self.algo)]

    def output(self):
        return luigi.LocalTarget(io_dir + "evaluate/evaluate_fn_{}_algo_{}.txt".format(self.fn, self.algo))
        return luigi.LocalTarget(io_dir + "evaluate/evaluate_fn_{}_algo_{}.txt".format(self.fn, self.algo))

    def run(self):
        # debug self input type
        # print "SELF INPUT TYPE" type(self.input()[0])
        with self.input()[0].open() as fin_0, self.input()[1].open() as fin_1, self.output().open('w') as fout:
            ids = pickle.load(fin_0)['ids']
            model = pickle.load(fin_1)
            results = edenutil.evaluate(self.fn, ids, model)
            fout.write("Cluster results:\n")
            fout.write(results[0].to_string()+"\n")
            fout.write("Macro results:\n")
            fout.write(results[1].to_string()+"\n")
            fout.write("Micro results:\n")
            fout.write(results[2].to_string()+"\n")

if __name__ == '__main__':
    luigi.run()
