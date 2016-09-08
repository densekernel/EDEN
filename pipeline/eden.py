import luigi
import json
import cPickle as pickle
import edenutil

io_dir = "io/"


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
    method = luigi.Parameter(default='ltc')

    def requires(self):
        return [ReadData(fn=self.fn)]

    def output(self):
        return luigi.LocalTarget(io_dir + "preprocess_data/preprocess_data_fn_{}_method_{}.p".format(self.fn, self.method))

    def run(self):
        with self.input()[0].open() as fin, self.output().open('wb') as fout:
            data = edenutil.preprocess_data(json.load(fin), self.method)
            pickle.dump(data, fout, pickle.HIGHEST_PROTOCOL)


class ClusterData(luigi.Task):
    fn = luigi.Parameter(default='35')
    method = luigi.Parameter(default='ltc')
    algo = luigi.Parameter(default='kmeans')
    params = luigi.Parameter(default='{}')

    def requires(self):
        return [PreprocessData(fn=self.fn, method=self.method)]

    def output(self):
        return luigi.LocalTarget(io_dir + "cluster_data/cluster_data_fn_{}_method_{}_algo_{}_params_{}.p".format(self.fn, self.method, self.algo, self.params))

    def run(self):
        with self.input()[0].open() as fin, self.output().open('wb') as fout:
            model = edenutil.cluster_data(
                pickle.load(fin), self.algo, self.params)
            pickle.dump(model, fout, pickle.HIGHEST_PROTOCOL)


class Evaluate(luigi.Task):
    fn = luigi.Parameter(default='35')
    method = luigi.Parameter(default='ltc')
    algo = luigi.Parameter(default='kmeans')
    params = luigi.Parameter(default='{}')

    def requires(self):
        return [PreprocessData(fn=self.fn, method=self.method), ClusterData(fn=self.fn, method=self.method, algo=self.algo, params=self.params)]

    def output(self):
        return luigi.LocalTarget(io_dir + "evaluate/evaluate_fn_{}_method_{}_algo_{}_params_{}.txt".format(self.fn, self.method, self.algo, self.params))

    def run(self):
        with self.input()[0].open() as fin_0, self.input()[1].open() as fin_1, self.output().open('w') as fout:
            ids = pickle.load(fin_0)['id']
            model = pickle.load(fin_1)
            results = edenutil.evaluate(self.fn, ids, model)
            fout.write("Cluster results:\n")
            fout.write(results[0].to_string() + "\n")
            fout.write("Macro results:\n")
            fout.write(results[1].to_string() + "\n")
            fout.write("Micro results:\n")
            fout.write(results[2].to_string() + "\n")


class CrossValidate(luigi.Task):
    fn = luigi.Parameter(default='35,30')
    method = luigi.Parameter(default='ltc')
    algo = luigi.Parameter(default='kmeans')
    params = luigi.Parameter(default='{}')
    train = luigi.Parameter(default='35')
    test = luigi.Parameter(default='30')

    def requires(self):
        return [PreprocessData(fn=self.train, method=self.method), PreprocessData(fn=self.test, method=self.method)]

    def output(self):
        return luigi.LocalTarget(io_dir + "crossvalidate/cross_validate_fn_{}_method_{}_algo_{}_params_{}_train_{}_test_{}.txt".format(self.fn, self.method, self.algo, self.params[:30], self.train, self.test))

    def run(self):
        with self.input()[0].open() as fin1, self.input()[1].open() as fin2, self.output().open('w') as fout:
            df_train = pickle.load(fin1)
            df_test = pickle.load(fin2)
            fout.write('TRAINING RESULTS\n')
            [max_micro_params, max_macro_params, output] = edenutil.cross_validate(
                self.train, df_train, self.algo, self.params, self.train)

            for o in output:
                fout.write(o)

            model_macro = edenutil.cluster_data(
                df_test, self.algo, params=max_micro_params)
            model_micro = edenutil.cluster_data(
                df_test, self.algo, params=max_macro_params)
            results_macro = edenutil.evaluate(
                self.test, df_test['id'], model_macro)
            results_micro = edenutil.evaluate(
                self.test, df_test['id'], model_micro)

            fout.write('TEST RESULTS\n')
            fout.write('MICRO\n')
            fout.write("Cluster results:\n")
            fout.write(results_macro[0].to_string() + "\n")
            fout.write("Macro results:\n")
            fout.write(results_macro[1].to_string() + "\n")
            fout.write("Micro results:\n")
            fout.write(results_macro[2].to_string() + "\n")
            fout.write('MACRO\n')
            fout.write("Cluster results:\n")
            fout.write(results_micro[0].to_string() + "\n")
            fout.write("Macro results:\n")
            fout.write(results_micro[1].to_string() + "\n")
            fout.write("Micro results:\n")
            fout.write(results_micro[2].to_string() + "\n")


class AnomalyDetection(luigi.Task):
    fn = luigi.Parameter(default='35,30')
    algo = luigi.Parameter(default='kmeans')
    method = luigi.Parameter(default='ltc')
    params = luigi.Parameter(default='{}')
    threshold = luigi.Parameter(default='{"tau": "24h", "t": 0, "k": 1.5}')

    def requires(self):
        return [PreprocessData(fn=self.fn, method=self.method), ClusterData(fn=self.fn, method=self.method, algo=self.algo, params=self.params)]

    def output(self):
        return luigi.LocalTarget(io_dir + "anomaly_detection/anomaly_detection_fn{}_method_{}_algo_{}_params_{}_threshold_{}.txt".format(self.fn, self.method, self.algo, self.params[:30], self.threshold))

    def run(self):
        with self.input()[0].open() as fin_0, self.input()[1].open() as fin_1, self.output().open('w') as fout:
            df_story = pickle.load(fin_0)
            model = pickle.load(fin_1)

            [p, r, f1, outliers, gold_clus] = edenutil.anomaly_detection(
                self.fn, df_story, model, self.threshold)

            fout.write("p:" + str(p) + "\n")
            fout.write("r:" + str(r) + "\n")
            fout.write("f1:" + str(f1) + "\n")
            fout.write("outliers:" + str(outliers) + "\n")
            fout.write("gold_clus:" + str(gold_clus) + "\n")

if __name__ == '__main__':
    luigi.run()
