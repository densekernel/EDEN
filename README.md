# EDEN

## (Anomalous) Event Detection in News

+ Analyse stream of news articles to find anomalous events
+ NLP modules to represent data
+ Apply Document Clustering algorithms for Event Detection
+ Characterise event-centric clusters and use statistical threshold models for Anomaly Detection

## Installation

To use this with an existing dataset, we recommend setting up the repostiory in the following way.

    app/
    datasets/
      eval/
        1.txt
        2.txt
        ...
      raw-data/
        1.json
        2.json
        ...
      word2vec_signal/
        word2vec_signal.p
    pipeline/
      /io/

Part of this structure will be set up during the process of cloning.

- `git clone https://github.com/jonathanmanfield/EDEN`
- `cd EDEN`
- `cp path/to/datasets .`

## Data Pipeline

Implements Python Luigi.

### Requirements 

TODO list all the librariesx


### Documentation

| Task             | Dependencies                  | Parameters                                                                                   | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|------------------|-------------------------------|----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ReadData         | {}                            | {‘fn’ : list}                                                                                | Reads respective data (news articles) from ‘fn’, a list of filenames. Returns a con- catenation of all files.  'fn':  Use a comma-seperated text string (e.g., '1,2,3').                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| PreprocessData   | {ReadData}                    | {‘fn’ : list, ‘method’: string}                                                              | Apply NLP techniques on unstructured data for stop- word removal, lower-casing and porter stemming. Convert to structure data format of Vector Space Model.  Method: Selected by parameter; ‘ltc’: tf-idf on entire content, ‘ltc-ent’: tf-idf on named entities, ‘word2vec’: Use pre-trained Google News vectors                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ClusterData      | {PreprocessData}              | {‘fn’ ‘method’: string, ‘algo’: string, ‘params’: dict}                                      | Run Document Cluster-ing algorithm (selected by‘algo’ with hyperparameters‘params’) on PreprocessedData'algo': Select from {'kmeans', 'dbscan', 'meanshift', 'birch', 'gac', 'gactemporal'} to run the corresponding algorithm'params': Use string representation of Python Dictionary (e.g., '{"n_clusters": 50}')n.b., for choice of parameters of algorithm see corresponding sklearn documentation, with the exception of 'gac' or 'gactemporal' which accept parameters: 'b=10.0, p=0.5, s=0.8, t=100,', b is factor (stopping criteria), s is minimum similarity threshold (stopping criteria), bucket size, p is a reduction 'gac has a parameter 're=5' for number of iterations to perform normally before re-bucketing. |
| Evaluate         | {PreprocessData, ClusterData} | {‘fn’ : list, ‘method’: string, ‘algo’: string, ‘params’: dict}                              | Evaluate performance of Document Clustering algoirthm using external criterion and labelled data in the style of TDT Pilot Study                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| CrossValidate    | {PreprocessData}              | {‘fn’ : list, ‘method’: string, ‘algo’: string, ‘params’: dict, 'train': list, 'test': list} | Perform grid search across range of hyperaparameters to optimise Document Clus- tering algorithms  train, test: Use a comma-separated value of filenames like 'fn' (e.g., '3,5,6')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| AnomalyDetection | {PreprocessData, ClusterData} | {'tau': string, 't': integer, 'k':float } | Create a statistical threshold model that classifies event-centric clusters as anomalous if the product of their cohesiveness (cosine similarity) and highest burst in publishing is outside a range.|
### Prompts

Each task can be run separately and parameterised. All intermediary files will be saved to `pipeline/io`
Please note that all the source python files are located in 'pipeline', i.e. you need to `cd pipeline`

- ReadData: `python eden.py ReadData --local-scheduler --fn '35,30'`
- PreprocessData: `python eden.py PreprocessData --local-scheduler --fn '35,30' --method 'ltc'`
- ClusterData:
  - `python eden.py ClusterData --local-scheduler --fn '35,30' --method 'ltc' --algo 'kmeans' --params '{"n_clusters": 8}'`
  - `python eden.py ClusterData --local-scheduler --fn '35,30' --method 'ltc' --algo 'gac' --params '{"b": 10, "s":0.9, "p":0.9}'`
  - `python eden.py ClusterData --local-scheduler --fn '35,30' --method 'ltc' --algo 'gactemporal' --params '{"b": 10, "s":0.9, "p":0.9, "re": 5}'`
- Evaluate:
  - `python eden.py Evaluate --local-scheduler --fn '35,30' --method 'ltc' --algo 'gactemporal' --params '{"b": 10, "s":0.9, "p":0.9, "re": 5}'`
- `AnomalyDetection: python eden.py AnomalyDetection --local-scheduler --fn '35,30' --method 'ltc' --algo 'gactemporal' --params '{"b":10, "s": 0.9, "p":0.9, "re":5}'`


Or the entire pipeline can be run at once. Of-course it can be parameterised at this stage too.

- `python eden.py Evaluate --local-scheduler`
- `python eden.py AnomalyDetection --local-scheduler`
- `python eden.py CrossValidation --local-scheduler`

Separately, there exists a cross-validation function to find the best hyperparameters and see how well they generalize to test data.

- `python eden.py CrossValidate --local-scheduler --fn '35,30' --method 'ltc' --algo 'kmeans' --params '{"n_clusters": [5,10,15,20]}' --train '35' --test '30'`


## App (Under Construction)

### Architecture

+ Python 2.7

#### Elasticsearch

News articles are indexed in a (running) instance of ElasticSearch.

+ Elasticsearch should be running (`http://locahost:9200`)
+ Mappings should match those in use by Signal

#### Back-end: Python Flask

The back-end of the application is powered by Python Flask.

+ Flask_RESTful powers API
+ Flask_CORS (cross-origin resource sharing), mainly for cross-origin AJAX

#### Front-end: Angular UI

+ Angular 1.4.3
+ Twitter Bootstrap
+ Routes exist to count the number of articles

### Installation (and operation)

Installation:

+ Run ElasticSearch with Signal 1M-Sample (See Elasticsearch section)
+ `git clone https://github.com/jonathanmanfield/EDEN`
+ `cd EDEN/app`
+ `virtualenv venv`
+ `source ./venv/bin/activate`
+ pip install -r requirements.txt (Needs testing)

Operation (in seperate terminal tabs):

Back-end:

+ `source ./venv/bin/activate`
+ `make backend`

Front-end:

+ `source ./venv/bin/activate`
+ `make frontend`

Access (web application):

+ visit `http://localhost:8000`
+ Test count articles by visiting `http://localhost:8000/#/articles/count`

## Notebooks (Coming soon)

+ Data Visualisation (With dimensionality reduction of article vectors and plot characterisations)

## Credits

+ [Building a Search-As-You-Type Feature with Elasticsearch, AngularJS and Flask](https://marcobonzanini.com/2015/08/10/building-a-search-as-you-type-feature-with-elasticsearch-angularjs-and-flask/)
+ [Building a search-as-you-type feature with Elasticsearch, AngularJS and Flask (Part 2: front-end)](https://marcobonzanini.com/2015/08/18/building-a-search-as-you-type-feature-with-elasticsearch-angularjs-and-flask-part-2-front-end/)
+ [SciPy Hierarchical Clustering and Dendrogram Tutorial](https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/)

