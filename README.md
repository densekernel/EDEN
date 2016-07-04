# EDEN

## (Anomalous) Event Detection in News

+ Open source code to analyse streams of news articles and detect events. Moreover, to detect amalous events, those with a story count or vocabulary representing an outlier based on statistical thresholds.

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
+ `cd EDEN`
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

### Notebooks

+ Signal Notebook (Contains exploratory insights into dataset format and clustering)
+ 

### Credits

+ [Building a Search-As-You-Type Feature with Elasticsearch, AngularJS and Flask](https://marcobonzanini.com/2015/08/10/building-a-search-as-you-type-feature-with-elasticsearch-angularjs-and-flask/)
+ [Building a search-as-you-type feature with Elasticsearch, AngularJS and Flask (Part 2: front-end)](https://marcobonzanini.com/2015/08/18/building-a-search-as-you-type-feature-with-elasticsearch-angularjs-and-flask-part-2-front-end/)

