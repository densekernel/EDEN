# EDEN

## (Anomalous) Event Detection in News

### Architecture

#### ElasticSearch

News articles are indexed in a (running) instance of ElasticSearch.

#### Back-end: Python Flask

The back-end of the application is powered by Python Flask.

+ Flask_RESTful powers API
+ Flask_CORS (cross-origin resource sharing), mainly for cross-origin AJAX

#### Front-end: Angular UI?

+ Currently under development

### Installation (and operation)

Installation:

+ Clone repo
+ Run ElasticSearch with Signal 1M-Sample

Operation (in seperate terminal tabs):

+ `make venv`
+ `make backend`

+ `make venv`
+ `make frontend`

Access (web application):

+ visit `http://localhost:8000`

### Credits

+ [Building a Search-As-You-Type Feature with Elasticsearch, AngularJS and Flask](https://marcobonzanini.com/2015/08/10/building-a-search-as-you-type-feature-with-elasticsearch-angularjs-and-flask/)
+ [Pt 2](https://marcobonzanini.com/2015/08/18/building-a-search-as-you-type-feature-with-elasticsearch-angularjs-and-flask-part-2-front-end/)

