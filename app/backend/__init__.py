# Flask imports
from flask import Flask
from flask_restful import reqparse, Resource, Api
from flask.ext.cors import CORS
# General imports
import requests
import json # (probably not necessary, this could be for parsing the data?)
# Local imports (config, ...)
from . import config

# app instance
app = Flask(__name__)
CORS(app)
api = Api(app)

parser = reqparse.RequestParser()

# Hello world route
@app.route("/")
def helloWorld():
  return "Hello, EDEN!"

# Article Count Resource
class ArticleCount(Resource):

  def get(self):
    print("Call for : GET /articles/count")
    url = config.es_base_url['articles'] + '/_count'
    resp = requests.post(url)
    data = resp.json()
    return data

class ArticleCluster(Resource):

  def get(self):
    print("Call for : GET /articles/cluster")
    url = config.es_base_url['articles']+'/_search'
    query = {
      "query" : {
          "match_all" : {},
      },
      "size" : 10
    }
    resp = requests.post(url, data=json.dumps(query))
    data = resp.json()

    # documents array
    articles = data['hits']['hits']
    
    # entity extraction
    print "Entity extraction"
    for i, article in enumerate(articles):
      try:
        entitites = article['_source']['signal-entities']
        article['_source']['signal-entities-text'] = " ".join([entity['surface-form'] for entity in entitites])
      except Exception:
        print "Exception extracting entities at article: ", i
        print Exception
        article['_source']['signal-entities'] = []
        article['_source']['signal-entities-text'] = ""

    # entity_vectorizer = Tfidf

    return articles


# API resource routes
api.add_resource(ArticleCount, config.api_base_url+'/articles/count')
api.add_resource(ArticleCluster, config.api_base_url+'/articles/cluster')