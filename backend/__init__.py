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




# API resource routes
api.add_resource(ArticleCount, config.api_base_url+'/articles/count')