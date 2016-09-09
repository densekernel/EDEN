#!/bin/bash

## Test the various scripts 

python pipeline/eden.py ReadData --local-scheduler --fn '35,30'

python pipeline/eden.py PreprocessData --local-scheduler --fn '35,30' --method 'ltc'

python pipeline/eden.py ClusterData --local-scheduler --fn '35,30' --method 'ltc' --algo 'kmeans' --params '{"n_clusters": 8}'
python pipeline/eden.py ClusterData --local-scheduler --fn '35,30' --method 'ltc' --algo 'gac' --params '{"b": 10, "s":0.9, "p":0.9}'
python pipeline/eden.py ClusterData --local-scheduler --fn '35,30' --method 'ltc' --algo 'gactemporal' --params '{"b": 10, "s":0.9, "p":0.9, "re": 5}'

python pipeline/eden.py Evaluate --local-scheduler --fn '35,30' --method 'ltc' --algo 'gactemporal' --params '{"b": 10, "s":0.9, "p":0.9, "re": 5}'

python pipeline/eden.py AnomalyDetection --local-scheduler --fn '35,30' --method 'ltc' --algo 'gactemporal' --params '{"b":10, "s": 0.9, "p":0.9, "re":5}'

python pipeline/eden.py CrossValidate --local-scheduler --fn '35,30' --method 'ltc' --algo 'kmeans' --params '{"n_clusters": [5,10,15,20]}' --train '35' --test '30'