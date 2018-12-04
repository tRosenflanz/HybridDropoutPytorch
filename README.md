# HybridDropoutPytorch

This is implementation of [Hybrid Dropout](https://arxiv.org/abs/1801.07316) for Pytorch with some benchmarking. 

* hybrid_dropout.py - contains implementations of normal and spatial hybrid dropouts 
* train_model.py - trains a model using one of the dropout types and saves metric progression to /data/
* Project_exploration.ipynb  - a notebook containing some sample results. Also contains initial attempts to use Hybrid Dropout to estimate model uncertainty.
* requirements.txt - if you do pip install -r requirements.txt it should install the Python packages required to run code in this project. 

