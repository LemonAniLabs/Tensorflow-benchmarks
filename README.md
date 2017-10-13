# Tensorflow-benchmarks
Tensorflow model inference Benchmarks via slim model zoo

## Introduction
This repository contains machine learning models implemented in TensorFlow. It was developed for easily test network architecture and performance via [slim](https://github.com/tensorflow/models/tree/master/research/slim) model define. The models are maintained by their respective authors. 

## Requirement
- Tensorflow
- opencv

## Installation
1. Clone this repository
    ```shell
    $> git clone https://github.com/LemonAniLabs/Tensorflow-benchmarks.git
    ```
2. Download the pre-train weight from tf-slim
    ```shell
    # Download resnet_v1_50 pre-train weight
    $> wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
    ```
## Run the benchmark
```shell    
    $> python resnet_v1_50_test.py
    # That's it
```

# Optional arguments:
```shell
  -h, --help            show this help message and exit
  -t TIMES, --TIMES TIMES
                        Times of iteration
  -o, --OFFICIAL        Use Slim.Net from tensorflow contrib
```
## Features
- [ ] Training a model from scratch
- [ ] Fine-tuning a model from an existing checkpoint
