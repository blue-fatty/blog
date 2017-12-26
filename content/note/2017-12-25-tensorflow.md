---
title: "Tensorflow"
date: 2017-12-25T14:46:16+08:00
---

Notes about using Tensorflow

## Quick notes

- trainable: If `True`, the default, also adds the variable to the graph collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as the default list of variables to use by the `Optimizer` classes.

<!--more-->

## CS 20SI: Tensorflow for Deep Learning Research

### 1. Overview of Tensorflow

- Why TensorFlow ?
    - Python API
    - Portability: deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device with a single API
    - Flexibility: from Raspberry Pi, Android, Windows, iOS, Linux to server farms
    - Visualization (TensorBoard is da bomb)
    - Checkpoints (for managing experiments)
    - Auto-differentiation autodiff (no more taking derivatives by hand. Yay)
    - Large community (> 10,000 commits and > 3000 TF-related repos in 1 year)
    - Awesome projects already using TensorFlow
- Simplified TensorFlow ?
    - TF Learn (tf.contrib.learn): simplified interface that helps users transition from the the world of one-liner such as scikit-learn
    - TF Slim (tf.contrib.slim): lightweight library for defining, training and evaluating complex models in TensorFlow.
    - High level API: Keras, TFLearn, Pretty Tensor
- Tensor
    - An n-dimensional array
- Session
    - A Session object encapsulates the environment in which Operation objects are executed, and Tensor objects are evaluated.
- No more than one graph, use subgraph
    - Multiple graphs require multiple sessions, each will try to use all available resources by default
    - Can't pass data between them without passing them through python/numpy, which doesn't work in distributed
    - It’s better to have disconnected subgraphs within one graph
- Why graphs
    - **Save computation** (only run subgraphs that lead to the values you want to fetch)
    - Break computation into small, differential pieces to **facilitates auto-differentiation**
    - **Facilitate distributed computation**, spread the work across multiple CPUs, GPUs, or devices
    - Many common machine learning models are commonly taught and visualized as directed graphs already

### 2. TensorFlow Ops

**Tensorboard**

``` py
import tensorflow as tf

a = tf.constant(2, name="a")
b = tf.constant(3, name="b")
x = tf.add(a, b, name="add")

with tf.Session() as sess:

    writer = tf.summary.FileWriter("./graphs", sess.graph)
    print sess.run(x)
```

In shell:

``` sh
tensorboard --logdir="./graphs" --port 6006
```

Then open your browser and go to `http://localhost:6006/`


