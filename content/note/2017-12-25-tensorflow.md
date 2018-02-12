---
title: "Tensorflow"
date: 2017-12-25T14:46:16+08:00
slug: tensorflow
---

Notes about using Tensorflow

## Quick notes

trainable: If `True`, the default, also adds the variable to the graph collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as the default list of variables to use by the `Optimizer` classes.

**Tensorflow gpu auto growth**

``` py
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
```

**Reset default graph**

`tf.reset_default_graph()`

**Tensorboard**

``` py
with tf.Session(config=config) as sess:

    # ...

    writer = tf.summary.FileWriter('./graphs/test', sess.graph)
    writer.flush()
    writer.close()
```

Startup tensorboard:

`tensorboasrd --logdir="<path>" --port 6006 --reload_interval=5`

<!--more-->
---

### Reuse `name_scope`

``` py
with tf.name_scope('variable') as var_scope:
    # ...
# ...
with tf.name_scope(var_scope):
    # ...

```

### Notes of Kernel extractor

- Add a new axis, [10] => [10, 1], `tf.expand_dims(a, 1)`
- Filter with boolean, `tf.boolean_mask(a, mask)`
- `tf.matrix_band_part(input, num_lower, num_upper)` WTF
    - `(0, -1)` Upper triangular part
    - `(-1, 0)` Lower triangular part
    - `(0, 0)` Diagonal
- [Matrix Math Functions](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/matrix_math_functions)
- Assign zeors to diagonal, `tf.matrix_set_diag(a, tf.zeros_like(tf.diag_part(a)))`
- Bool Variable, `tf.Variable(tf.cast(tf.ones([2, 2]), tf.bool))`
- Get variables under `name_scope`, `tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='<scope_name>')`

---

### Tensorflow Assignment

一个重要的观念或者是习惯上的转变是

1. Variable
1. Operation
1. 用操作去修改变量

Operation 一旦定义好了，是可以反复使用的，像optimizer。这其中的关键是，operation包含了把结果存储到变量的操作。

另外一个观念上的转变是，Graph和Session的分离

1. 构建Graph，包括Variable和Operation
1. 用Session去运行Operation
1. 不可以在run的过程中构建新的Operation，会出错, device error?

#### Usage of `tf.scatter_update`

``` py
tf.scatter_update(selection, index, False)
```

#### Usage of `tf.scatter_nd_update`

Change the value of `x[1,:]` and `x[:,1]` simultaneously.

``` py
tf.reset_default_graph()
with tf.Session() as sess:
    n = 3
    k = 1
    x = tf.Variable(tf.zeros([n, n]))
    indices = zip([k]*n + range(n), range(n) + [k]*n) # [[1,0], [1,1], ...]
    updates = -1 * tf.ones([len(indices)])
    update_op = tf.scatter_nd_update(x, indices, updates)

    sess.run(tf.global_variable_initializer())
    sess.run(update_op)
    print sess.run(x)
'''
[[0, 0, 0],
 [0, 0, 0],
 [0, 0, 0]]
=>
[[0, -1, 0],
 [-1, -1, -1],
 [0, -1, 0]]
'''
```

### Tensorboard

``` py
tf.summary.scalar('accuracy', accuracy) # accuracy is a Tensor
summary_op = tf.summary.merge_all()
summary = sess.run(summary_op)
writer = tf.summary.FileWriter('/graphs/sub_path', graph=sess.graph)
writer.add_summary(summary, epoch) # epoch is an integer
```

The best practice is use different `sub_path` name for different training. So we can compare one spercific metric of each training in one grapph.

For example, using:

``` py
from datetime import datetime
def get_log_path(model_name):
    t = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return './graphs/' + model_name + '-' + t
```


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


