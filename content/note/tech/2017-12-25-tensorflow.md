---
title: "Tensorflow"
date: 2017-12-25T14:46:16+08:00
slug: tensorflow
weight: 8
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

---

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

---

### Padding mode

[What is the difference between 'SAME' and 'VALID' padding in tf.nn.max_pool of tensorflow?](https://stackoverflow.com/a/39371113)

- Input width = 13
- Filter width = 6
- Stride = 5

`padding='valid'`:

``` py
inputs:         1  2  3  4  5  6  7  8  9  10 11 (12 13)
              |________________|                dropped
                             |_________________|
```

`padding='same'`:

``` py
           pad|                                      |pad
inputs:      0 |1  2  3  4  5  6  7  8  9  10 11 12 13|0  0
           |________________|
                          |_________________|
                                         |________________|
```

