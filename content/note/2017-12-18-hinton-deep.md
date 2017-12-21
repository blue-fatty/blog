---
title: "Hinton - Neural Networks for Machine Learning"
date: 2017-12-18T09:32:46+08:00
slug: hinton
tags: ["deep learning"]
---

Learning notes of Hinton's [Neural Networks for Machine Learning](https://www.coursera.org/learn/neural-networks/home/info) in Coursera.

## 1. Introduction

### 1a. Why do we need machine learning

**What, Why**

- We don't know how to program to solve some problems or the program might be very complicated.
- Rules may need to change frequently, like recognizing fraud.
- Cheap computation

**How**

- Collect lots of cases with inputs and correct outputs.
- ML algorithms takes examples and produces a program to do the job.

**Good at**

- Recognizing patterns
  - Objects
  - Face
  - Spoken words
- Recognizing anomalies (unusual)
  - Transactions
  - Sensor readings in a nuclear power plant
- Prediction
  - Stock prices, exchange rates
  - Movie recommendations

<!--more-->

**Examples**

- Handwriting, hard to find a template.
- MNIST
- Imagenet
- Top-k error

**Speech Recognition Task**

- (wave to vector) wave into a vector of acoustic coefficients (声学系数？), 10ms/vector
- (find phoneme) adjacent vectors, which part of which phoneme (音素) is being spoken
- (to sentence) decoding, fintting acoustic data and human habit
- Phone recognition on the TIMIT benchmark[^1]

[^1]: Mohamed, Dahl, & Hinton, 2012


**Word error rates from MSR, IBM, & Google**[^2]

[^2]: Hinton et. al. IEEE Signal Processing Magazine, Nov 2012

| The task                          | Hours of training data   | Deep neural network   | Gaussian Mixture Model   | GMM with more data   |
| --------------------------------- | ------------------------ | --------------------- | :----------------------: | -------------------- |
| Switchboard (MicrosoftResearch)   | 309                      | 18.5%                 | 27.4%                    | 18.6%
| English broadcastnews  (IBM)      | 50                       | 17.5%                 | 18.8%                    |
| Google voice search (android 4.1) | 5,870                    | 12.3% (and falling)   |                          | 16.0% (>>5,870 hrs)

### Quiz notes

- Data change and old relationship not. How can we prevent a neural network from forgetting old data?
    - True
        - Prevent the system form changing the weights too much
        - Train on a mix of old and new data
    - False
        - Ignore
        - Train two networks. (We don't know use which one for test data)
- Theory: local neural circuits in most parts of the cortex all use the same general purpose learning algorithm
    - If part of the cortex is removed early in life, the function that it would have served often gets relocated to another part of cortex.
    - The fine-scale anatomy of the cortex looks pretty much the same all over.
    - If the visual input is sent to the auditory cortex of a newborn ferret, the "auditory" cells learn to do vision.

### 1b. What are neural netowrks

Hinton seems have a biological background.

- neural computation
    - understant brain
    - understand parallel computation inspired by neurons and their adaptive? connections
        - vision, math(\\(23 \times 71\\))
    - solution to practical problemsinspired by brain(neural networks)
- cortical neuron (皮层神经元)
    - axon 轴突
    - dendritic tree 树突树
    - axon hillock 轴突丘
    - spike 单神经元发的信号
- synapses (突触)
    - vesicles 囊泡
    - transmitter chemical released
    - 传送的化学分子扩散到突触间隙，与突触后神经细胞膜上的受体分子结合，来改变它们的形状
        - 打开了特定的进或者出的离子通道
- synapses adapt

## The Perceptron learning procedure

## References
