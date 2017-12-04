---
title: Word embedding
tags: [word embedding, NLP]
---

## 拟合目标

条件概率：
`!$P(w_j|w_i)$`

二元词概率：
`!$P(w_i, w_j)$`

互信息：
`!$log \frac{P(w_i,w_j)}{P(w_i)P(w_j)}$`

如果两个词语义无关，分布独立，互信息为0：
`!$P(w_j,w_i) \approx P(w_i)P(w_j) \to \frac{P(w_i,w_j)}{P(w_i)P(w_j)} \approx 1 \to log \frac{P(w_i,w_j)}{P(w_i)P(w_j)}$`


## 参考

- 有谁可以解释下word embedding? - 李韶华的回答 - 知乎
https://www.zhihu.com/question/32275069/answer/109446135

- Mikolov的word2vec —第一个现代的词嵌入生成方法
	- [Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
	- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)