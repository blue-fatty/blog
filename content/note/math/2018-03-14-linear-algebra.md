---
title: "Linear Algebra"
date: 2018-03-14T15:47:18+08:00
slug: la
group: math
weight: 3
---

Linear Algebra notes.

## Matrix multiplication

### The origin of matrix multiplication

**参考：[数学家最初发明行列式和矩阵是为了解决什么问题？ - 马同学的回答 - 知乎](https://www.zhihu.com/question/19919917/answer/270694029)**


最初目的：解线性方程组

**举例：\\(YC_rC_b \to RGB\\)**

- 黑白电视到彩色电视
  - 兼容问题
  - \\(Y\\): 灰度图

<div>
$$
\begin{cases}
0.299R + 0.587G + 0.114B = Y \\
0.500R - 0.419G - 0.081B + 128 = C_r \\
-0.169R - 0.331G + 0.500B + 128 = C_b
\end{cases}
$$
</div>

<!--more-->

**演变过程：**

- 解方程方法：
    - 高斯消元法
        - 初中？小学？解方程的方法，逐个元素消除
    - 凯莱的高斯消元法
        - 用`数块`表示线性方程组
        - 变换写在横线上很不数学
        - `数块`乘法
        - 数块被命名为`矩阵`


高斯消元法到数块乘法的对应：

<div>
 $$
 \begin{pmatrix}
 1 & 2 & 3 \\
 3 & 4 & 5
 \end{pmatrix}
 \xrightarrow{r^{'}_2 = r_2 - 3r_1}
 \begin{pmatrix}
 1 & 2 & 3 \\
 0 & -2 & -4
 \end{pmatrix}
 $$
</div>

对应

<div>

$$
 \begin{pmatrix}
 1 & 0 \\
 -3 & 1 
 \end{pmatrix}
 \begin{pmatrix}
 1 & 2 & 3 \\
 3 & 4 & 5 
 \end{pmatrix} =
 \begin{pmatrix}
 1 & 2 & 3 \\
 0 & -2 & -4 
 \end{pmatrix} 
$$

</div>

高斯消元法完全用数块乘法表示：

<div>

$$
 \begin{pmatrix}
 1 & -2 \\
 0 & 1
 \end{pmatrix}
\begin{pmatrix}
 1 & 0 \\
 0 & -\frac{1}{2}
 \end{pmatrix}
 \begin{pmatrix}
 1 & 0 \\
 -3 & 1
 \end{pmatrix}
 \begin{pmatrix}
 1 & 2 & 3 \\
 3 & 4 & 5
 \end{pmatrix} =
 \begin{pmatrix}
 1 & 0 & -1 \\
 0 & 1 & 2
 \end{pmatrix}
$$

</div>

## PCA

### Compute with Python

- [Principal Component Analysis in Python](https://plot.ly/ipython-notebooks/principal-component-analysis/)

<!-- Linear Algebra Done right -->

## Properties of complex arithmetic

1. **commutativity (交换性)**: <br>\\(\alpha + \beta = \beta = \alpha \ \text{and} \ \alpha\beta = \beta\alpha \ \text{for all} \ \alpha,\beta \in \mathbf{C}\\)
1. **associativity (结合性)**: <br>\\((\alpha + \beta)+\lambda = \alpha + (\beta + \lambda) \ \text{and} \ (\alpha\beta)\lambda = \alpha(\beta\lambda) \ \text{for all} \ \alpha, \beta, \lambda \in \mathbf{C} \\)
1. **identities (单位元)**: <br>\\(\lambda + 0 = \lambda \ \text{and} \ \lambda 1 = \lambda \ \text{for all} \ \lambda \in \mathbf{C} \\)
1. **additive inverse (加法逆元)**: <br>\\(\text{for every } \alpha \in \mathbf{C}, \ \text{there exists a unique } \beta \in \mathbf{C} \ \text{such that} \alpha + \beta = 0\\)
1. **multiplicative inverse (乘法逆元)**: <br>\\(\text{for every } \alpha \in \mathbf{C}, \ \text{with } \alpha \ne 0, \ \text{there exists a unique } \beta \in \textbf{C} \text{ such that } \alpha \beta = 1 \\)
1. **distributive property (分配性)**: <br>\\(\lambda (\alpha + \beta) = \lambda \alpha + \lambda \beta \ \text{for all} \ \alpha,\beta \in \mathbf{C}\\)

