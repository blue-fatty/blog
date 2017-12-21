---
title: "Forward Learning"
date: 2017-12-18T19:01:49+08:00
slug: forward
tags: ['deep learning', 'CNN', 'OBV']
---

Papers read during research **forward deep learning** algorithm.

## Orthogonal Bipolar Target Vectors[^1]

**Can OBV construct a middle target for CNN?**

A kind of target representation.

- conventional
    - BNV - binary: \\((0, 0, 1, 0, 0)\\)
    - BPV - bipolar?: \\((-1, -1, 1, -1, -1)\\)
- OBV - orthogonal bipolar vectors
- NOV - Non-Orthogonal Vecotrs
    - For fail comparision
    - \\(V_i=(\overbrace{-1 , \cdots , -1}^{i-1}, 1, \overbrace{-1 , \cdots , -1}^{n-i})\\)
    - \\(cos \theta = \frac{n-2}{n}\\)
- degraded characters?
    - They use degraded license plate images as expirement data. (车牌号)

**How to generate OBV from conventional target?**

<!--more-->

---

**OBV (Orthogonal Bipolar Vector)**

- Bipolar: \\(0 \to -1 \\)
- Orthogonal: \\( V\_{2^k}^{i} \cdot  V\_{2^k}^{j} = 0 \\)
- \\(V_{2^k}^m\\)
    - \\(2^k\\) - Can be used to represent \\(2^k\\) classed
    - \\(k\\) - Can be constructed in k steps (裂变)
    - \\(m\\) - \\(m_{th}\\) vector

Number of components in an OBV:

$$
n=2^km \\\\\\
V_{m}^{0} = (\overbrace{1, 1, \cdots , 1}^{m})^T
$$

**Example of generating OBVs**

Take four classes classification for example. Let's say four labels are 1, 2, 3, 4. <br>

**Step.1** Initialize parameters.

$$
m=1, k=2
$$

- \\(m\\) can be set to 1, 2, 3, 4, ... 
- \\(k\\) should satisfy \\(2^k \ge 4\\)

**Step.2** Initialize \\(V\_1^0 = (1)^T\\) <br>
**Step.3**
$$
\begin{align}
& V\_2^1 = ({V\_1^0}^T, {V\_1^0}^T) = (1, 1)^T \\\\\\
& V\_2^2 = ({V\_1^0}^T, -{V\_1^0}^T) = (1, -1)^T
\end{align}
$$

- Obviously, \\(V\_2^1 \cdot V\_2^2 = 0\\)

**Step.4**
$$
\begin{align}
& V\_4^1 = ({V\_2^1}^T, {V\_2^1}^T) = (1, 1, 1, 1)^T \\\\\\
& V\_4^2 = ({V\_2^1}^T, -{V\_2^1}^T) = (1, 1, -1, -1)^T \\\\\\
& V\_4^3 = ({V\_2^2}^T, {V\_2^2}^T) = (1, -1, 1, -1)^T \\\\\\
& V\_4^4 = ({V\_2^2}^T, -{V\_2^2}^T) = (1, -1, -1, 1)^T \\\\\\
\end{align}
$$

- We can use these four vectors to represent 1, 2, 3, 4

[^1]: Improved MLP Learning via Orthogonal Bipolar Target Vectors

## References

