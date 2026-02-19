# 312-avoiding-rl

This repository contains several reinforcement learning models which generate
312-avoiding matrices with large permanents. Jupyter notebooks are provided for
interactive testing for several models in both Python and Julia.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Parameters](#parameters)
- [Methods](#methods)
- [Results](#results)
- [Theory](#theory)

## Introduction

We introduce pattern-avoiding matrices informally, for more information on the
topics of pattern-avoiding permutations and patter-avoiding matrices, see (find
some material).

Given an $n \times n$ (0,1)-matrix $A$, we say $A$ avoids pattern $\sigma \in
S_k$ it if it avoids the $k \times k$ permutation matrix associated with
$\sigma$. We are interested in the permutation $\sigma = 312 \in S_3$, where
here the notation means $1 \rightarrow 3$, $2 \rightarrow 1$, and $3
\rightarrow 2$. The associated $3 \times 3$ permutation matrix is

$$
\begin{bmatrix}
0 & 0 & 1\\
1 & 0 & 0\\
0 & 1 & 0
\end{bmatrix}
$$


```math
\begin{bmatrix}
0 & 0 & 1\\
1 & 0 & 0\\
0 & 1 & 0
\end{bmatrix}
```


## Methods

### Single agent models

These work by first placing the 5 free 1's, then using the single agent to
place additional 1's.

### Two agent models

Here, the first agent places crucial 1's then fill the resulting main branch.
The second agent then fills the remaining 1's
