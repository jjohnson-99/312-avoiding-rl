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
- [Output](#output)
- [Theory](#theory)

## Introduction

## Installation

## Usage

## Parameters

## Methods

### Single agent models

These work by first placing the 5 free 1's, then using the single agent to
place additional 1's.

### Two agent models

Here, the first agent places crucial 1's then fill the resulting main branch.
The second agent then fills the remaining 1's

## Output

## Theory

We introduce pattern-avoiding matrices informally, for more information on the
topics of pattern-avoiding permutations and patter-avoiding matrices, see (find
some material).

Given an $n \times n$ (0, 1)-matrix $A$, we say $A$ avoids pattern $\sigma \in
S_k$ it if it avoids the $k \times k$ permutation matrix associated with
$\sigma$. We are interested in the permutation $\sigma = 312 \in S_3$, where
here the notation means $1 \rightarrow 3$, $2 \rightarrow 1$, and $3
\rightarrow 2$. The associated $3 \times 3$ permutation matrix is

```math
P_{312} = \begin{bmatrix}
0 & 0 & 1\\
1 & 0 & 0\\
0 & 1 & 0
\end{bmatrix}
```

Informally, the (0, 1)-matrix $A$ contains $P_{312}$ if $A$ has a $3 \times 3$
submatrix $B$ such that $B$ contains 1's where $P_{312}$ does. Otherwise $A$
avoids $P_{312}$, and thus avoids pattern $\sigma = 312$. As examples

```math
A_1 = \begin{bmatrix}
1 & 1 & 1 & 0\\
0 & 1 & 1 & 1\\
0 & 1 & 1 & 1\\
1 & 1 & 0 & 1
\end{bmatrix},
\qquad
A_2 = \begin{bmatrix}
0 & 1 & 1 & 1\\
0 & 1 & 1 & 1\\
0 & 1 & 0 & 1\\
1 & 1 & 1 & 1
\end{bmatrix}
```

$A_1$ is 312-avoiding while $A_2$ is not. To see why $A_2$ is not 312-avoiding,
take, for example, entries $a_{14}$, $a_{22}$, and $a_{43}$. There are often
multiple instances of the pattern, we could have taken the triple ($a_{14}$,
$a_{32}$, $a_{43}$) or ($a_{24}$, $a_{32}$, $a_{43}$).

### The permanent

Since we are primarily interested in generating 312-avoiding matrices with large
permanent, we discuss it rigorously here.

The permanent of an $n \times n$ matrix $A$ is defined as

```math
\text{perm}(A) = \sum_{\sigma \in S_n}\prod_{i=1}^n a_{i,\sigma(i)}.
```

If $A$ is a (0, 1)-matrix, the permanent is the number of permutations which
$A$ contains. When seeking 312-avoiding matrices with large permanents, we are
in reality seeking matrices which contain many permutations without introducing
the pattern 312.

You may have noticed the usage of the terms 'pattern' and 'permutation'.
Generally, pattern will mean a permutation contained in $S_k$ where $k\leq n$,
while a permutation will be in the usual sense, i.e., an element of $S_n$.


