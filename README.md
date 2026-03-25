# 312-avoiding-rl

This repository contains several reinforcement learning models which generate
312-avoiding matrices with large permanents. Jupyter notebooks are provided for
interactive testing for several models in both Python and Julia.

I will continue to document this repo, though development of these models have
been discontinued in favor of better models utilizing FunSearch along with a
set of heuristics observed with the help of this repo.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Parameters](#parameters)
- [Methods](#methods)
- [Output](#output)
- [Theory](#theory)

## Introduction
I will add more discussion of the theory. If you would like to run the model and assess the output, you should know the largest possible permanents of 
312-avoiding (0,1)-matrices for n = 1 to 8 are known to be: 1, 2, 4, 8, 16, 32, 64, 120. That is, as an example, the largest possible permanent of a 
5x5 312-avoiding (0,1)-matrix is 32. The value is thought to be 225 for n = 9 and 424 for n = 10. Beyond that, large examples are known but none are thought
to be optimal.

Note that computing the permanent is extremely computationally expensive. The time to run grows exponentially in n, you will see this even at n = 6. With this
model, it is possible to achieve the maximal values for n = 1 to 8 given enough experimenting with the parameters. It is difficult to achieve the values
225 for n = 9 and 424 for n = 10. Beyond n = 10, large values are known but this model becomes too naive to produce them.

## Installation

* Clone the repository:
    ```bash
    git clone https://github.com/jjohnson-99/312-avoidig-rl.git
    ```

## Running the model

* Create a virtual environment and install dependencies with:
    ```bash
    uv sync
    ```
* Activate the virtual environment with:
   - MacOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
* Run the model with, assuming you remain in the root directory:
    ```bash
    cd python-models
    uv run python src/model.py
    ```

## Usage

The main script for running experiments is `python-models/src/model.py`. You can run it with various options depending on the method you want to use. Assuming you are
currently in the `python-models` directory, an example run could be:

```bash
uv run python src/model.py --size 5 --experiment_name 5x5 --epochs 25
```

## Parameters

- `--size`: The size of the nxn matrices to be generated. Default is `5`.
- `--device`: What device you want to train on (`cuda`, `cpu`, `mps`). Default is `cpu`.
- `--batch_size`: What batch size of the model. Default is `100`.
- `--percentile`: The top 100-x percentile the agent will learn from. Default is `90`.
- `--super_percentile`: The top 100-x percentile that survives to the next generation. Default is `95`.
- `--lr`: Learning rate of the model. Default is `0.0001`.
- `--epochs`: Number of epochs to run. Default is `25`, though the number should be larger to see optimal permanents for n>6.
- `--experiment_name`: Name of the experiment. Default is `experiment`.
- `--data_directory`: Directory where the data should be saved. Default is `../data`.

## Output
The experiments results will be saved in the `data` directory as `txt` files. Note that new data is appened to existing experiment output files.

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


