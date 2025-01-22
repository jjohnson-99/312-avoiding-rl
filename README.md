I plan to discuss the problem and eventually link to a preprint.

# 312-avoiding-rl
Reinforcement learning models aimed to find 312-avoiding matrices with maximal permanent. This is an extremely crude draft. It works for its intended use, but it is not pretty. I plan to package it neatly into a (multiple) script(s).


## One agent models
These work by first placing the 5 free 1's, then using the single agent to place additional 1's.

## Two agent models
Here, the first agent places crucial 1's then fill the resulting main branch. The second agent then fills the remaining 1's
