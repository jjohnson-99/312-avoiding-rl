"""
Computing the permanent is #P. The below uses the Glynn
formula with a Gray code for updates. The Glynn formula
is complexity O(2^(n-1)*n^2). Using Gray codes reduces the
complexity to O(2^(n-1)*n).
"""

import numpy as np

def cmp(a, b):
    return (int(a) > int(b)) - (int(a) < int(b))  

def glynn(M):
    # add rows of M
    row_sum = np.sum(M, axis=0)
    n = len(M)

    total = 0
    old_gray = 0
    sign = +1

    
    # sum over all 2**(n-1) deltas
    binary_power_dict = {2**i:i for i in range(n)}
    num_loops = 2**(n-1)

    for bin_index in range(1, num_loops + 1):
        total += sign * np.prod(row_sum)

        # new_gray is the i-th gray code
        # gray_diff records the index which flipped between the (i-1)th and i-th gray code
        new_gray = bin_index ^ (bin_index//2)
        gray_diff = old_gray ^ new_gray
        gray_diff_index = binary_power_dict[gray_diff]

        new_vector = M[gray_diff_index]
        direction = 2 * cmp(old_gray, new_gray)

        for i in range(n):
            row_sum[i] += new_vector[i] * direction

        sign = -sign
        old_gray = new_gray

    return total/num_loops
