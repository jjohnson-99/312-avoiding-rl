function glynn(M::AbstractMatrix{T}) where T
    """
    Computing the permanent is #P. The below uses the Glynn
    formula with a Gray code for updates. The Glynn formula
    is complexity O(2^(n-1)*n^2). Using Gray codes reduces the
    complexity to O(2^(n-1)*n).
    """
    size(M)[1] == size(M)[2] ? n=size(M)[1] : error("Non square matrix as input")

    row_ = [M[:,i] for i in 1:n]
    row_sum = [sum(row) for row in row_]

    total = 0
    old_gray = 0
    sign = +1

    binary_power_dict = Dict(2^k => k for k in 0:n)
    num_iter = 2^(n-1)

    for bin_index in 1:num_iter
        total += sign * reduce(*, row_sum)

        new_gray = bin_index ⊻ trunc(Int, bin_index/2) # ⊻ XOR
        gray_diff = old_gray ⊻ new_gray
        gray_diff_index = binary_power_dict[gray_diff]

        new_vector = M[gray_diff_index+1,:]
        direction = 2 * cmp(old_gray, new_gray)

        for i in 1:n
            row_sum[i] += new_vector[i] * direction
        end

        sign = -sign
        old_gray = new_gray
    end

    return total/num_iter

end


function fast_glynn(M, niter)

    """
    Approximates the permanent of M up to an additive error through the Gurvits/Glynn algorithm
    see https://arxiv.org/abs/1212.0025
    """


    function glynn_estimator(M,x)

        function product_M_x(M, x)
            result_product = one(typeof(M[1,1]))

            for j = 1:n
                result_product *= sum(M[j, :] .* x)
            end

            result_product
        end

        prod(x) * product_M_x(M, x)

    end

    n = size(M,1)
    result = zero(eltype(M))

    for i = 1:niter

        x = rand([-1,1], n)
        result += 1/niter * glynn_estimator(M,x)

    end

    return result

end