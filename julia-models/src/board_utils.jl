function board_to_string(board, n)
    # board is currently a Vector
    board = reshape(convert(Vector{Int}, board), (n,n))
    
    output = "["
    for i in 1:n-1
        output = output * string(board[:,i]) * '\n' * ' '
    end
    output = output * string(board[:,n]) * "]"

    return output
end

