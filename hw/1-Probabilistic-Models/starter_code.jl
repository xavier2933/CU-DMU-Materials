import DMUStudent.HW1

#------------- 
# Problem 4
#-------------

function f(a, bs)

    aType = eltype(a) # get type of a to return correct type
    res = Vector{Vector{aType}}()

    for val in bs
        push!(res, a * val)
    end
    @show res

    """
    ChatGPT told me about this function, liked it more than iterating with for loop
    Applies max element wise to vector of vectors, returning max of each position
    in each vector
    maxRes = reduce(max, res)
    
    This forum post says these are equivalent, couldn't find anything good on reduce 
    so didn't want to use it
    https://discourse.julialang.org/t/max-of-a-vector/108294/10
    """

    maxRes = maximum(res)

    return maxRes
end


# You can can test it yourself with inputs like this
a = [1.0 2.0; 3.0 4.0]
@show a
bs = [[1.0, 2.0], [3.0, 4.0]]
@show bs
@show f(a, bs)

# This is how you create the json file to submit
HW1.evaluate(f, "xavier.okeefe@colorado.edu")
