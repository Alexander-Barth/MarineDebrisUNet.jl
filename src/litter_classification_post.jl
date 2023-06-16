include("litter_classification.jl")

using Glob,JSON3


files = sort(glob("*/params.json",basedir))

function loadparam(fn)
    param = Dict(JSON3.read(read(fn,String))...)
    for (k,d) = [(:beta1,0),
                 (:beta2,0),
                 (:std_noise_bands,0),
                 (:clip_grad_value,0)]
        if !haskey(param,k)
            param[k] = d
        end
    end
    return param
end

df = DataFrame([loadparam(fn) for fn in files])

df[!,:file] .= @. last(splitdir(dirname(files)))


display(select(df,:val_mean_IoU,:test_mean_IoU,:train_mean_IoU,:file))
