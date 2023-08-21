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

i = findmax(df.val_mean_IoU)[2]

allnames = [
    "activation",
    "batchsize",
    "nchannels",
    "clip_grad_value",
    "beta2",
#    "std_noise_bands",
    "lr",
    "epoch",
    "train_mean_IoU",
    "val_mean_IoU",
    "test_mean_IoU",
]

nicenames = Dict(
    "activation" => "activation function",
    "batchsize" => "batch size",
    "nchannels" => "number of channels",
    "clip_grad_value" => "gradient clipping threshold",
#    "beta2",
    "std_noise_bands" => "standard deviation of noise",
    "lr" => "learning rate",
    "epoch" => "number of epochs",
#    "val_mean_IoU",
#    "train_mean_IoU",
#    "test_mean_IoU",
)


for n in allnames
    nn = get(nicenames,n,n)
    v = df[i,n]
    if v isa AbstractFloat
        v = round(v,digits=3)
    end
    println(nn,'\t',v)
end
df[i,:]
