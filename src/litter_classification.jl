#ENV["DISPLAY"] = ""

# julia environement
using Pkg
Pkg.activate("litter",shared=true)


using ArchGDAL
using BSON
using Base.Iterators: partition
using DataStructures
using Dates
using Flux
using OneHotArrays
using PyPlot
using Random
using Statistics
using Zygote
using Interpolations
using JSON3
using DataFrames

# load a single TIFF file and concatenate all bands
function load(fname)
    raster = ArchGDAL.read(fname)
    data = cat([Matrix(ArchGDAL.getband(raster, i)) for i = 1:ArchGDAL.nraster(raster)]...,
               dims=3)
    #@show typeof(data)
    return data
end

# convert to Int 8 class index (1-based)
to_class(c) = Int8(c) + 1

# load a batch of files under the directory basedir
function loadbatch!(T,basedir,train_X,sz,nbands,idx,bands,confidence,classes,class_mapping)
    fun(x) = class_mapping[Int8(x)]
    for (li,i) in enumerate(idx)
        dir = join(split(train_X[i],'_')[1:end-1],'_')

        fname_bands = joinpath(basedir,"patches/S2_$(dir)/S2_$(train_X[i]).tif")
        fname_classes = joinpath(basedir,"patches/S2_$(dir)/S2_$(train_X[i])_cl.tif")
        fname_conf = joinpath(basedir,"patches/S2_$(dir)/S2_$(train_X[i])_conf.tif")
        # note add 1 to classes and confidence levels
        # so that it can be used more directy as an index to an
        # array

        bands[:,:,:,li] = load(fname_bands);
        confidence[:,:,li] = to_class.(load(fname_conf))
        classes[:,:,li] = fun.(load(fname_classes)[:,:,1])
    end
end


# rotate the image `bands` by θ (in degress)
function rotate_center!(bands_rot,bands,θ,extrapval)
    sinθ,cosθ = sincosd(θ)
    sz = size(bands)
    center = sz[1:2] ./ 2
    @inbounds Threads.@threads for n = 1:size(bands,4)
        for c = 1:size(bands,3)
            itp = extrapolate(
                interpolate((1:sz[1],1:sz[2]),
                            (@view bands[:,:,c,n]),Gridded(Constant())),extrapval)
            for j = 1:sz[2]
                for i = 1:sz[1]
                    ip = cosθ * (i - center[1]) - sinθ * (j - center[2]) + center[1]
                    jp = sinθ * (i - center[1]) + cosθ * (j - center[2]) + center[2]
                    bands_rot[i,j,c,n] = itp(ip,jp)
                end
            end
        end
    end
end

# concatenate channels
cat_channels(mx,x) = cat(mx, x, dims=3)

# helper function to show the size of a layer
function showsize(tag)
    return function s(x)
        @show tag,size(x)
        return x
    end
end

# individual block of a UNet
function unet_block(l,nchannels,nbands,nclasses,activation)
    nchan = nchannels[l]
    last_activation = activation
    if l == 1
        nin,nout = nbands,nclasses
        last_activation = identity
    else
        nin = nchannels[l-1]
        nout = nchannels[l]
    end
    @show l,nchannels,nin,nout,nchan

    if l == length(nchannels)
        return Conv((3,3),nchannels[l-1] => nchan,activation,pad = SamePad())
        #return identity
    end

    inner = unet_block(l+1,nchannels,nbands,nclasses,activation)


    return Chain(
        #showsize("A $nin"),
        Conv((3,3),nin => nchan,activation,pad = SamePad()),
        #showsize("B $nchan"),
        Conv((3,3),nchan => nchan,activation,pad = SamePad()),
        SkipConnection(
            Chain(
                MaxPool((2,2)),
                inner,
                #showsize("C $(nchannels[l+1]) => $nchan"),
                #Upsample(mode,scale=(2,2)),
                ConvTranspose((2,2),nchannels[l+1] => nchan,#=activation,=#stride=2),
                #showsize("D"),
            ),
            cat_channels
        ),
        #showsize("G $(2*nchan)"),
        Conv((3,3),2*nchan => nchan,activation,pad = SamePad()),
        Conv((3,3),nchan => nout,last_activation,pad = SamePad()),
        #showsize("out level $nout"),
    )
end

# plot the matrix classes
function classplot(classes; nclasses = 12, unclassified = nclasses)
    c = Float64.(copy(classes))

    new_map = plt.cm.get_cmap("jet", nclasses-1)

    c[c .== unclassified] .= NaN
    pcolormesh(c, cmap=new_map, vmin = 0.5, vmax = nclasses-0.5);
    colorbar(orientation="horizontal")
end


# compute the confusion matrix
confusion_matrix(classes,pred,obs) = [count((pred.==i) .&& (obs.==j)) for i in classes, j in classes]


"""
    compare_result(T,sz,device,model,nbands,nclasses,val_X,basedir,resdir;
                        batchsize = 8, doplot = true,
                        predicted_classes = nothing,
                        mean_bands = nothing,
                        std_bands = nothing,
                        )

If `predicted_classes` is not nothing, is should be a 3D arrays of the correct size
to save the predicted classes.

For example

```julia
predicted_classes = zeros(Int8,sz...,length(train_X))
```
"""
function compare_result(T,sz,device,model,nbands,nclasses,val_X,basedir,resdir;
                        batchsize = 8, doplot = true,
                        predicted_classes = nothing,
                        mean_bands = nothing,
                        std_bands = nothing,
                        )

    if !isnothing(resdir)
        figdir = joinpath(resdir,"Fig")
        mkpath(figdir)
    end

    val_bands = zeros(T,sz...,nbands,batchsize);
    val_confidence = zeros(Int8,sz...,batchsize);
    val_classes = zeros(Int8,sz...,batchsize);


    CM = zeros(Int,nclasses,nclasses)

    intersections = zeros(Int,nclasses-1)
    union = zeros(Int,nclasses-1)

    for idx = partition(1:length(val_X),batchsize)
        loadbatch!(T,basedir,val_X,sz,nbands,idx,val_bands,val_confidence,val_classes,class_mapping)
        X = val_bands |> device;

        if !isnothing(mean_bands)
            X = (X .- device(mean_bands)) ./ device(std_bands)
        end
        yhat = model(X);

        #    Y = permutedims( onehotbatch(val_classes,1:nclasses), (2,3,1,4)) |> device;
        #    Y_without_unclassfied = Y[:,:,1:end-1,:]

        val_class_pred = onecold(permutedims(yhat,(3,1,2,4))) |> cpu;

        # the last class means unclassfied
        is_classified = val_classes .!= nclasses;
        #@show mean(val_classes[is_classified] .== val_class_pred[is_classified])

        for (obs,prediction) in zip(val_class_pred[is_classified],val_classes[is_classified])
            for i = 1:length(intersections)
                intersections[i] += (i == obs) && (i == prediction)
                union[i] += (i == obs) || (i == prediction)
            end
        end
        CM .= CM + confusion_matrix(1:nclasses,val_class_pred,val_classes)

        if predicted_classes !== nothing
            for i = 1:length(idx)
                predicted_classes[:,:,idx[i]] = val_class_pred[:,:,i]
            end
        end

        if doplot
             for i = 1:length(idx)
                clf()
                subplot(1,2,1);
                classplot(val_classes[:,:,i])
                title("Observations")

                subplot(1,2,2);
                classplot(val_class_pred[:,:,i])
                title("Prediction")
                savefig(joinpath(figdir,val_X[idx[i]] * ".png"))
            end
        end
    end

    IoU = intersections ./ union
    mean_IoU = mean(IoU)

    #@show mean_IoU
    # CM[i,j] : i prediction, j observations
    #class_names[1:end-1]
    #CM[1:end-1,1:end-1]

    if !isnothing(resdir)
        fname = joinpath(resdir,"litter_classification_validation.bson")
        BSON.@save fname class_names IoU mean_IoU CM
    end

    return (; mean_IoU, IoU, CM)
end

# add noise to bands with the prescribed standard deviation
function addnoise!(bands,stddev)
    @inbounds for i in eachindex(bands)
        bands[i] += stddev * randn(eltype(bands))
    end
end


#=
class_names = [
    "Marine Debris",
    "Sargassum",
    "Natural Organic Material",
    "Water",
    "Sediment-Laden Water",
    "Foam",
    "unclassified/other",
]

class_mapping = Dict{Int8,Int8}(
    0 => 7, # unclassified
    1 => 1,
    2 => 2,
    3 => 2,
    4 => 3,
    5 => 7, # other
    6 => 7, # other
    7 => 4,
    8 => 5, # Sediment-Laden Water
    9 => 6, # Foam
    10 => 4,
    11 => 4,
    12 => 4,
    13 => 7,
    14 => 4,
    15 => 4,
)

=#
class_names = [
    "Marine Debris",
    "Dense Sargassum",
    "Sparse Sargassum",
    "Natural Organic Material",
    "Ship",
    "Clouds",
    "Marine Water (SC)", # 7
    "Sediment-Laden Water",
    "Foam",
    "Turbid Water",
    "Shallow Water",
#    "Waves",
#    "Cloud Shadows",
#    "Wakes",
#    "Mixed Water",
    "unclassified",
]

class_mapping = Dict{Int8,Int8}(
    (i => i for i = 1:11)...,
    12 => 7,
    13 => 7,
    14 => 7,
    15 => 7,
    0 => 12);


# basedir is assumed to have the directories splits and patches

basedir = expanduser("~/Data/MARIDA_dataset")
train_X = readlines(joinpath(basedir,"splits/train_X.txt"))
test_X = readlines(joinpath(basedir,"splits/test_X.txt"))
val_X = readlines(joinpath(basedir,"splits/val_X.txt"))



#train_X = train_X[1:64] # for testing

nclasses = maximum(values(class_mapping))

T = Float32
sz = (256,256)
nbands = 11
