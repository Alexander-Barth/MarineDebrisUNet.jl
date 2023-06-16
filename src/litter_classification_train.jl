include("litter_classification.jl")

function main(T,sz,
              basedir,
              train_X,
              test_X,
              val_X,
              class_mapping,
              nbands,
              nclasses;
              lr = 0.0001,
              nepochs = 500,
              #nepochs = 1,
              std_noise_bands = 0.,
              clip_grad_value = 0.,
              activation = relu,
              mode = :nearest,
              seed = 1234,
              nchannels_base = 64,
              nchannels = (nchannels_base,nchannels_base*2,nchannels_base*4),
              device = gpu,
              batchsize = 8,
              beta2 = T(0),
              beta1 = T(0),
              )


    # allocation
    all_bands = zeros(T,sz...,nbands,length(train_X));
    all_confidence = zeros(Int8,sz...,length(train_X));
    all_classes = zeros(Int8,sz...,length(train_X));
    loadbatch!(T,basedir,train_X,sz,nbands,1:length(train_X),all_bands,all_confidence,all_classes,class_mapping)

    #=
    using PyPlot
    bands = all_bands[:,:,:,1:8];
    bands_rot = zeros(T,size(bands));
    #figure(),pcolormesh(bands[:,:,1,1]')
    θ = 20;
    @time rotate_center!(bands_rot,bands,θ,NaN)
    @time rotate_center!(bands_rot,bands,θ,NaN)

    figure();pcolor(bands_rot[:,:,1,1]')
    =#
    # weight for all classes

    count_classes = counter(all_classes)

    ww = [1/count_classes[i] for i = 1:nclasses]
    ww[end] = 0 # last is unclassified
    ww = reshape(ww,(1,1,nclasses,1)) |> device

    isgood = count(isnan.(all_bands),dims=(1,2,3))[1,1,1,:] .== 0
    @show train_X[findall(.!isgood)]
    size(train_X)

    all_bands = all_bands[:,:,:,findall(isgood)];
    all_confidence = all_confidence[:,:,findall(isgood)];
    all_classes = all_classes[:,:,findall(isgood)];
    train_X = train_X[findall(isgood)];

    mean_bands = mean(all_bands,dims=(1,2,4))
    std_bands = std(all_bands,dims=(1,2,4))

    @show mean_bands
    @show std_bands

    # normalize
    all_bands = (all_bands .- mean_bands) ./ std_bands;

    idx = 1:batchsize
    bands = zeros(T,sz...,nbands,batchsize)
    confidence = zeros(Int8,sz...,batchsize)
    classes = zeros(Int8,sz...,batchsize)

    confidence_weights = [0.5, 1., 1., 1.]

    loadbatch!(T,basedir,train_X,sz,nbands,idx,bands,confidence,classes,class_mapping)
    bands .= all_bands[:,:,:,idx]
    confidence .= all_confidence[:,:,idx]
    classes .= all_classes[:,:,idx]

    # make sure that net cannot predict unclassified (last class)
    model = unet_block(1,nchannels,nbands,nclasses-1,activation)
    model = model |> device

    #=

    #mean_bands = mean(bands,dims=(1,2)) |> device
    #std_bands = std(bands,dims=(1,2)) |> device

    X = bands |> device;
    Y = permutedims( onehotbatch(classes,1:nclasses), (2,3,1,4)) |> device
    Y_without_unclassfied = Y[:,:,1:end-1,:]

    #nchannels = (64,64)


    yhat = model(X);
=#

    losses = []

    parameters = Flux.params(model)


    @info "parameters" lr nepochs std_noise_bands clip_grad_value activation mode nchannels seed beta1 beta2

    opt = Flux.Optimiser(Flux.Optimise.ClipValue(clip_grad_value), ADAM(lr))

    Random.seed!(seed)

    ww_without_unclassfied = ww[:,:,1:end-1,1]
    confidence = all_confidence[:,:,1:batchsize]
    classes = all_classes[:,:,idx]


    timestamp = Dates.format(Dates.now(),"yyyymmddTHHMMSS")
    outdir = joinpath(basedir,timestamp)
    mkpath(outdir)

    for fn in ["litter_classification.jl",
               "litter_classification_train.jl",
               "litter_classification_validate.jl"]
        println("copy ",joinpath(dirname(@__FILE__),fn))
        cp(joinpath(dirname(@__FILE__),fn),joinpath(outdir,fn))
    end


    val_stats = []

    @time for n = 1:nepochs
        average_loss = 0.
        average_count = 0

        for idx in partition(shuffle(1:size(all_bands,4)),batchsize)
            #loadbatch!(T,basedir,train_X,sz,nbands,idx,bands,confidence,classes,class_mapping)
            #bands = all_bands[:,:,:,idx]
            #confidence = all_confidence[:,:,idx]
            #classes = all_classes[:,:,idx]

            # random rotation
            # edges will be filled with zeros and "unclassfied" class

            θ = 360*rand()
            rotate_center!(bands,all_bands[:,:,:,idx],θ,0)
            #rotate_center!(confidence,all_confidence[:,:,idx],θ,0)
            rotate_center!(classes, all_classes[:,:,idx],θ,nclasses)

            if rand(1:2) == 1
                bands = reverse(bands,dims=1)
                classes = reverse(classes,dims=1)
            end

            addnoise!(bands,std_noise_bands)
            # TO GPU
            #confidence = confidence |> device

            X = bands |> device;
            Y = permutedims( onehotbatch(classes,1:nclasses), (2,3,1,4)) |> device;

            Y_without_unclassfied = Y[:,:,1:end-1,:]

            loss,back = Flux.pullback(parameters) do
                yhat = model(X);

                loss = sum(-sum(ww_without_unclassfied .* Y_without_unclassfied .* logsoftmax(yhat; dims=3); dims=3))

                if beta2 != 0
                    loss += T(beta2) * sum(parameters) do p
                        ( ndims(p) > 1 ? sum(abs2,p) : 0)
                    end
                end

                loss
            end

            average_loss += loss
            average_count += 1
            grads = back(T(1))


            Flux.Optimise.update!(opt, parameters, grads)
        end

        # if n % 50 == 0
        #     opt.eta /= 2
        # end

        average_loss /= average_count
        push!(losses,average_loss)
        print("$n: $average_loss")

        #=
        resdir = joinpath(basedir,timestamp,"val")
        val_stat = compare_result(T,sz,nbands,nclasses,val_X,basedir,resdir;
        doplot = false,
        mean_bands = mean_bands,
        std_bands = std_bands)
        push!(val_stats,val_stat)
        println("$(val_stat.mean_IoU)")
        =#

        println()
    end


    #=
    figure()
    n = 1
    subplot(2,2,1); classplot(classes[:,:,n])
    subplot(2,2,2); classplot(classes_pred[:,:,n])
    subplot(2,2,3); pcolormesh(bands[:,:,11,n]); colorbar(orientation="horizontal")
    =#

    #Conv((3,3),64 => 64,selu,pad = SamePad())


    #=
    pcolormesh(classes[:,:,1]' .== 1); colorbar()
    =#
    #isfile(fname_bands)

    #splits/test_X.txt
    #splits/val_X.txt

    model_cpu = cpu(model)
    mean_bands_cpu = cpu(mean_bands)
    std_bands_cpu = cpu(std_bands)

    BSON.@save joinpath(basedir, "litter_classification_model_$(timestamp).bson") model_cpu mean_bands_cpu std_bands_cpu lr nepochs batchsize nclasses losses val_stats std_noise_bands



    doplot = false;
    resdir = joinpath(basedir,timestamp,"train")
    train_stat = compare_result(T,sz,device,model,nbands,nclasses,train_X,basedir,resdir,
                                doplot = doplot,
                                mean_bands = mean_bands,
                                std_bands = std_bands)

    resdir = joinpath(basedir,timestamp,"val")
    val_stat = compare_result(T,sz,device,model,nbands,nclasses,val_X,basedir,resdir;
                              doplot = doplot,
                              mean_bands = mean_bands,
                              std_bands = std_bands)


    resdir = joinpath(basedir,timestamp,"test")
    test_stat = compare_result(T,sz,device,model,nbands,nclasses,test_X,basedir,resdir;
                               doplot = doplot,
                               mean_bands = mean_bands,
                               std_bands = std_bands)


    paramsname = joinpath(basedir,timestamp,"params.json")
    open(paramsname,"w") do f
        JSON3.write(f,OrderedDict(
            "lr" => lr,
            "epoch" => nepochs,
            "seed" => seed,
            "activation" => "$activation",
            "nchannels" => nchannels,
            "batchsize" => batchsize,
            "beta2" => beta2,
            "beta1" => beta1,
            "mode" => mode,
            "std_noise_bands" => std_noise_bands,
            "clip_grad_value" => clip_grad_value,
            "val_mean_IoU" =>  val_stat.mean_IoU,
            "test_mean_IoU" =>  test_stat.mean_IoU,
            "train_mean_IoU" =>  train_stat.mean_IoU,
        ))
    end



    @show test_stat.mean_IoU
    @show train_stat.mean_IoU

    return train_stat,val_stat,test_stat
end


Random.seed!(Dates.value(Dates.now()))

lr = rand([0.0001,0.0003,0.001])
nepochs = rand(300:2000)
std_noise_bands = rand([0.0, 0.001, 0.01])
clip_grad_value = rand([1e5, 5, 1, 1e-1, 1e-2])
activation = rand([relu,selu,gelu])

nchannels_base = rand(8:64)
nchannels = (nchannels_base,nchannels_base*2,nchannels_base*4)
#beta2 = T(10 .^ (-2 + -3 * rand()))

#p2 = JSON3.read(read(expanduser("~/Data/MARIDA_dataset/20230422T104137/params.json"),String))

p = JSON3.read(read(expanduser("~/Data/MARIDA_dataset/20230318T201117/params.json"),String))
lr = p.lr
nepochs = p.epoch
std_noise_bands = p.std_noise_bands
clip_grad_value = p.clip_grad_value
activation = selu

nchannels_base = p.nchannels[1]
nchannels = (nchannels_base,nchannels_base*2,nchannels_base*4)
beta2 = 0.f0


params  = (lr=p.lr,
           nepochs = p.epoch,
           activation = selu,
           nchannels_base = p.nchannels[1])


other(x::AbstractFloat) = (.95*x,1.05*x)
other(x::Integer) = round.(typeof(x),(.95*x,1.05*x))
other(x::Function) = (relu,gelu)

for (k,v) in pairs(params)
    for v_pert = other(v)
        @show k,v,v_pert
        params_pert = merge(params,NamedTuple((k => v_pert,)))
        @show params_pert

        train_stat,val_stat,test_stat = main(
            T,sz,basedir,train_X,test_X,val_X,class_mapping,nbands,nclasses;
            std_noise_bands = std_noise_bands,
            clip_grad_value = clip_grad_value,
            params_pert...
        )
    end
end

#nepochs = 2


#=
display(DataFrame(class_names=class_names[1:end-1],IoU = test_stat.IoU))
=#


@time train_stat,val_stat,test_stat = main(
    T,sz,basedir,train_X,test_X,val_X,class_mapping,nbands,nclasses;
    lr = lr,
    nepochs = nepochs,
    std_noise_bands = std_noise_bands,
    clip_grad_value = clip_grad_value,
    activation = activation,
    beta2 = beta2,
    nchannels = nchannels,
)
