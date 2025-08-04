include("litter_classification.jl")


# train the neural network to classify marine litter
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
#    ww[1] = ww[1]*10 #more weight for plastic
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

    # normalize all the dataset 
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

    parameters = Flux.trainables(model)


    @info "parameters" lr nepochs std_noise_bands clip_grad_value activation mode nchannels seed beta1 beta2

    opt = OptimiserChain(Flux.Optimise.ClipValue(clip_grad_value), Flux.Adam(lr))
    opt_state = Flux.setup(opt, model)

    Random.seed!(seed)
    @show seed
    
    ww_without_unclassfied = ww[:,:,1:end-1,1]
    confidence = all_confidence[:,:,1:batchsize]
    classes = all_classes[:,:,idx]


    timestamp = Dates.format(Dates.now(),"yyyymmddTHHMMSSs")
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
        @show n
        
        for idx in partition(shuffle(1:size(all_bands,4)),batchsize) #Shuffle training images
            @show idx
#        for idx in partition(1:size(all_bands,4),batchsize) #No Shuffle
#            @show idx
            #loadbatch!(T,basedir,train_X,sz,nbands,idx,bands,confidence,classes,class_mapping)
            #bands = all_bands[:,:,:,idx]
            #confidence = all_confidence[:,:,idx]
            #classes = all_classes[:,:,idx]

            # random rotation
            # edges will be filled with zeros and "unclassfied" class

            θ = 360*rand()

                   
            rotate_center!(bands,all_bands[:,:,:,idx],θ,0)
#            rotate_center!(confidence,all_confidence[:,:,idx],θ,0)
            rotate_center!(classes, all_classes[:,:,idx],θ,nclasses)

            if rand(1:2) == 1
                bands = reverse(bands,dims=1)
 #               confidence = reverse(confidence,dims=1)
                classes = reverse(classes,dims=1)
            end

            addnoise!(bands,std_noise_bands)
            # TO GPU
            #confidence = confidence |> device FAIT BUGGER SI ACTIVé

            X = bands |> device;
            Y = permutedims( onehotbatch(classes,1:nclasses), (2,3,1,4)) |> device;

            Y_without_unclassfied = Y[:,:,1:end-1,:]

  #          confidence = reshape(confidence,size(confidence,1),size(confidence,2),1,size(confidence,3))


            loss,grads = Flux.withgradient(model) do m
                yhat = m(X);
                
                loss = sum(-sum(ww_without_unclassfied .* Y_without_unclassfied .* logsoftmax(yhat; dims=3); dims=3)) 
                #confidence taken into account in loss function
                
#                loss = sum(-sum(((4 .-confidence)./3).*ww_without_unclassfied .* Y_without_unclassfied .* logsoftmax(yhat; dims=3); dims=3))


                if beta2 != 0
                    loss += T(beta2) * sum(parameters) do p
                        ( ndims(p) > 1 ? sum(abs2,p) : 0)
                    end
                end

                loss
            end

            average_loss += loss
            average_count += 1


            Flux.update!(opt_state, model, grads[1])
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

    #######Calculating the metrics #########
    
    train_bands, train_confidence, train_classes = loadall(T,sz,nbands,train_X,basedir)
    val_bands, val_confidence, val_classes = loadall(T,sz,nbands,val_X,basedir)
    test_bands, test_confidence, test_classes = loadall(T,sz,nbands,test_X,basedir)

    ####CALCULATING METRICS####


    #####TRAINING DATA#####
    isclassified = train_classes .!== nclasses;
    train_jaccard_score_macro = metrics.jaccard_score(train_classes[isclassified],
                      train_predicted_classes[isclassified],average="macro")



    train_f1_score = metrics.f1_score(train_classes[isclassified],
                 train_predicted_classes[isclassified],average="macro")


    train_jaccard_score_nothing =  metrics.jaccard_score(train_classes[isclassified],
                      train_predicted_classes[isclassified],average=nothing)


    #####VALIDATE DATA#####
    # last class means "unclassified"
    isclassified = val_classes .!== nclasses;
    val_jaccard_score_macro = metrics.jaccard_score(val_classes[isclassified],
                      val_predicted_classes[isclassified],average="macro")



    val_f1_score = metrics.f1_score(val_classes[isclassified],
                 val_predicted_classes[isclassified],average="macro")


    val_jaccard_score_nothing =  metrics.jaccard_score(val_classes[isclassified],
                      val_predicted_classes[isclassified],average=nothing)


    ####TEST DATA####
    isclassified = test_classes .!== nclasses;
    test_jaccard_score_macro = metrics.jaccard_score(test_classes[isclassified],
                      test_predicted_classes[isclassified],average="macro")



    test_f1_score = metrics.f1_score(test_classes[isclassified],
                 test_predicted_classes[isclassified],average="macro")



    test_jaccard_score_nothing =  metrics.jaccard_score(test_classes[isclassified],
                      test_predicted_classes[isclassified],average=nothing)


    ####SAVING METRICS####
    metricsname = joinpath(basedir,timestamp,"metrics.json")
    open(metricsname,"w") do f
        JSON3.write(f,OrderedDict(
            "train_mean_IoU" => train_jaccard_score_macro,
            "train_IoU" => train_jaccard_score_nothing,
            "train_f1_score" => train_f1_score,
            "val_mean_IoU" => val_jaccard_score_macro,
            "val_IoU" => val_jaccard_score_nothing,
            "val_f1_score" => val_f1_score,
            "test_mean_IoU" => test_jaccard_score_macro,
            "test_IoU" => test_jaccard_score_nothing,
            "test_f1_score" => test_f1_score,
        ))
    end


    fname = joinpath(basedir,timestamp,"results-$timestamp.nc")

    ds = NCDataset(fname,"c")
    defVar(ds,"val_predicted_classes",val_predicted_classes,("x","y","val_sample"))
    defVar(ds,"test_predicted_classes",test_predicted_classes,("x","y","test_sample"))

    defVar(ds,"val_classes",val_classes,("x","y","val_sample"))
    defVar(ds,"test_classes",test_classes,("x","y","test_sample"))

    defVar(ds,"val_X",val_X,("val_sample",))
    defVar(ds,"test_X",test_X,("test_sample",))

    close(ds)

    

    return train_stat,val_stat,test_stat
end




# # random search
# lr = rand([0.0001,0.0003,0.001])
# nepochs = rand(300:2000)
# std_noise_bands = rand([0.0, 0.001, 0.01])
# clip_grad_value = rand([1e5, 5, 1, 1e-1, 1e-2])
# activation = rand([relu,selu,gelu])

# nchannels_base = rand(8:64)
# nchannels = (nchannels_base,nchannels_base*2,nchannels_base*4)
# #beta2 = T(10 .^ (-2 + -3 * rand()))



#Seed reevaluated with time
#Random.seed!(Dates.value(Dates.now()))
#@show rand()

# best values so far
lr = 0.0001
# nepochs = 1746

nepochs = 1000 # testing
activation = selu
#activation = relu
nchannels_base = 64
nchannels = (nchannels_base,nchannels_base*2,nchannels_base*4)
#beta2 = parse(Float32,ARGS[1])
#beta2 = 0.000001f0
#beta2 = T(10 .^ (-2 + -5 * rand())) #need a not fixed seed
beta2 = 0.0f0
#beta2 = 0.0001f0
#std_noise_bands = parse(Float32,ARGS[2])
std_noise_bands = 0.0
#std_noise_bands = 0.0001
clip_grad_value = 5


#Seed fixed to get same initial weights
seed = 1234
Random.seed!(seed)
@show rand()
metrics = pyimport("sklearn.metrics")

#CUDA.seed!(seed)
#random_cuda = CUDA.rand(1)
#@show typeof(random_cuda)
#@show Array(random_cuda)[1]



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

# Fine Tuning

# # read parameters from params.json file
# paramsname = expanduser("~/Data/MARIDA_dataset/20230318T201117/params.json")
# p = JSON3.read(read(paramsname,String))
# lr = p.lr
# nepochs = p.epoch
# std_noise_bands = p.std_noise_bands
# clip_grad_value = p.clip_grad_value

# params  = (lr=p.lr,
#            nepochs = p.epoch,
#            activation = selu,
#            nchannels_base = p.nchannels[1])


# other(x::AbstractFloat) = (.95*x,1.05*x)
# other(x::Integer) = round.(typeof(x),(.95*x,1.05*x))
# other(x::Function) = (relu,gelu)

# for (k,v) in pairs(params)
#     for v_pert = other(v)
#         @show k,v,v_pert
#         params_pert = merge(params,NamedTuple((k => v_pert,)))
#         @show params_pert

#         train_stat,val_stat,test_stat = main(
#             T,sz,basedir,train_X,test_X,val_X,class_mapping,nbands,nclasses;
#             std_noise_bands = std_noise_bands,
#             clip_grad_value = clip_grad_value,
#             params_pert...
#         )
#     end
# end
