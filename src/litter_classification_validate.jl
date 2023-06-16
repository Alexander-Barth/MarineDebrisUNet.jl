include("litter_classification.jl")

timestamp = "20230215T223530"

device = gpu

BSON.@load joinpath(basedir, "litter_classification_model_$(timestamp).bson") model_cpu mean_bands_cpu std_bands_cpu lr nepochs batchsize nclasses


model = model_cpu |> device
mean_bands = mean_bands_cpu |> device
std_bands = std_bands_cpu |> device


doplot = false;
resdir = joinpath(basedir,timestamp,"train-test")
train_stat = compare_result(T,sz,nbands,nclasses,train_X,basedir,resdir,
                            doplot = doplot,
                            mean_bands = mean_bands,
                            std_bands = std_bands,
                            predicted_classes = train_predicted_classes)


val_predicted_classes = zeros(Int8,sz...,length(val_X))
resdir = joinpath(basedir,timestamp,"val-test")
val_stat = compare_result(T,sz,nbands,nclasses,val_X,basedir,resdir;
                          doplot = doplot,
                          mean_bands = mean_bands,
                          std_bands = std_bands,
                          predicted_classes = val_predicted_classes)

test_predicted_classes = zeros(Int8,sz...,length(test_X))
resdir = joinpath(basedir,timestamp,"test-test")
test_stat = compare_result(T,sz,nbands,nclasses,test_X,basedir,resdir,
                           doplot = doplot,
                           mean_bands = mean_bands,
                           std_bands = std_bands,
                           predicted_classes = test_predicted_classes)


function loadall(T,sz,nbands,train_X,basedir)
    all_bands = zeros(T,sz...,nbands,length(train_X));
    all_confidence = zeros(Int8,sz...,length(train_X));
    all_classes = zeros(Int8,sz...,length(train_X));
    loadbatch!(T,basedir,train_X,sz,nbands,1:length(train_X),all_bands,all_confidence,all_classes,class_mapping)

    return all_bands, all_confidence, all_classes
end


val_bands, val_confidence, val_classes = loadall(T,sz,nbands,val_X,basedir)
test_bands, test_confidence, test_classes = loadall(T,sz,nbands,test_X,basedir)


using PyCall
metrics = pyimport("sklearn.metrics")

# last class means "unclassified"
isclassified = val_classes .!== nclasses;
metrics.jaccard_score(val_classes[isclassified],
                      val_predicted_classes[isclassified],average="macro")


metrics.jaccard_score(val_classes[isclassified],
                      val_predicted_classes[isclassified],average=nothing)



isclassified = test_classes .!== nclasses;
metrics.jaccard_score(test_classes[isclassified],
                      test_predicted_classes[isclassified],average="macro")

fname = joinpath(basedir,timestamp,"results-$timestamp.nc")
using NCDatasets
ds = NCDataset(fname,"c")
defVar(ds,"val_predicted_classes",val_predicted_classes,("x","y","val_sample"))
defVar(ds,"test_predicted_classes",test_predicted_classes,("x","y","test_sample"))

defVar(ds,"val_classes",val_classes,("x","y","val_sample"))
defVar(ds,"test_classes",test_classes,("x","y","test_sample"))

defVar(ds,"val_X",val_X,("val_sample",))
defVar(ds,"test_X",test_X,("test_sample",))

close(ds)
