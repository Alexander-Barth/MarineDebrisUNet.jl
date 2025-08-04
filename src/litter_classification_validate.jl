
include("litter_classification.jl")

timestamp = "20230215T223530"
timestamp = "20230422T104137"
timestamp = "20250404T115717"
#timestamp = "20250515T144653"
#timestamp = "20250519T172334"
#timestamp = "20250519T135743"
timestamp = "20250709T155235308"

device = gpu

BSON.@load joinpath(basedir, "litter_classification_model_$(timestamp).bson") model_cpu mean_bands_cpu std_bands_cpu lr nepochs batchsize nclasses


model = model_cpu |> device
mean_bands = mean_bands_cpu |> device
std_bands = std_bands_cpu |> device


train_predicted_classes = zeros(Int8,sz...,length(train_X))
val_predicted_classes = zeros(Int8,sz...,length(val_X))
test_predicted_classes = zeros(Int8,sz...,length(test_X))

doplot = false;
resdir = joinpath(basedir,timestamp,"train")
train_stat = compare_result(T,sz,device,model,nbands,nclasses,train_X,basedir,resdir,
                            doplot = doplot,
                            mean_bands = mean_bands,
                            std_bands = std_bands,
                            predicted_classes = train_predicted_classes)


val_predicted_classes = zeros(Int8,sz...,length(val_X))
resdir = joinpath(basedir,timestamp,"val")
val_stat = compare_result(T,sz,device,model,nbands,nclasses,val_X,basedir,resdir;
                          doplot = doplot,
                          mean_bands = mean_bands,
                          std_bands = std_bands,
                          predicted_classes = val_predicted_classes)

test_predicted_classes = zeros(Int8,sz...,length(test_X))
resdir = joinpath(basedir,timestamp,"test")
test_stat = compare_result(T,sz,device,model,nbands,nclasses,test_X,basedir,resdir,
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

train_bands, train_confidence, train_classes = loadall(T,sz,nbands,train_X,basedir)
val_bands, val_confidence, val_classes = loadall(T,sz,nbands,val_X,basedir)
test_bands, test_confidence, test_classes = loadall(T,sz,nbands,test_X,basedir)

function loadall(T,sz,nbands,train_X,basedir)
    all_bands = zeros(T,sz...,nbands,length(train_X));
    all_confidence = zeros(Int8,sz...,length(train_X));
    all_classes = zeros(Int8,sz...,length(train_X));
    loadbatch!(T,basedir,train_X,sz,nbands,1:length(train_X),all_bands,all_confidence,all_classes,class_mapping)

    return all_bands, all_confidence, all_classes
end

train_bands, train_confidence, train_classes = loadall(T,sz,nbands,train_X,basedir)
val_bands, val_confidence, val_classes = loadall(T,sz,nbands,val_X,basedir)
test_bands, test_confidence, test_classes = loadall(T,sz,nbands,test_X,basedir)

####CALCULATING METRICS####

using PyCall
metrics = pyimport("sklearn.metrics")

#####TRAINING DATA#####
isclassified = train_classes .!== nclasses;
train_jaccard_score_macro = metrics.jaccard_score(train_classes[isclassified],
                      train_predicted_classes[isclassified],average="macro")

@show train_jaccard_score_macro

train_f1_score = metrics.f1_score(train_classes[isclassified],
                 train_predicted_classes[isclassified],average="macro")

@show train_f1_score 
train_jaccard_score_nothing =  metrics.jaccard_score(train_classes[isclassified],
                      train_predicted_classes[isclassified],average=nothing)

@show train_jaccard_score_nothing 


#####VALIDATE DATA#####
# last class means "unclassified"
isclassified = val_classes .!== nclasses;
val_jaccard_score_macro = metrics.jaccard_score(val_classes[isclassified],
                      val_predicted_classes[isclassified],average="macro")

@show val_jaccard_score_macro

val_f1_score = metrics.f1_score(val_classes[isclassified],
                 val_predicted_classes[isclassified],average="macro")

@show val_f1_score 
val_jaccard_score_nothing =  metrics.jaccard_score(val_classes[isclassified],
                      val_predicted_classes[isclassified],average=nothing)

@show val_jaccard_score_nothing 
####TEST DATA####
isclassified = test_classes .!== nclasses;
test_jaccard_score_macro = metrics.jaccard_score(test_classes[isclassified],
                      test_predicted_classes[isclassified],average="macro")

@show test_jaccard_score_macro

test_f1_score = metrics.f1_score(test_classes[isclassified],
                 test_predicted_classes[isclassified],average="macro")

@show test_f1_score

test_jaccard_score_nothing =  metrics.jaccard_score(test_classes[isclassified],
                      test_predicted_classes[isclassified],average=nothing)

@show test_jaccard_score_nothing

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
using NCDatasets
ds = NCDataset(fname,"c")
defVar(ds,"val_predicted_classes",val_predicted_classes,("x","y","val_sample"))
defVar(ds,"test_predicted_classes",test_predicted_classes,("x","y","test_sample"))

defVar(ds,"val_classes",val_classes,("x","y","val_sample"))
defVar(ds,"test_classes",test_classes,("x","y","test_sample"))

defVar(ds,"val_X",val_X,("val_sample",))
defVar(ds,"test_X",test_X,("test_sample",))

close(ds)
