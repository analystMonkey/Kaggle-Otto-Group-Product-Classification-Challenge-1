# http://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/12947/achieve-0-50776-on-the-leaderboard-in-a-minute-with-xgboost
# https://github.com/dmlc/xgboost/wiki/Parameters

cat("\014"); rm(list = ls())
set.seed(2030)
# 1. Load the required packages for the project
source("lib/load_libraries.R")
## Register the parallel backend
registerDoParallel(makeCluster(detectCores()))

# 2. Load package files
sapply(list.files(pattern="[.]R$", path="./functions/", full.names=TRUE), source)

# 3. Load data files
train  <- read.csv("./data/train.csv.gz",header=TRUE)[,-1]   
test   <- read.csv("./data/test.csv.gz",header=TRUE)[,-1]   

dtrain <- x2xgb.DMatrix(train[,-ncol(train)],train[,ncol(train)])
dtest  <- x2xgb.DMatrix(test)

# 4. Set necessary parameter
param <- list(
        #Parameter for Tree Booster
        "eta"=0.1,
        "max.depth"=8,
        "subsample"=0.5,
        "colsample_bytree"=0.5,
        #Task Parameters      
        "objective" = "multi:softprob",              
        "num_class" = 9,
        "eval_metric" = "mlogloss")

# Run Cross Valication
# cv.nround = 50
# bst.cv = xgb.cv(param=param, data = x[trind,], label = y, 
#                 nfold = 3, nrounds=cv.nround)

# Train the model
mdl.xgb <- xgb.train(params=param, data=dtrain, "nrounds"= 50, verbose = 1)
# mdl.xgb

# Make prediction
pred <- predict(mdl.xgb,dtest)
pred <- t(matrix(pred,9,length(pred)/9))
pred <- apply(pred, 1, function(x) which.max(x))

