# http://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/12947/achieve-0-50776-on-the-leaderboard-in-a-minute-with-xgboost
# https://github.com/dmlc/xgboost/wiki/Parameters

cat("\014"); rm(list = ls())
# 1. Load the required packages for the project
source("lib/load_libraries.R")
## Register the parallel backend
registerDoParallel(makeCluster(detectCores()))

# 2. Load package files
sapply(list.files(pattern="[.]R$", path="./functions/", full.names=TRUE), source)

# 3. Load data files
data.train  <- read.csv("./data/train.csv.gz",header=TRUE)[,-1]   
# data.test   <- read.csv("./data/test.csv.gz",header=TRUE)[,-1]   

# 4. Split the data set
## ----------------------------------------------------------------------------                         
## |  Train set  70%                 | Validation 10% |      Holdout 20%       |
## ----------------------------------------------------------------------------
set.seed(2030)
Ncol <- ncol(data.train)
ind  <- runif(nrow(data.train))
batch.train   <- data.train[ind<0.7,]
batch.valid   <- data.train[0.7<=ind & ind<0.8,]
batch.holdout <- data.train[0.8<=ind,]

dtrain   <- x2xgb.DMatrix(batch.train[,-Ncol],   batch.train[,Ncol])
dvalid   <- x2xgb.DMatrix(batch.valid[,-Ncol],   batch.valid[,Ncol])
dholdout <- x2xgb.DMatrix(batch.holdout[,-Ncol], batch.holdout[,Ncol])

# 5. Initialization
watchlist <- list(eval = dvalid, train = dtrain)


cat('
## -------------------------------------------------------------------------- ##
## Stage 1
## -------------------------------------------------------------------------- ##')
# A. Set necessary parameter
param1 <- list(
        #Parameter for Tree Booster
        "eta"=0.1, "max.depth"=10, "subsample"=0.5, "colsample_bytree"=0.5,
        #Task Parameters
        "objective"="multi:softprob", "num_class"=9, "eval_metric"="mlogloss")
# B. Train xgboost for 120 rounds
nround <- 200 
bst.1 <- xgb.train(params=param1, data=dtrain, nrounds=nround, watchlist=watchlist)

cat('
## -------------------------------------------------------------------------- ##
## Stage 2
## -------------------------------------------------------------------------- ##')
# A. Set necessary parameter
param2 <- list(
        #Parameter for Tree Booster
        "eta"=0.05, "max.depth"=10, "subsample"=0.5, "colsample_bytree"=0.5,
        #Task Parameters
        "objective"="multi:softprob", "num_class"=9, "eval_metric"="mlogloss")
# Note: we need the margin value instead of transformed prediction in
# set_base_margin do predict with output_margin=TRUE, will always give you
# margin values before logistic transformation
ptrain   <- predict(bst.1, dtrain,   outputmargin=TRUE)
pvalid   <- predict(bst.1, dvalid,   outputmargin=TRUE)
pholdout <- predict(bst.1, dholdout, outputmargin=TRUE)
# set the base_margin property of dtrain and dvalid
# base margin is the base prediction we will boost from
setinfo(dtrain,   "base_margin", ptrain)
setinfo(dvalid,   "base_margin", pvalid)
setinfo(dholdout, "base_margin", pholdout)

# B. Train xgboost for 120 rounds
nround <- 120 
bst.2 <- xgb.train(params=param2, data=dtrain, nrounds=nround, watchlist=watchlist)

## -------------------------------------------------------------------------- ##
## Validate the Resualts
## -------------------------------------------------------------------------- ##
## Last insanity checkpoint before submission
pred   <- predict(bst.2,dholdout)
pred   <- t(matrix(pred,9,length(pred)/9))
actual <- model.matrix(~ batch.holdout[,Ncol] - 1)
LogLoss(actual,pred)
# pred <- apply(pred, 1, function(x) which.max(x))

# require(Metrics)
# actual    <- model.matrix(~ batch.holdout[,Ncol]-1)
# predicted <- predict(bst.2,dholdout)
# logLoss(actual, predicted)
