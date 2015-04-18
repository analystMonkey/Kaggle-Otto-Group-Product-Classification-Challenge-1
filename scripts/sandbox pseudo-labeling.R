## -------------------------------------------------------------------------- ##
## Pseudo-labeling
##
## Here we exploite the information in the test set was by a combination of
## pseudo-labeling and knowledge distillation, this mostly had a regularizing
## effect [1].
##
## Learning schema:
## 1. Split the data into train and validation sets.
## 2. Build a model utilizing the train set
## 3. Predict the test set
## 4. Build a model utilizing the train set and test set with their pseudo
##    labels by balancing such that:
## ----------------------------------------------------
## |      Train set (67%)       | Pseudo-labeled (33%)|
## ----------------------------------------------------
## 5. Predict the test set once more
## -------------------------------------------------------------------------- ##
## Initialization
## -------------------------------------------------------------------------- ##
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

## -------------------------------------------------------------------------- ##
## Step 1: Build the best model (~0.462)
## -------------------------------------------------------------------------- ##
watchlist <- list(eval = dvalid, train = dtrain)
param1 <- list(
        #Parameter for Tree Booster
        "eta"=0.1, "max.depth"=2, "subsample"=0.5, "colsample_bytree"=0.5,
        #Task Parameters
        "objective"="multi:softprob", "num_class"=9, "eval_metric"="mlogloss")
# B. Train xgboost for 200 rounds
nround <- 500
bst.1 <- xgb.train(params=param1, data=dtrain, nrounds=nround, watchlist=watchlist)
## -------------------------------------------------------------------------- ##
## Step 2: Predict the test set (~0.462); hard prediction
## -------------------------------------------------------------------------- ##
## Get hard prediction
ptrain <- predict(bst.1,dtrain)
ptrain <- t(matrix(ptrain,9,length(ptrain)/9))
ptrain <- as.factor(apply(ptrain, 1, function(x) which.max(x)))
levels(ptrain) <- levels(data.train[,Ncol])

pvalid <- predict(bst.1,dvalid)
pvalid <- t(matrix(pvalid,9,length(pvalid)/9))
pvalid <- as.factor(apply(pvalid, 1, function(x) which.max(x)))
levels(pvalid) <- levels(data.train[,Ncol])

pholdout <- predict(bst.1,dholdout)
pholdout <- t(matrix(pholdout,9,length(pholdout)/9))
pholdout <- as.factor(apply(pholdout, 1, function(x) which.max(x)))
levels(pholdout) <- levels(data.train[,Ncol])
## Merge the prediction to the original data sets
atrain   <- data.train[ind<0.7,Ncol]
avalid   <- data.train[0.7<=ind & ind<0.8,Ncol]
aholdout <- data.train[0.8<=ind,Ncol]
batch.train   <- cbind(data.train[ind<0.7,-Ncol],            pseudo=as.numeric(ptrain)-1,   target=atrain)
batch.valid   <- cbind(data.train[0.7<=ind & ind<0.8,-Ncol], pseudo=as.numeric(pvalid)-1,   target=avalid)
batch.holdout <- cbind(data.train[0.8<=ind,-Ncol],           pseudo=as.numeric(pholdout)-1, target=aholdout)
## Merge the train and test sets
Ncol2 = ncol(batch.train)
x <- rbind(batch.train[,-Ncol2],batch.valid[,-Ncol2])
y <- as.factor(c(batch.train[,Ncol2],pvalid))
levels(y) <- levels(batch.train[,Ncol2])
dtrain   <- x2xgb.DMatrix(x,y)
dvalid   <- x2xgb.DMatrix(batch.valid[,-Ncol2], batch.valid[,Ncol2])
dholdout <- x2xgb.DMatrix(batch.holdout[,-Ncol2], batch.holdout[,Ncol2])
## -------------------------------------------------------------------------- ##
## Step 3: Rebuild the best model with the hard predictions
## -------------------------------------------------------------------------- ##
watchlist <- list(eval = dvalid, train = dtrain)
param2 <- list(
        #Parameter for Tree Booster
        "eta"=0.1, "max.depth"=10, "subsample"=0.5, "colsample_bytree"=0.5,
        #Task Parameters
        "objective"="multi:softprob", "num_class"=9, "eval_metric"="mlogloss")
# B. Train xgboost for 200 rounds
nround <- 500
bst.2 <- xgb.train(params=param2, data=dtrain, nrounds=nround, watchlist=watchlist)

## -------------------------------------------------------------------------- ##
## Validate the Resualts
## -------------------------------------------------------------------------- ##
## Last insanity checkpoint before submission
pred   <- predict(bst.2,dholdout)
pred   <- t(matrix(pred,9,length(pred)/9))
actual <- model.matrix(~ batch.holdout[,Ncol2] - 1)
LogLoss(actual,pred)

## -------------------------------------------------------------------------- ##
## References
## -------------------------------------------------------------------------- ##
## [1]: http://arxiv.org/abs/1503.02531
##
