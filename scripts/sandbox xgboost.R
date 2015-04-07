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
data.train  <- read.csv("./data/train.csv.gz",header=TRUE)[,-1]   
# data.test   <- read.csv("./data/test.csv.gz",header=TRUE)[,-1]   

# 4. Split the data set
Ncol <- ncol(data.train)
ind.train <- createDataPartition(y=data.train[,'target'], p=0.7, list=FALSE,
                                 groups=nlevels(data.train[,'target']))
batch.train <- data.train[ind.train,]
batch.valid <- data.train[-ind.train,]
# Since the data set is "big", we sample the training for grid search purposes
ind.cv <- createDataPartition(y=batch.train[,'target'], p=0.2, list=FALSE,
                                 groups=nlevels(data.train[,'target']))
batch.cv <- batch.train[ind.cv,]

dtrain <- x2xgb.DMatrix(batch.train[,-Ncol], batch.train[,Ncol])
dcv    <- x2xgb.DMatrix(batch.cv[,-Ncol], batch.cv[,Ncol])
dvalid <- x2xgb.DMatrix(batch.valid[,-Ncol], batch.valid[,Ncol])

# dtest  <- x2xgb.DMatrix(data.test)


## -------------------------------------------------------------------------- ##
## Grid Search
## -------------------------------------------------------------------------- ##
tuneGrid <- expand.grid(        
        #Parameter for Tree Booster
        "eta"=0.1,
        "max.depth"=c(10,25),
        "subsample"=0.5,
        "colsample_bytree"=1,
        #Task Parameters      
        "objective" = "multi:softprob",              
        "num_class" = 9,
        "eval_metric" = "mlogloss",
        stringsAsFactors = FALSE)
nround = 50 # the max number of iterations
results <- data.frame()
time.start <- Sys.time()
for (t in 1:dim(tuneGrid)[1]){ 
        set.seed(90210)
        cat('## -------------------------------------------------------------------------- ## \n')
        cat('Grid Search #',t,'out of',dim(tuneGrid)[1],'\n')
        bst.cv <- xgb.cv(params=as.list(tuneGrid[t,]), data = dcv,
                         nrounds = nround, nfold = 3, showsd = TRUE,
                         verbose=TRUE) 
        res.cv <- as.data.frame(bst.cv)
        per <- cbind(res.cv[nround,c("test.mlogloss.mean","test.mlogloss.std")],
                     tuneGrid[t,])
        results <- rbind(results,per)
        
        time.now <- Sys.time(); time.diff <- difftime(time.now,time.start,units="secs")
        cat('Estimated time to complete:',
            round((time.diff/t)*(dim(tuneGrid)[1]-t)),'[sec] \n')
}
colnames(results)[1:2] <- c("logloss.mean","logloss.std")
rownames(results) <- NULL
results <- arrange(results, logloss.mean)
results

## -------------------------------------------------------------------------- ##
## Simple Cross Validation
## -------------------------------------------------------------------------- ##
# The cross validation function is might be usual to determine the number of
# iteration to apply

# A. Set necessary parameter
param <- as.list(results[1,-2:-1], stringsAsFactors = FALSE)

# B. Run Cross Valication
nround = 120 #the max number of iterations
bst.cv = xgb.cv(params=param, data = dtrain, nrounds = nround,
                nfold = 5, showsd = TRUE, verbose=TRUE)
tail(bst.cv,1)

## -------------------------------------------------------------------------- ##
## Validate the Resualts
## -------------------------------------------------------------------------- ##
# # Last insanity checkpoint before submission
# # A. Train the model
# param <- as.list(results[1,-2:-1], stringsAsFactors = FALSE)
# mdl.xgb <- xgb.train(params=param,
#                      data=dtrain, "nrounds"= 120, verbose = 1)
# 
# # B. Make prediction
# require(Metrics)
# pred <- predict(mdl.xgb,dvalid)
# pred <- t(matrix(pred,9,length(pred)/9))
# actual <- model.matrix(~ batch.valid[,Ncol] - 1)
# LogLoss(actual,pred)
# 
# # pred <- apply(pred, 1, function(x) which.max(x))
