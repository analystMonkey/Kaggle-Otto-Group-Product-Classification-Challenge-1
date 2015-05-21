cat("\014"); rm(list = ls())
################################################################################
## T-SNE and meta bagging - LB Score: 0.41599
## R script by Mike Kim
## https://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/14295/41599-via-tsne-meta-bagging
################################################################################
if(!require("xgboost")) {
        install.packages("devtools")
        devtools::install_github('dmlc/xgboost',subdir='R-package')
}
library(methods)
library(randomForest)
library(Rtsne)
library(data.table)
options(scipen=999)
set.seed(1004)
## 1. Load data files
train <- read.csv("./data/train.csv.gz",header=TRUE)[,-1]   
test  <- read.csv("./data/test.csv.gz",header=TRUE)[,-1]   
cat('train/test set are',paste0(round(object.size(train)/2^20),'Mb','/',round(object.size(test)/2^20),'Mb'))

y = train[,ncol(train)]
y = gsub('Class_','',y)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass)

x = rbind(train[,-ncol(train)],test)
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))

x = 1/(1+exp(-sqrt(x)))

tsne <- Rtsne(as.matrix(x), check_duplicates = FALSE, pca = FALSE, 
              perplexity=30, theta=0.5, dims=2)

x = round(x,3)

##test here
#trind = 1:length(y)
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
              "nthread" = 7)

#bst.cv = xgb.cv(param=param, data = x[trind,], label = y, 
#                nfold = 4, 
#                column_subsample = 0.8, #subsample changes only hurt.
#                nrounds=60, max.depth=11, eta=0.46, min_child_weight=10)

#[59]  train-mlogloss:0.204678+0.004478        test-mlogloss:0.488262+0.011556 no add
#[55]  train-mlogloss:0.190645+0.003466	test-mlogloss:0.479680+0.010798 added tsne 


x = cbind(x, tsne$Y[,1]) 
x = cbind(x, tsne$Y[,2])

trind = 1:length(y)
teind = (nrow(train)+1):nrow(x)

trainX = x[trind,]
testX = x[teind,]


# Set necessary parameter
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
              "nthread" = 7)


# Train the model
nround = 1200
bst = xgboost(param=param, data = trainX, label = y,
              nrounds=nround, max.depth=8, eta=0.03, min_child_weight=3)

# Make prediction
pred = predict(bst,testX)
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)


tmpC = 1:240
tmpL = length(trind)
gtree = 200
for (z in tmpC) {
        print(z)
        tmpS1 = sample(trind,size=tmpL,replace=T)
        tmpS2 = setdiff(trind,tmpS1)
        
        tmpX2 = trainX[tmpS2,]
        tmpY2 = y[tmpS2]
        
        cst = randomForest(x=tmpX2, y=as.factor(tmpY2), replace=F, ntree=100, do.trace=T, mtry=7)
        
        tmpX1 = trainX[tmpS1,]
        tmpY1 = y[tmpS1]
        
        tmpX2 = predict(cst, tmpX1, type="prob")
        tmpX3 = predict(cst, testX, type="prob")
        
        bst = xgboost(param=param, data = cbind(tmpX1,tmpX2), label = tmpY1, column_subsample = 0.8, 
                      nrounds=60, max.depth=11, eta=0.46, min_child_weight=10) 
        
        # Make prediction
        pred0 = predict(bst,cbind(testX,tmpX3))
        pred0 = matrix(pred0,9,length(pred0)/9)
        pred = pred + t(pred0)
}
pred = pred/(z+1)

## Save submission to disk
pred <- data.frame(1:nrow(pred),pred)
colnames(pred) <- c("id",paste0("Class_",1:9)) 
con  <- gzfile(paste0("./data/submission",Sys.Date(),'T-SNE',".csv.gz"))
write.csv(pred, con, row.names=FALSE)
unlink(con)