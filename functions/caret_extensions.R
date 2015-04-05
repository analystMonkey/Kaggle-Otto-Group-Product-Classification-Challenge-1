head.cast_df <- function (x, n = 6L, ...)  {
        stopifnot(length(n) == 1L)
        n <- if (n < 0L) {
                max(nrow(x) + n, 0L)
        } else min(n, nrow(x))
        h <- x[seq_len(n), , drop = FALSE]
        ## fix cast_df-specific row names element
        attr(h,"rdimnames")[[1]] <- rdimnames(h)[[1]][seq_len(n),,drop=FALSE]
        h
}
## -------------------------------------------------------------------------- ##
LogLoss <- function(actual, pred, eps = 1e-15) {
        stopifnot(all(dim(actual) == dim(pred)))
        pred[pred < eps] <- eps
        pred[pred > 1 - eps] <- 1 - eps
        -sum(actual * log(pred)) / nrow(pred) 
}
## -------------------------------------------------------------------------- ##
## Multi-Class Summary Function
## http://www.r-bloggers.com/error-metrics-for-multi-class-problems-in-r-beyond-accuracy-and-kappa/
## -------------------------------------------------------------------------- ##
require(compiler)
multiClassSummary <- cmpfun(function (data, lev = NULL, model = NULL){
        if (is.character(data$obs)) data$obs <- factor(data$obs, levels = lev)
        pred <- data[, "pred"]
        obs <- data[, "obs"]
        isNA <- is.na(pred)
        pred <- pred[!isNA]
        obs <- obs[!isNA]
        data <- data[!isNA, ]
        cls <- levels(obs)
        pred <- factor(pred, levels = levels(obs))
        require("e1071")
        out <- unlist(e1071::classAgreement(table(obs, pred)))[c("diag","kappa")]         
        probs <- data[, cls]
        actual <- model.matrix(~ obs - 1)
        out2 <- LogLoss(actual = actual, pred = probs)
        out <- c(out, out2)
        names(out) <- c("Accuracy", "Kappa", "LogLoss")
        
        if (any(is.nan(out))) out[is.nan(out)] <- NA 
        
        return(out)           
})

# ## -------------------------------------------------------------------------- ##
# ## xgboost Model
# ## -------------------------------------------------------------------------- ##
# ## https://github.com/topepo/caret/tree/master/models/files
# xgboost <- function(){
#         x2xgb.DMatrix <- function(x,y=NULL){
#                 # Convert x to numeric
#                 x <- as.matrix(x)
#                 x <- matrix(as.numeric(x),nrow(x),ncol(x))
#                 # Convert y to integer
#                 if(is.null(y)){
#                         # Create xgb.DMatrix object
#                         d <- xgb.DMatrix(x)
#                 } else {
#                         if(class(y)=="character") y <- as.factor(y)
#                         y <- as.numeric(y)-1 #xgboost take features in [0,numOfClass)
#                         # Create xgb.DMatrix object
#                         d <- xgb.DMatrix(x, label = y)   
#                 }
#                 return(d)
#         }
#         # XGBoost 
#         require(methods)
#         ## Model Components
#         xgboost <- list(label = "eXtreme Gradient Boosting",
#                         type = c("Classification", "Regression"),
#                         loop = NULL,
#                         library = "xgboost")
#         ## The parameters Element
#         xgboost$parameters <- data.frame(parameter = c("objective",
#                                                        "eval_metric",
#                                                        "eta",
#                                                        "max.depth",
#                                                        "subsample",
#                                                        "colsample_bytree",
#                                                        "nrounds",
#                                                        "verbose"),
#                                          class = c(rep("character",2),rep("numeric",6)),
#                                          label = c("Learning Task and Objective",
#                                                    "Evaluation Metric",
#                                                    "Shrinkage",
#                                                    "Max Tree Depth",
#                                                    "Subsample Ratio of Rows",
#                                                    "Subsample Ratio of Columns",
#                                                    "Max Number of Iterations",
#                                                    "Verbose"))
#         ## The grid Element
#         xgboost$grid <- function(x, y, len = NULL) {
#                 ## If no grid parameters are entered, we use these:
#                 # https://github.com/dmlc/xgboost/wiki/Parameters
#                 expand.grid(
#                         # Parameter for Tree Booster 
#                         "eta"=0.1,
#                         "max.depth"=8,
#                         "subsample"=0.5,
#                         "colsample_bytree"=0.5,
#                         # Task Parameters    
#                         "objective"="multi:softprob",
#                         "eval_metric"="mlogloss", # Multiclass logloss
#                         # Other Arguments
#                         "nrounds"=120,
#                         "verbose"=1)
#         }
#         ## The fit Element
#         xgboost$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
#                 dtrain <- x2xgb.DMatrix(x,y)
#                 params <- list(
#                         # Parameter for Tree Booster
#                         eta=param$eta,
#                         max.depth=param$max.depth, 
#                         subsample=param$subsample,
#                         colsample_bytree=param$colsample_bytree,
#                         # Task Parameters
#                         objective=param$objective,
#                         num_class = 9)
#                 xgb.train(params = params, data = dtrain,
#                           nrounds = param$nrounds, verbose = param$verbose, ...)
#         }
#         ## The predict Element
#         xgboost$predict <- function(predict, newdata){
#                 dtest <- x2xgb.DMatrix(newdata)
#                 # Make prediction
#                 pred <- predict(modelFit, dtest)
#                 if (sum(abs(pred-round(pred,0)))==0){
#                         # Option 1: Multiclass classification
#                         pred <- pred+1
#                 } else {
#                         # Option 2: The result contains predicted probability  
#                         pred <- t(matrix(pred,9,length(pred)/9))
#                         pred <- apply(pred, 1, function(x) which.max(x))    
#                 }            
#                 return(pred)
#         }
#         ## The prob Element
#         xgboost$prob <- function(predict, newdata){
#                 dtest <- x2xgb.DMatrix(newdata)
#                 # Make prediction
#                 pred <- predict(modelFit, dtest)
#                 if (sum(abs(pred-round(pred,0)))==0){
#                         # Option 1: Multiclass classification
#                         pred <- pred+1
#                 } else {
#                         # Option 2: The result contains predicted probability        
#                         pred <- t(matrix(pred,9,length(pred)/9))
#                 }            
#                 return(pred)
#         }
#         ## The sort Element
#         xgboost$sort <- function(x) x[order(x$nrounds), ]        
#         return(xgboost)
# }


