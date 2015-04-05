cat("\014"); rm(list = ls(all = TRUE)); gc(reset=TRUE)
set.seed(2030)

# 1. Load the required packages for the project
source("lib/load_libraries.R")
## Register the parallel backend
registerDoParallel(makeCluster(detectCores()))

# 2. Load package files
sapply(list.files(pattern="[.]R$", path="./functions/", full.names=TRUE), source)

# 3. Load data set
data.train <- read.csv("./data/train.csv.gz")[,-1]   
data.test  <- read.csv("./data/test.csv.gz")[,-1]    
sampleSubmission  <- read.csv("./submissions/sampleSubmission.csv.gz")   

# 4. Split the data set by stratified sampling
ind.train <- createDataPartition(y=data.train[,'target'], p=0.1, list=FALSE,
                                 groups=nlevels(data.train[,'target']))
batch.train <- data.train[ind.train,]
batch.valid <- data.train[-ind.train,]

# 5. Train model
fitControl <- trainControl(method = "cv", number = 5,
                           classProbs = TRUE,
                           summaryFunction = multiClassSummary,
                           allowParallel = F, verboseIter = TRUE,
                           returnData = FALSE) # saves memory

mdl.rft1 <- train(target ~., data=batch.train,
                  method = "parRF", 
                  metric = "LogLoss",
                  maximize = FALSE,
#                   distribution = "multinomial",                
                  trControl = fitControl,
                  tuneGrid = expand.grid(mtry=c(25))) # Recommended: sqrt(#predictor)
mdl.rft1


# plot(mdl.rft1)
# 6. Show model performance
require(Metrics)
predicted <- predict(mdl.rft1, batch.valid, type = "prob")
# Create a design matrix (withput intercept)
obs <- batch.valid[,"target"]
actual <- model.matrix(~ obs - 1)

LogLoss <- function(actual, pred, eps = 1e-15) {
        stopifnot(all(dim(actual) == dim(pred)))
        pred[pred < eps] <- eps
        pred[pred > 1 - eps] <- 1 - eps
        -sum(actual * log(pred)) / nrow(pred) 
}

LogLoss(actual,predicted)


# dim(actual)
# dim(predicted)

# 7. Make sure all predictions are in [0,1]
