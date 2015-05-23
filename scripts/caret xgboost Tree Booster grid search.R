################################################################################
## Caret XGBoost Tree Booster - Grid Search
## https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
## https://github.com/topepo/caret/issues/147
## Step 1: Split the data into 10%/70%/20% for grid-search/validation/holdout
## Step 2: Perform grid serach on 10% of the data
################################################################################
cat("\014"); rm(list = ls(all = TRUE)); gc(reset=TRUE)
set.seed(2030)
## 1. Load the required packages for the project
require("caret")
if(!require("xgboost")) devtools::install_github('dmlc/xgboost',subdir='R-package')
require("doParallel")
require("pROC")
## 2. Load and split the data files
train <- read.csv("./data/train.csv.gz",header=TRUE)[,-1]   
test  <- read.csv("./data/test.csv.gz",header=TRUE)[,-1]
u     <- runif(nrow(train))
X_gs  <- train[u<=0.1,]         # grid-search            
X_va  <- train[0.1<u & u<=0.8,] # validation
X_ho  <- train[0.8<u,]         # holdout
## 3. Perform Grid Search
### Register the parallel backend
cl <- makeCluster(detectCores(),outfile="")
registerDoParallel(cl)
### Setup the computational nuances of the model training phase
fitControl <- trainControl(## 10-fold CV
        method = "cv",
        number = 10,
        classProbs = TRUE,
        summaryFunction = mnLogLoss,
        verboseIter = TRUE,
        allowParallel = TRUE,
        returnData = FALSE)      # saves memory
### Grid Search - Tree Booster
XGBoostTreeBoosterGrid <- expand.grid(
        # Number of Boosting Iterations
        nrounds = 1e2,
        # Max Tree Depth
        max_depth = (3:6)*2,
        # Shrinkage
        eta = exp(seq(log(1e-3),log(1e-1),length.out=3)))

mod1 <- train(target ~ ., data = X_gs, 
              type = 'Classification',
              objective = 'multi:softprob',
              method = 'xgbTree',
              metric = 'logLoss', maximize = FALSE,
              tuneGrid = XGBoostTreeBoosterGrid,
              trControl = fitControl)
mod1

## 4. Extract Best Parameters
XGBoostTreeBoosterGrid <- data.frame(nrounds=mod1$finalModel$tuneValue[1],
                                     max_depth=mod1$finalModel$tuneValue[2],
                                     eta=mod1$finalModel$tuneValue[3])


# Stop parallel cluster
stopCluster(cl)
