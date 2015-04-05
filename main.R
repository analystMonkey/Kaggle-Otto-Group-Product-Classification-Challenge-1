cat("\014"); rm(list = ls(all = TRUE)); gc(reset=TRUE)
set.seed(2030)
# options(error=recover)

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
ind.train <- createDataPartition(y=data.train[,'target'], p=0.7, list=FALSE,
                                 groups=nlevels(data.train[,'target']))
batch.train <- data.train[ind.train,]
batch.valid <- data.train[-ind.train,]




