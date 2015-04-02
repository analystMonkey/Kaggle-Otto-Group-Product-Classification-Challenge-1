## -------------------------------------------------------------------------- ##
##                           Save csv to gz file                              ##
## -------------------------------------------------------------------------- ##
# Train set
train <- read.csv("./data/train.csv",header=TRUE)
con1 <- gzfile("./data/train.csv.gz")
write.csv(train, con1, row.names=FALSE)
# Test set
test <- read.csv("./data/test.csv",header=TRUE)
con2 <- gzfile("./data/test.csv.gz")
write.csv(test, con2, row.names=FALSE)
# Tender Submission
sampleSubmission <- read.csv("./data/sampleSubmission.csv",header=TRUE)
con3 <- gzfile("./data/sampleSubmission.csv.gz")
write.csv(sampleSubmission, con3, row.names=FALSE)

## -------------------------------------------------------------------------- ##
##                          Load csv from gz file                             ##
## -------------------------------------------------------------------------- ##
train <- read.csv("./data/train.csv.gz")   
test  <- read.csv("./data/test.csv.gz")   
sampleSubmission  <- read.csv("./data/sampleSubmission.csv.gz")   





