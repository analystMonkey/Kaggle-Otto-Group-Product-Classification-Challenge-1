######################################################################
## Beating the Benchmark with H2O - LB Score: 0.50103
## R script by Arno Candel @ArnoCandel
## https://www.kaggle.com/users/234686/arno-candel
## More information at http://h2o.ai/
## Source code: http://github.com/h2oai/h2o-dev/
######################################################################
cat("\014"); rm(list = ls(all = TRUE)); gc(reset=TRUE)
# Load the required packages for the project
source("lib/load_libraries.R")
# Load package files
sapply(list.files(pattern="[.]R$", path="./functions/", full.names=TRUE), source)


######################################################################
## Step 1 - Download and Install H2O
######################################################################

# # The following two commands remove any previously installed H2O packages for R.
# if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
# if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }
# 
# # Next, we download packages that H2O depends on.
# if (! ("methods" %in% rownames(installed.packages()))) { install.packages("methods") }
# if (! ("statmod" %in% rownames(installed.packages()))) { install.packages("statmod") }
# if (! ("stats" %in% rownames(installed.packages()))) { install.packages("stats") }
# if (! ("graphics" %in% rownames(installed.packages()))) { install.packages("graphics") }
# if (! ("RCurl" %in% rownames(installed.packages()))) { install.packages("RCurl") }
# if (! ("rjson" %in% rownames(installed.packages()))) { install.packages("rjson") }
# if (! ("tools" %in% rownames(installed.packages()))) { install.packages("tools") }
# if (! ("utils" %in% rownames(installed.packages()))) { install.packages("utils") }
# 
# # Now we download, install and initialize the H2O package for R.
# install.packages("h2o", type="source", repos=(c("http://h2o-release.s3.amazonaws.com/h2o-dev/master/1112/R")))


######################################################################
## Step 2 - Launch H2O
######################################################################

## Load h2o R module
library(h2o)

## Launch h2o on localhost, using all cores
h2oServer = h2o.init(max_mem_size = paste0(round(memory.limit()/1024),"g"),
                     nthreads=-1)

## Point to directory where the Kaggle data is
dir <- getwd()

## For Spark/Hadoop/YARN/Standalone operation on a cluster, follow instructions on http://h2o.ai/download/
## Then connect to any cluster node from R

#h2oServer = h2o.init(ip="mr-0xd1",port=53322)
#dir <- "hdfs://mr-0xd6/users/arno/h2o-kaggle/otto/"


######################################################################
## Step 3 - Import Data and create Train/Validation Splits
######################################################################

train.hex <- h2o.importFile(paste0(dir,"/data/train.csv.gz"), key="train.hex")
test.hex  <- h2o.importFile(paste0(dir,"/data/test.csv.gz"), key="test.hex")
dim(train.hex)
summary(train.hex)

predictors <- 2:(ncol(train.hex)-1) #ignore first column 'id'
response   <- ncol(train.hex)

## Split into 70/30 Train/Validation
rnd <- h2o.runif(train.hex, 1234)
batch_train60.hex <- h2o.assign(train.hex[rnd<0.6,], "batch_train60.hex") # 60%
batch_train80.hex <- h2o.assign(train.hex[rnd<0.8,], "batch_train80.hex") # 80%
batch_valid.hex   <- h2o.assign(train.hex[0.6<=rnd & rnd<0.8,], "batch_valid.hex")
batch_holdout.hex <- h2o.assign(train.hex[0.8<=rnd,], "batch_holdout.hex")


######################################################################
## Step 4 - Use H2O Flow to inspect the data and build some models on 
## batch_train60.hex/batch_valid.hex to get a feeling for the problem
######################################################################

## Connect browser to http://localhost:54321 (or http://cluster-node-ip:port)


######################################################################
## Step 5 - DNN Hyper-Parameter Tuning
######################################################################
# http://cran.r-project.org/web/packages/h2o/h2o.pdf
#Create a set of network topologies
tuneGrid <- expand.grid("activation"=c("RectifierWithDropout"),#"MaxoutWithDropout","TanhWithDropout"),
                        "hidden"=list(c(50,50),c(100,100)),
                        "epochs"=c(1e2),
                        "input_dropout_ratio"=c(0,0.2,0.4),
                        "hidden_dropout_ratios"=list(rep(0,1),rep(0.5,1)),
                        "sparse"=c(FALSE,TRUE),
                        stringsAsFactors = FALSE)
tuneGrid

model_grid <- c()
model_per  <- data.frame()
for (k in 1:dim(tuneGrid)[1]){
        time.start <- Sys.time()

        cat('## -------------------------------------------------------------------------- ##\n')
        cat('Training model',k,'from',dim(tuneGrid)[1],'\n')
        model <- h2o.deeplearning(x=predictors, 
                                       y=response, 
                                       training_frame=batch_train60.hex,
                                       validation_frame=batch_valid.hex,
                                       destination_key="DNN_Tuning",
                                       override_with_best_model=F,
                                       activation=tuneGrid[k,"activation"],       #*
                                       hidden=unlist(tuneGrid[k,"hidden"]),
                                       autoencoder=FALSE,             #*
                                       use_all_factor_levels=TRUE,
                                       epochs=tuneGrid[k,"epochs"],
                                       train_samples_per_iteration=-1,
                                       adaptive_rate=TRUE, rho=0.99 ,epsilon=1e-8,
                                       input_dropout_ratio=tuneGrid[k,"input_dropout_ratio"],
                                       hidden_dropout_ratios=unlist(tuneGrid[k,"hidden_dropout_ratios"]),
                                       loss="CrossEntropy",
                                       classification_stop = -1,
                                       quiet_mode = FALSE,
                                       balance_classes = TRUE,
                                       sparse = tuneGrid[k,"sparse"])
        time.now <- Sys.time();
        ## Training set performance metrics
        train_perf <- h2o.performance(model, batch_train60.hex)
        train_perf@metrics$cm$table
        train_perf@metrics$logloss
        
        ## Validation set performance metrics
        valid_perf <- h2o.performance(model, batch_valid.hex)
        valid_perf@metrics$cm$table
        valid_perf@metrics$logloss
        
        # Save model performance and arguments
        per <- cbind(round(difftime(time.now,time.start,units="mins"),1),
                     round(valid_perf@metrics$logloss,3),
                     round(train_perf@metrics$logloss,3),
                     tuneGrid[k,])        
        model_per <- rbind(model_per,per)
        cat('Done in',unlist(per[1]),'[mins].',
            'Valid/Train Error:',unlist(per[2]),'/',unlist(per[3]),'\n')
        # Save model
        model_grid <- c(model_grid,model)
}
colnames(model_per)[1:3] <- c("Training.Duration","Out.Sample.Err","In.Sample.Err")
rownames(model_per) <- NULL
model_per <- arrange(model_per, Out.Sample.Err)
model_per
## Export to Excel
# creating work book
wb <- createWorkbook()
# add the data to the new sheet
sheet1 <- createSheet(wb, sheetName="Summary")
# add the data to the new sheet
addDataFrame(model_per, sheet1, col.names=TRUE, row.names=FALSE)
# saving the workbook
saveWorkbook(wb, paste0("./data/DNNGridSearch",Sys.Date(),".xlsx"))



## Print the Model Summary
# print(unlist(model@model[[1]]@model$params))   
# model_grid@model$scoring_history

#print out a *short* summary of each of the models (indexed by parameter)
# model_grid@sumtable
#print out *full* summary of each of the models
# all_params = lapply(model_grid@model, function(x) { x@model$params })
# all_params


## Training set performance metrics
# train_perf <- h2o.performance(model, batch_train60.hex)
# train_perf@metrics$cm$table
# train_perf@metrics$logloss

## Validation set performance metrics
# valid_perf <- h2o.performance(model, batch_valid.hex)
# valid_perf@metrics$cm$table
# valid_perf@metrics$logloss

######################################################################
## Step 6 - Sanity Testing
######################################################################
# model <- h2o.gbm(x=predictors, 
#                  y=response,
#                  destination_key="Sanity_Check",
#                  training_frame=batch_train80.hex, 
#                  loss="multinomial",
#                  ntrees=42,
#                  max_depth=10, 
#                  min_rows=10,
#                  learn_rate=0.175)
# pred   <- as.matrix(predict(model_grid, batch_holdout.hex)[,-1])
# actual <- as.matrix(batch_holdout.hex[,ncol(batch_holdout.hex)])
# actual <- model.matrix(~ actual - 1)
# LogLoss(actual,pred)


# ######################################################################
# ## Step 7 - Build Final Model using the Full Training Data
# ######################################################################
# 
# model <- h2o.gbm(x=predictors, 
#                  y=response,
#                  destination_key="final_model",
#                  training_frame=train.hex, 
#                  loss="multinomial",
#                  ntrees=42,
#                  max_depth=10, 
#                  min_rows=10,
#                  learn_rate=0.175)
# 
#
# ######################################################################
# ## Step 8 - Make Final Test Set Predictions for Submission
# ######################################################################
# 
# ## Predictions: label + 9 per-class probabilities
# pred <- predict(model, test.hex)
# head(pred)
# 
# ## Remove label
# pred <- pred[,-1]
# head(pred)
# 
# ## Paste the ids (first col of test set) together with the predictions
# submission <- h2o.cbind(test.hex[,1], pred)
# head(submission)
# 
# ## Save submission to disk
# con <- gzfile(paste0("./data/submission",Sys.Date(),".csv.gz"))
# write.csv(as.matrix(submission), con, row.names=FALSE)
# unlink(con)
# 
h2o.shutdown(h2oServer, prompt = TRUE)