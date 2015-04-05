## Load the required packages for the project
packages.list = c("knitr","pander","rmarkdown",      # Documentation packages
#                 "magrittr",                        # Make code readable
                  "ggplot2","gridExtra",             # Visualisation 
                  "devtools","testthat",             # Development tools for R
                  "caret","e1071",                   # Classification Algorithms
                  "kernlab","randomForest","h2o",    # SVM, RF, DNN
                  "plyr","reshape2",                 # Data Munging Tools
                  "pROC","ROCR","Metrics",           # Performance tools 
                  "doParallel","foreach",            # Parallel Tools
                  "devtools")
for (i in 1:length(packages.list)){
        if(!require(packages.list[i], character.only=TRUE)) {
                install.packages(packages.list[i])}
        require(packages.list[i], character.only = TRUE)  
}

# install the R-package of xgboost
if(!require("xgboost")) devtools::install_github('dmlc/xgboost',subdir='R-package')

rm(packages.list, i)