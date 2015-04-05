## -------------------------------------------------------------------------- ##
##                             Utility Functions                              ##
## -------------------------------------------------------------------------- ##
x2xgb.DMatrix <- function(x,y=NULL){
        # Convert x to numeric
        x <- as.matrix(x)
        x <- matrix(as.numeric(x),nrow(x),ncol(x))
        # Convert y to integer
        if(is.null(y)){
                # Create xgb.DMatrix object
                d <- xgb.DMatrix(x)
        } else {
                if(class(y)=="character") y <- as.factor(y)
                y <- as.numeric(y)-1 #xgboost take features in [0,numOfClass)
                # Create xgb.DMatrix object
                d <- xgb.DMatrix(x, label = y)   
        }
        return(d)
}
## -------------------------------------------------------------------------- ##