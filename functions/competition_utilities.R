## -------------------------------------------------------------------------- ##
##                   Create submission form data frame                        ##
## -------------------------------------------------------------------------- ##

# pred <- as.factor(pred+1)
# levels(pred) <- paste0('Class_',levels(pred))

# Output submission
# pred = matrix(pred,9,length(pred)/9)
# pred = t(pred)
# pred = format(pred, digits=2,scientific=F) # shrink the size of submission
# pred = data.frame(1:nrow(pred),pred)
# names(pred) = c('id', paste0('Class_',1:9))
# 
# con <- gzfile(paste0('submissions/submission ',Sys.Date(),'.csv.gz'))
# write.csv(pred, con, row.names=FALSE, quote=FALSE)
## -------------------------------------------------------------------------- ##

