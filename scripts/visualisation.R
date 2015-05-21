cat("\014"); rm(list = ls()); 
set.seed(20150519)
################################################################################
##           t-distributed stochastic neighbor embedding (t-SNE)              ##
################################################################################
library(ggplot2)
library(Rtsne)
## Get the data
train <- read.csv("./data/train.csv.gz",header=TRUE)[,-1]   
## Subset the train set to hold only the numeric variables
train.numeric <- train[,-ncol(train)]
tsne <- Rtsne(as.matrix(train.numeric),
              check_duplicates = FALSE, pca = FALSE, 
              perplexity=30, theta=0.5, dims=2)
X_plot <- cbind(as.data.frame(tsne$Y), "class"=train[,ncol(train)])

p <- ggplot(X_plot, aes(x=V1, y=V2, color=class, shape=class)) +
        geom_point(size=4) +
        scale_shape_manual(values=1:9) +
        xlab("") + ylab("") +
        theme_light(base_size=20) +
        ggtitle("t-SNE Otto Group Product Classification Challenge Visualization") + 
        theme(strip.background = element_blank(),
              strip.text.x     = element_blank(),
              axis.text.x      = element_blank(),
              axis.text.y      = element_blank(),
              axis.ticks       = element_blank(),
              axis.line        = element_blank(),
              panel.border     = element_blank())
plot(p)
# ggsave("tsne.png", p, height=8, width=8, units="in")

################################################################################
##                  Non-Classical Multidimensional Scaling                    ##
################################################################################






