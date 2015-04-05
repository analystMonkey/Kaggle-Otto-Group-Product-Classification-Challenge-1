cat("\014"); rm(list = ls())
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
sampleSubmission  <- read.csv("./data/sampleSubmission.csv.gz")   

# 4. Split the data set by stratified sampling
ind.train <- createDataPartition(y=data.train[,'target'], p=0.7, list=FALSE,
                                 groups=nlevels(data.train[,'target']))
batch.train <- data.train[ind.train,]
batch.valid <- data.train[-ind.train,]

## -------------------------------------------------------------------------- ## 
##                       Between-Predictor Correlation                        ##
## -------------------------------------------------------------------------- ## 
require(corrplot)
r <- cor(batch.train[,-ncol(batch.train)])
rownames(r) <- colnames(r) <- NULL
corrplot(r, method = "color" ,addgrid.col=NA, order = "AOE", mar = c(1,1,4,1),
         title="Correlation Matrix of the Unknown Features")

## -------------------------------------------------------------------------- ## 
##                                   PCA                                      ##
## -------------------------------------------------------------------------- ## 
require(scales)
require(gridExtra)

Otto.pca <- prcomp(batch.train[,-ncol(batch.train)],
                   center = TRUE, scale. = TRUE) 
summary(Otto.pca)

# A. Scree Plot
vars <- apply(Otto.pca$x, 2, var)  
props <- vars / sum(vars)
loadings <- data.frame(x=1:length(cumsum(props)), y=100*cumsum(props))
p1 <- ggplot(loadings,aes(x,y)) + geom_line() +
        coord_fixed(ratio=1) +
        labs(x = "Compnent", y = "Cumulative Percentage of Total Variance") +
        scale_x_continuous(breaks=pretty_breaks(10)) +
        scale_y_continuous(breaks=pretty_breaks(10)) +
        ggtitle("Scree Plot")

# B. PCs coefficients associated with variables in the dataset
theta <- seq(0,2*pi,length.out = 100)
circle <- data.frame(x = cos(theta), y = sin(theta))
p2 <- ggplot(circle,aes(x,y)) + geom_path()

loadings <- data.frame(Otto.pca$rotation, 
                       .names = row.names(Otto.pca$rotation))
p2 <- p2 + geom_text(data=loadings, 
              mapping=aes(x = PC1, y = PC2, label = .names, colour = .names)) +
        coord_fixed(ratio=1) +
        scale_colour_discrete(guide = FALSE) +
        labs(x = "PC1", y = "PC2") + ggtitle("Unit Circle")

# Display Plots
grid.newpage() # Open a new page on grid device
pushViewport(viewport(layout = grid.layout(1, 2)))
print(p1, vp = viewport(layout.pos.row = 1, layout.pos.col = 1))
print(p2, vp = viewport(layout.pos.row = 1, layout.pos.col = 2))












