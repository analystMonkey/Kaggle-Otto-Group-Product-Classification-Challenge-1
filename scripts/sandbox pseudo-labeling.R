## -------------------------------------------------------------------------- ##
## Pseudo-labeling
##
## Here we exploite the information in the test set was by a combination of
## pseudo-labeling and knowledge distillation, this mostly had a regularizing
## effect [1].
##
## Learning schema:
## 1. Split the data into train and validation sets.
## 2. Build a model utilizing the train set
## 3. Predict the test set
## 4. Build a model utilizing the train set and test set with their pseudo
##    labels by balancing such that:
## ----------------------------------------------------
## |      Train set (67%)       | Pseudo-labeled (33%)|
## ----------------------------------------------------
## 5. Predict the test set once more
## -------------------------------------------------------------------------- ##


## -------------------------------------------------------------------------- ##
## References
## -------------------------------------------------------------------------- ##
## [1]: http://arxiv.org/abs/1503.02531
##
