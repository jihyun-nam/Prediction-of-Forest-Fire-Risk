library("WeightSVM")
library("e1071")
library("mclust")
library("caret")
library("mvtnorm")

datafile_path = "army_fire_dummy.csv"


get.gmm.oversample <-function(gmc.model.pos, data.gcwsvm.pos, n.oversample){
  # generate synthetic samples from the learned GMM, and add them to the training dataset

  #1. extract parameters from the GMM result
  G <- gmc.model.pos$G; # number of clusters, determined by BIC
  d <- gmc.model.pos$d; # number of predictor variables. In our data, d=2
  prob <- gmc.model.pos$parameters$pro # learned mixing coefficients pi_1, ... pi_G
  means <- gmc.model.pos$parameters$mean # learned cluster mean vectors mu_1, ... mu_G
  vars <- gmc.model.pos$parameters$variance$sigma #learned cluter covariance matrix Sigma_1, ... Sigma_G
  data.gmm <- data.frame(matrix(NA, n.oversample, d + 1)) #initialize a matrix for storing synthetic minority  samples
  colnames(data.gmm ) <- colnames(data.gcwsvm.pos)
  
  #2. Generate synthetic minority samples from the learned Guassian mixture
  gmc.index <- sample(x = 1:G, size = n.oversample, replace = T, prob = prob) #randomly assign group, according to the learned group membership probability.
  for(i in 1 : n.oversample) {
    data.gmm [i, ] <- c(
      rmvnorm(1, mean = means[ , gmc.index[i]],sigma=vars[,,gmc.index[i]]),
      1
    )
  }
  data.gmm$IS_FIRE <- factor(data.gmm$IS_FIRE, levels = c("-1", "1")) # turn the y variable into a factor
  return(data.gmm)
} # end of  function get.gmm.oversample



#####################################
# Preliminary step
#####################################
replication <- 100

#tuning parameters range
param.set.c <- 2 ^ (-10:10)
param.set.gamma = 2 ^ (-10:10)
param.set.desired_imbalance_ratio = c(1,2,5,10, 15, 20)

# arrays for saving tuning result
tune.mat.gcwsvm <- array(0, dim = c(
  length(param.set.c),
  length(param.set.gamma),
  length(param.set.desired_imbalance_ratio)
))
tune.mat.ksvm <- array(0, dim = c(
  length(param.set.c),
  length(param.set.gamma)
))

tune.mat.lsvm <- array(0, dim = c(
  length(param.set.c)
))

result.mat.spe <- result.mat.sen <- result.mat.gme <- array(0, dim = c(replication, 3) )
# read data
dataset <- read.csv(datafile_path)
dataset$IS_FIRE <- factor(dataset$IS_FIRE, levels = c(-1, 1)) # turn the y variable into a factor
imbalance.ratio.original <-  nrow(dataset) / nrow(dataset[dataset$IS_FIRE == 1,]) -1



###################################
# Main part
###################################

################################################
# For loop level 1: monte carlo simulation. Since data is fixed, this means trying independent splits
for (rep in 1:replication){
################################################
  cat(rep, "th rep", "\n")
  set.seed(rep) # for reproducible result
  
  k.fold.test <- createFolds(dataset$IS_FIRE, k = 5, list = FALSE, returnTrain = FALSE)
  ################################################
  # For loop level 2: 5-fold CV for performance metric evaluation
  for (foldnum.test in 1:5) {
  ################################################
    cat( rep, "th rep, ", foldnum.test, "th test fold\n ", sep = "")
    set.seed(foldnum.test) # for reproducible result
    # training-test set split
    indices.train <- k.fold.test != foldnum.test
    indices.test  <- k.fold.test == foldnum.test
    data.train <- dataset[indices.train,]
    data.test  <- dataset[indices.test ,]

    k.fold.tune <- createFolds(data.train$IS_FIRE, k = 5, list = FALSE, returnTrain = FALSE)
    ################################################
    # for loop level 3: 5-fold CV for hyperparameter tuning
    for (foldnum.tune in 1:5) {
    ################################################
      cat( rep, "th rep, ", foldnum.test, "th test fold, ", foldnum.tune, "th tune fold\n", sep = "")
      
      # [tuning] training-validation set split
      data.train.tunetrain    <- data.train[k.fold.tune != foldnum.tune,]
      data.train.tunevalid    <- data.train[k.fold.tune == foldnum.tune,]
      
      # [tuning] standardization
      preProcValues <- preProcess(data.train.tunetrain[1:4],  method = c("center", "scale")) #learn standarization parameters
      data.train.tunetrain <- predict(preProcValues, data.train.tunetrain) #standardize training set
      data.train.tunevalid <- predict(preProcValues, data.train.tunevalid) #standardize validation set
      
      # [tuning] fit GMM for GC-WSVM
      data.train.tunetrain.pos <- data.train.tunetrain[data.train.tunetrain$IS_FIRE == 1,] # pick observations with positive label
      gmc.model.tune <- Mclust(data.train.tunetrain.pos[1:4], modelNames = c("EII")) #learn GMM
      
      # [tuning] tuning GC-WSVM
      for (r in 1 : length(param.set.desired_imbalance_ratio)){ #loop over desired imbalance ratio of GC-WSVM
      ################################################  
        # step 1 of GC-WSVM: generate synthetic minority samples using GMM
        oversample.ratio.now <- imbalance.ratio.original / param.set.desired_imbalance_ratio[r]  - 1
        n.oversample.now <- round(length(data.train.tunetrain.pos$IS_FIRE) * oversample.ratio.now)
        data.gcwsvm.tunetrain <- rbind(
          get.gmm.oversample(gmc.model.tune, data.train.tunetrain.pos,  n.oversample.now), ##synthetic oversampling
          data.train.tunetrain
          )
        cat("number of synthetic minority samples:", n.oversample.now, "\n")
        
        # step 2 of GC-WSVM: fit weighted Gaussian kernel SVM

        for (i in 1:length(param.set.c)){ #hyperparameter tuning: loop over C
          for (j in 1:length(param.set.gamma)){ #hyperparameter tuning: loop over gamma
            # fit weighted svm model
            model.gcwsvm.now <- wsvm(data = data.gcwsvm.tunetrain, IS_FIRE ~ .,
                weight =
                  ( 1/( nrow(data.train.tunetrain.pos) + n.oversample.now              )) * (data.gcwsvm.tunetrain$IS_FIRE ==  1) +
                  ( 1/( nrow(data.train.tunetrain)     - nrow(data.train.tunetrain.pos))) * (data.gcwsvm.tunetrain$IS_FIRE == -1),
                gamma = param.set.gamma[j],
                cost = param.set.c[i],
                kernel = "radial", scale = FALSE)
            # calculate g-mean
            svm.cmat <- table("truth" = data.train.tunevalid$IS_FIRE, "pred.gcwsvm" = predict(model.gcwsvm.now, data.train.tunevalid[1:4]))
            sen <- svm.cmat[2, 2] / sum(svm.cmat[2, ]) # sensitivity
            spe <- svm.cmat[1, 1] / sum(svm.cmat[1, ]) #specificity
            gme <- sqrt(sen * spe) #G-mean
            tune.mat.gcwsvm[i,j,r] <- tune.mat.gcwsvm[i,j,r] + (gme / 5) # mean G-mean over 5 folds
          } #end of gamma loop
        }# end of c loop
      } #end of r loop
      
  
      # [tuning] tuning kernel SVM
      for (i in 1:length(param.set.c)){
        for (j in 1:length(param.set.gamma)){
          # fit weighted svm model
          model.ksvm.now <- svm(data = data.train.tunetrain, IS_FIRE ~ .,
            gamma = param.set.gamma[j],
            cost = param.set.c[i],
            kernel = "radial", scale = FALSE)
          # calculate g-mean
          svm.cmat <- table("truth" = data.train.tunevalid$IS_FIRE, "pred.ksvm" =  predict(model.ksvm.now, data.train.tunevalid[1:4]))
          sen <- svm.cmat[2, 2] / sum(svm.cmat[2, ]) # sensitivity
          spe <- svm.cmat[1, 1] / sum(svm.cmat[1, ]) # specificity
          gme <- sqrt(sen * spe) #G-mean
          tune.mat.ksvm[i,j] <- tune.mat.ksvm[i,j] + (gme / 5) # mean G-mean over 5 folds
           } #end of gamma loop
      }# end of c loop   

          
      # [tuning] tuning linear SVM
      for (i in 1:length(param.set.c)){
          # fit linear svm mode
          model.lsvm.now <- svm(
            data = data.train.tunetrain, IS_FIRE ~ .,
            gamma = param.set.gamma[j],
            cost = param.set.c[i],
            kernel = "linear", scale = FALSE
          )
          # calculate g-mean
          svm.cmat <- table("truth" = data.train.tunevalid$IS_FIRE, "pred.ksvm" = predict(model.lsvm.now, data.train.tunevalid[1:4]))
          sen <- svm.cmat[2, 2] / sum(svm.cmat[2, ]) # sensitivity
          spe <- svm.cmat[1, 1] / sum(svm.cmat[1, ]) # specificity
          gme <- sqrt(sen * spe)#G-mean
          tune.mat.lsvm[i] <- tune.mat.lsvm[i] + (gme / 5) # mean G-mean over 5 folds
      }# end of c loop  
    } #end of for loop level 3 (5-fold CV for hyperparameter tuning)       
    
    ############################################################################################################
    # Now we fit the models with best hyperparameters and evaluate performance of the models.
    ############################################################################################################
    
    # standardization
    preProcValues <- preProcess(data.train[1:4],  method = c("center", "scale"))
    data.train <- predict(preProcValues, data.train)
    data.test  <- predict(preProcValues, data.test)
    
    
    # Fit best GC-WSVM
    param.best.gcwsvm <- which(tune.mat.gcwsvm == max(tune.mat.gcwsvm), arr.ind = TRUE)
    
    cat("best parameter for gcwsvm: c = ",  
        param.set.c[param.best.gcwsvm[1]],
        ", gamma = ",
        param.set.gamma[param.best.gcwsvm[2]],
        ", desired imbalance ratio = ",
        param.set.desired_imbalance_ratio[param.best.gcwsvm[3]],
        "\n"
    )
    
    oversample.ratio.best <- imbalance.ratio.original / param.set.desired_imbalance_ratio[param.best.gcwsvm[3]]  - 1
    n.oversample.best <- round(length(data.train.tunetrain.pos$IS_FIRE) * oversample.ratio.best)
    data.train.pos <- data.train[data.train$IS_FIRE == 1,] 
    gmc.model.best <- Mclust(data.train.pos[1:4], modelNames = c("EII"))  # learn GMM
    data.gcwsvm.train <- rbind(
      get.gmm.oversample(gmc.model.best, data.train.pos, n.oversample.best), #synthetic oversampling
      data.train
      )
    cat("number of synthetic minority samples:", n.oversample.best, "\n")
  
    
    model.gcwsvm.best <- wsvm(data = data.gcwsvm.train, IS_FIRE ~ .,
      weight = 
        ( 1/( nrow(data.train.pos) + n.oversample.best   )) * (data.gcwsvm.train$IS_FIRE ==  1) +
        ( 1/( nrow(data.train)     - nrow(data.train.pos))) * (data.gcwsvm.train$IS_FIRE == -1) ,
      gamma = param.set.gamma[param.best.gcwsvm[2]],
      cost = param.set.c[param.best.gcwsvm[1]],
      kernel = "radial",
      scale = FALSE
    )
    
    cmat.gcwsvm <- table("truth" = data.test$IS_FIRE, "pred.gcwsvm" = predict(model.gcwsvm.best, data.test[1:4]))
    sen <- cmat.gcwsvm[2, 2] / sum(cmat.gcwsvm[2, ]) # sensitivity
    spe <- cmat.gcwsvm[1, 1] / sum(cmat.gcwsvm[1, ]) # specificity
    gme <- sqrt(sen * spe) # G-mean
    result.mat.spe[rep,1] <-  result.mat.spe[rep,1] + spe/5 # mean sensitivity over 5 folds
    result.mat.sen[rep,1] <-  result.mat.sen[rep,1] + sen/5 # mean sensitivity over 5 folds
    result.mat.gme[rep,1] <-  result.mat.gme[rep,1] + gme/5 # mean G-mean over 5 folds


    # Fit best kernel SVM
    param.best.ksvm <- which(tune.mat.ksvm == max(tune.mat.ksvm), arr.ind = TRUE)
    cat("best parameter for ksvm: c = ",
        param.set.c[param.best.ksvm[1]],
        ", gamma = ",
        param.set.gamma[param.best.ksvm[2]], "\n"
        )
    
    model.ksvm.best <- svm(
      data = data.train,
      IS_FIRE ~ .,
      gamma = param.set.gamma[param.best.ksvm[2]],
      cost = param.set.c[param.best.ksvm[1]],
      kernel = "radial",
      scale = FALSE
    )
    
    cmat.ksvm <- table("truth" = data.test$IS_FIRE, "pred.ksvm" = predict(model.ksvm.best, data.test[1:4]))
    sen <- cmat.ksvm[2, 2] / sum(cmat.ksvm[2, ]) # sensitivity
    spe <- cmat.ksvm[1, 1] / sum(cmat.ksvm[1, ]) # specificity
    gme <- sqrt(sen * spe) # G-mean
    result.mat.spe[rep,2] <-  result.mat.spe[rep,2] + spe/5 # mean sensitivity over 5 folds
    result.mat.sen[rep,2] <-  result.mat.sen[rep,2] + sen/5 # mean sensitivity over 5 folds
    result.mat.gme[rep,2] <-  result.mat.gme[rep,2] + gme/5 # mean G-mean over 5 folds 

  
    
    # fit best linear svm
    param.best.lsvm <- which(tune.mat.lsvm == max(tune.mat.lsvm), arr.ind = TRUE)
    cat("best parameter for lsvm: c = ",  
        param.set.c[param.best.lsvm[1]], "\n"
    )
    
    svm.model.linearsvm <-
      svm(data = data.train,
          IS_FIRE ~ .,
          cost = param.set.c[param.best.lsvm[1]],
          kernel = "linear",
          scale = FALSE)
    
    cmat.linearsvm <- table("truth" = data.test$IS_FIRE, "pred.linearsvm" =  predict(svm.model.linearsvm, data.test[1:4]))
    sen <- cmat.linearsvm[2, 2] / sum(cmat.linearsvm[2, ]) # sensitivity
    spe <- cmat.linearsvm[1, 1] / sum(cmat.linearsvm[1, ]) # specificity
    gme <- sqrt(sen * spe) # G-mean
    result.mat.spe[rep,3] <-  result.mat.spe[rep,3] + spe/5 # mean sensitivity over 5 folds
    result.mat.sen[rep,3] <-  result.mat.sen[rep,3] + sen/5 # mean sensitivity over 5 folds
    result.mat.gme[rep,3] <-  result.mat.gme[rep,3] + gme/5 # mean G-mean over 5 folds 
  } # end of for loop level 2 (5-fold CV for performance metric evaluation)
}# end of for loop level 1 (monte carlo simulation)

print(colMeans(result.mat.sen))
print(colMeans(result.mat.spe))
print(colMeans(result.mat.gme))