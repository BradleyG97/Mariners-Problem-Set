library(tidyverse)
library(xgboost)
library(caTools)
train <- read.csv("/Users/Bradl/Downloads/data-train.csv")
test <- read.csv("/Users/Bradl/Downloads/data-test.csv")

outfields <- summarise(group_by(train %>% filter(top == 1),venue_id),
                       Out = mean(is_airout,na.rm=T),
                       n = n())

modelData <- train %>% select(is_airout,exit_speed,hit_spin_rate,vert_exit_angle,horz_exit_angle,temperature,bat_side)
oneHot <- dummyVars(~ bat_side, data = modelData)
temp <- predict(oneHot, modelData[1:nrow(modelData),])
modelData <- cbind(modelData,temp) %>% select(-bat_side) 

set.seed(103)

sample <- sample.split(modelData$is_airout,SplitRatio = .85)
trainSubset <- subset(modelData,sample == T)
testSubset <- subset(modelData,sample == F)

airoutTrain <- xgb.DMatrix(data = as.matrix(trainSubset %>% select(-is_airout)),
                           label = trainSubset$is_airout)
airoutTest <- xgb.DMatrix(data = as.matrix(testSubset %>% select(-is_airout)),
                          label = testSubset$is_airout)
xgbGrid <- expand.grid(
  nrounds = c(50,75,100),
  max_depth = c(3, 4, 5, 6),
  eta = c(0.1, 0.2, 0.3, .4),
  gamma = 0,
  colsample_bytree = c(0.6, 0.8, 1),
  min_child_weight = c(1,3,5),
  subsample = c(0.8,.9,1)
)
trainControl <- trainControl(
  method = "cv",      
  number = 5,         
  verboseIter = TRUE
)
xgbTrain <- train(
  x = as.matrix(trainSubset %>% select(-is_airout)), 
  y = trainSubset$is_airout, 
  trControl = trainControl,
  tuneGrid = xgbGrid,
  method = "xgbTree",
  objective = "binary:logistic",
  eval_metric = "logloss"
)
xgbTrain$bestTune
# Best Iteration - Round 645
xgb_params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  max.depth = 6, 
  eta = 0.2, 
  gamma = 0,
  colsample_bytree = 1, 
  min_child_weight = 5, 
  subsample = 0.9
)
airoutModel <- xgb.train(
  params = xgb_params,
  data = airoutTrain,
  nrounds = 100,
  early_stopping_rounds = 3,
  watchlist = list(train = airoutTrain, test = airoutTest),
  verbose = TRUE
)
xgb.save(airoutModel,"airoutModel.model")

xgb.plot.importance(xgb.importance(model = airoutModel))

modelData <- test %>% select(exit_speed,hit_spin_rate,vert_exit_angle,horz_exit_angle,temperature,bat_side)
oneHot <- dummyVars(~ bat_side, data = modelData)
temp <- predict(oneHot, modelData[1:nrow(modelData),])
modelData <- cbind(modelData,temp) %>% select(-bat_side) 
xOut <- predict(airoutModel,as.matrix(modelData))
test <- test %>% mutate(p_airout = xOut)

# Problem 2
modelData <- train %>% select(exit_speed,hit_spin_rate,vert_exit_angle,horz_exit_angle,temperature,bat_side)
oneHot <- dummyVars(~ bat_side, data = modelData)
temp <- predict(oneHot, modelData[1:nrow(modelData),])
modelData <- cbind(modelData,temp) %>% select(-bat_side) 
xOut <- predict(airoutModel,as.matrix(modelData))
train <- train %>% mutate(p_airout = xOut)

train <- train %>% mutate(firstFielderPos = case_when(
  first_fielder == rf_id ~ "RF",
  first_fielder == cf_id ~ "CF",
  first_fielder == lf_id ~ "LF",
  is.na(first_fielder) ~ NA
))

modelData <- train %>% filter(!(is.na(firstFielderPos))) %>% select(firstFielderPos,exit_speed,hit_spin_rate,vert_exit_angle,horz_exit_angle,temperature,bat_side)
oneHot <- dummyVars(~ bat_side, data = modelData)
temp <- predict(oneHot, modelData[1:nrow(modelData),])
modelData <- cbind(modelData,temp) %>% select(-bat_side) 

set.seed(703)

sample <- sample.split(modelData$firstFielderPos,SplitRatio = .85)
trainSubset <- subset(modelData,sample == T)
testSubset <- subset(modelData,sample == F)

modelData$firstFielderPos <- as.factor(modelData$firstFielderPos)
trainSubset$firstFielderPos <- as.numeric(as.factor(trainSubset$firstFielderPos)) - 1
testSubset$firstFielderPos <- as.numeric(as.factor(testSubset$firstFielderPos)) - 1

fielderTrain <- xgb.DMatrix(data = as.matrix(trainSubset %>% select(-firstFielderPos)),
                            label = trainSubset$firstFielderPos)
fielderTest <- xgb.DMatrix(data = as.matrix(testSubset %>% select(-firstFielderPos)),
                           label = testSubset$firstFielderPos)

xgb_params <- list(
  objective = "multi:softmax",
  eval_metric = "mlogloss",
  max.depth = 6,
  eta = 0.2,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 5,
  subsample = 0.9,
  num_class = length(unique(modelData$firstFielderPos))
)

fielderClassifier <- xgb.train(
  params = xgb_params,
  data = fielderTrain,
  nrounds = 100,
  early_stopping_rounds = 3,
  watchlist = list(train = fielderTrain, test = fielderTest),
  verbose = TRUE
)

modelData <- train %>% select(exit_speed,hit_spin_rate,vert_exit_angle,horz_exit_angle,temperature,bat_side)
oneHot <- dummyVars(~ bat_side, data = modelData)
temp <- predict(oneHot, modelData[1:nrow(modelData),])
modelData <- cbind(modelData,temp) %>% select(-bat_side) 
prediction <- predict(fielderClassifier,as.matrix(modelData))
train <- cbind(train,prediction) %>% mutate(maxFielder = case_when(prediction == 0 ~ "CF",
                                                                   prediction == 1 ~ "LF",
                                                                   prediction == 2 ~ "RF"),
                                            maxFielderID = case_when(maxFielder == "CF" ~ cf_id,
                                                                     maxFielder == "RF" ~ rf_id,
                                                                     maxFielder == "LF" ~ lf_id))



p <- train %>% filter(cf_id == 15411 & maxFielder == "CF")
sum(p$is_airout[p$first_fielder == 15411],na.rm=T)/sum(p$maxFielder == "CF",na.rm=T)
train <- train %>% mutate(OAE = ifelse(is_airout == 1,1-p_airout,-1*(p_airout)))
outfielder <- summarise(group_by(train %>% filter(maxFielder == "CF"),maxFielderID,maxFielder),
                        outs = sum(is_airout,na.rm=T)/n(),
                        out_prob = mean(p_airout,na.rm=T),
                        out_diff = round(outs-out_prob,3),
                        OAE = sum(OAE,na.rm=T),
                        n = n()) %>% filter(n >= 100)
percentiles <- quantile(outfielder$outs, probs = seq(0, 1, by = 0.01))
outfielder$outsPercentile <- findInterval(outfielder$outs, percentiles, all.inside = TRUE)
percentiles <- quantile(outfielder$out_diff, probs = seq(0, 1, by = 0.01))
outfielder$outDiffPercentile <- findInterval(outfielder$out_diff, percentiles, all.inside = TRUE)
percentiles <- quantile(outfielder$OAE, probs = seq(0, 1, by = 0.01))
outfielder$oaePercentile <- findInterval(outfielder$OAE, percentiles, all.inside = TRUE)

ggplot(outfielder %>% filter(n >= 100),aes(x=1,y=OAE)) + geom_violin() + theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5, size = 15),
        axis.title.x = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank()) + 
  labs(title = "Outs Above Expected Ranking",
       y = "Outs Above Expected") + geom_hline(yintercept = 4.70702331,color = "red")

ggplot(p, aes(x = exit_speed, y = vert_exit_angle, shape = factor(is_airout), color = p_airout)) +
  geom_point() +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, size = 15)) +
  scale_shape_manual(values = c(16, 17), labels = c("Hit", "Out")) +
  labs(x = "Exit Velo.",
       y = "Launch Angle",
       title = "In Play Results for 15411 given EV and LA",
       shape = "In Play Result",
       color = "xOut%")

ggplot(p, aes(x = horz_exit_angle, y = exit_speed, shape = factor(is_airout), color = p_airout)) +
  geom_point() +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, size = 15)) +
  scale_shape_manual(values = c(16, 17), labels = c("Hit", "Out")) +
  labs(y = "Exit Velo.",
       x = "Spray Angle",
       title = "In Play Results for 15411 given EV and LA",
       shape = "In Play Result",
       color = "xOut%")
