 # load data and change to rds file
library(data.table)
setwd('D:/gitcode/kaggle-allstate-claims-severity/')
train <- fread(input = "D:/gitcode/kaggle-allstate-claims-severity/input/train.csv", nrows = 100000)
# saveRDS(train, file = './cache/train.rds')

# load data
# train <- readRDS(file = './cache/train.rds')

# library(ggplot2)
# library(dplyr)
# library(hash)
# ggplot(data= train)+geom_histogram(aes(log(loss)))
# 
# # hash the category
# all_cate <- NULL
# for(i in 1:length(names(train))){
#   if(grepl(pattern = 'cat*', names(train)[i])) {
#         cat('符合条件，第',i,'个变量n')
#         cate_detail <- unlist(unique(train[, ..i]))
#         all_cate <<- unique(c(all_cate, cate_detail))
#   } else {
#     cat('不符合条件', 'n')
#   }
# }
# 
# hashcate <- hash(all_cate, 1:length(all_cate))
# tt <- values(hashcate, keys =c('A','B'))
# 
# # 对变量进行转换
# for(i in 1:116){
#   eval(parse(text=paste0("train <- mutate(train, cat_t_",i," = values(hashcate, keys = train$cat",i,"))")))
# }
# 
# summary(train)
# ggplot()+geom_point(data = train, aes(x = cat_t_10, y =loss))

#  one-hot encoding 
library(caret)
library(dplyr)
ohe_feats = c(paste0("cat",1:116, collapse = '+'))
dummies <- dummyVars(~ cat1+cat2+cat3+cat4+cat5+cat6+cat7+cat8+cat9+cat10+cat11+cat12+cat13+cat14+cat15+cat16+cat17+cat18+cat19+cat20+cat21+cat22+cat23+cat24+cat25+cat26+cat27+cat28+cat29+cat30+cat31+cat32+cat33+cat34+cat35+cat36+cat37+cat38+cat39+cat40+cat41+cat42+cat43+cat44+cat45+cat46+cat47+cat48+cat49+cat50+cat51+cat52+cat53+cat54+cat55+cat56+cat57+cat58+cat59+cat60+cat61+cat62+cat63+cat64+cat65+cat66+cat67+cat68+cat69+cat70+cat71+cat72+cat73+cat74+cat75+cat76+cat77+cat78+cat79+cat80+cat81+cat82+cat83+cat84+cat85+cat86+cat87+cat88+cat89+cat90+cat91+cat92+cat93+cat94+cat95+cat96+cat97+cat98+cat99+cat100+cat101+cat102+cat103+cat104+cat105+cat106+cat107+cat108+cat109+cat110+cat111+cat112+cat113+cat114+cat115+cat116, data = train)
train_category <- as.data.frame(predict(dummies, newdata = train))
train_combine <- cbind(train%>%select(which(colnames(train)%in%c('loss', 'cont1', 'cont2', 'cont3', 'cont4', 'cont5', 'cont6', 'cont7', 'cont8', 'cont9', 'cont10', 'cont11', 'cont12', 'cont13', 'cont14'))), train_category)

# train model
library(xgboost)
# create train and test sets
trainsample <- caret::createDataPartition(train_combine$loss, p = 0.6, list =F)
traindata <- train_combine[trainsample,]
testdata <- train_combine[-trainsample,]
rm(trainsample)

# change to dmatrix
dtrain <- xgb.DMatrix(data.matrix(traindata[,-15]), label = traindata$loss, missing = NA)
dtest <- xgb.DMatrix(data.matrix(testdata[,-15]), label = testdata$loss, missing = NA)

# 
param <- list(booster = 'gbtree',
              max_depth = 5, 
              subsample= 0.7, 
              colsample_bytree = 0.7,
              objective = 'reg:linear',
              eval_metric = 'logloss',
              eta = 0.05
              )

model.xgb.cv <- xgb.cv(params = param, 
                       nrounds = 1500, 
                       data = dtrain, 
                       early_stopping_rounds = 15, 
                       maximize = T,
                       nfold = 5)













