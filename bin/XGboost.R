##FItting XGBoost classifier ##

set.seed(666)
library(parallel) 
# Calculate the number of cores
no_cores <- detectCores()-1

library(doParallel)
# create the cluster for caret to use
cl <- makePSOCKcluster(no_cores)
registerDoParallel(cl)

##Preprocessing the data for XGboost

target=as.numeric(recode_factor(target,'no'="0",'yes'="1"))
target=ifelse(target==1,0,1)#assiging 1 for default and 0 for non-default

data_xgb=cbind(quanti,quali.dummy,target)

#Test train split in 80:20 ratio
set.seed(666) #For Reproducibility

ind=sample(nrow(data_xgb),nrow(data_xgb)*0.8,replace = F)
train_xgb=data_xgb[ind,]
test_xgb=data_xgb[-ind,]

#Using caret package to tune the hyperparameters further
xgb_control=trainControl(method="cv",number = 3,allowParallel = T,
                         classProbs = T,summaryFunction = twoClassSummary)
xgb_grid=expand.grid(nrounds=seq(100,200,by=25),eta=c(0.08,0.09,seq(0.1,0.5,by=0.2)),max_depth=seq(2,6,by=1),gamma=c(0,0.5,1),
                     subsample=c(0.8,1),colsample_bytree=seq(0.8,1,by=0.1),min_child_weight=seq(1,3,by=1))
model.xgb=train(x=as.matrix(train_xgb[,-88]),y=recode_factor(as.factor(train_xgb$target),'0'="no",'1'="yes"),
                method="xgbTree",
                tuneGrid = xgb_grid
                ,trControl = xgb_control)

# model.xgb$bestTune

#The final model

set.seed(666)
xgb_param=trainControl(method="none",classProbs = T,summaryFunction = twoClassSummary,allowParallel = T)
model.xgb=train(x=as.matrix(train_xgb[,-88]),y=recode_factor(as.factor(train_xgb$target),'0'="no",'1'="yes"),
                method="xgbTree",verbose=T,
                tuneGrid = expand.grid(eta=0.08,nrounds=125,
                                       max_depth=6,colsample_bytree=0.9,gamma=0.5,
                                       min_child_weight=1,subsample=0.8)
                ,trControl = xgb_param)



stopCluster(cl)
registerDoSEQ()

#Alternative Method

#xgb_param=list(booster="xgbtree",eta=0.3,eval_metric="auc",objective="binary:logistic",max_depth=5)

#model.xgb_cv=xgb.cv(params = xgb_param,data = as.matrix(train_xgb[,-88]),nrounds = 100,nfold = 5,verbose = T,
#label=train_xgb$target,early_stopping_rounds = 50,print_every_n = 1)
#Stopping. Best iteration:
#[16]	train-auc:0.819109+0.001889	test-auc:0.777968+0.007074

#xgb_param=list(booster="gbtree",eta=0.1,eval_metric="auc",objective="binary:logistic",max_depth=10)

#model.xgb_cv=xgb.cv(params = xgb_param,data = as.matrix(train_xgb[,-88]),nrounds = 300,nfold = 5,verbose = T,
#label=train_xgb$target,early_stopping_rounds = 50,print_every_n = 1)
#Stopping. Best iteration:
#[38]	train-auc:0.935541+0.004116	test-auc:0.774223+0.005458 #we observe a little overfitting with eta=0.1 and max_depth=10

#Tuning eta and max_depth for different nrounds
#eta=c(0.08,0.09,seq(0.1,0.8,by=0.1))
#eval=data.frame()
#for(k in 1:10){

#  for(i in 1:10){
xgb_param=list(booster="gbtree",eta=eta[k],eval_metric="auc",objective="binary:logistic",
               max_depth=i)
model.xgb_cv=xgb.cv(params = xgb_param,data = as.matrix(train_xgb[,-88]),nrounds = 400,nfold = 5,verbose = T,
                    label=train_xgb$target,early_stopping_rounds = 50)

temp=data.frame(eta=eta[k],max_depth=i,iter=which.max(data.frame(model.xgb_cv$evaluation_log)[,4])
                ,auc=max(data.frame(model.xgb_cv$evaluation_log)[,4]))

eval=rbind(eval,temp) 

#}

#head(eval) 
#max(eval$auc)
#which.max(eval$auc)


#Prediction for train set
p_train=predict(model.xgb,newdata = as.matrix(train_xgb[,-88]))
conf4.train = table(predict=p_train,true=train_xgb$target)

err1[4] = (conf4.train[1,2]+conf4.train[2,1])/sum(conf4.train)

p_train_prob = predict(model.xgb,newdata = as.matrix(train_xgb[,-88]),type="prob")$yes

#Prediction for test set
p_test=predict(model.xgb,newdata = as.matrix(test_xgb[,-88]))
conf4 = table(predict=p_test,true=test_xgb$target)
err2[4]= (conf4[1,2]+conf4[2,1])/sum(conf4)

xgb.prob=predict(model.xgb,newdata = as.matrix(test_xgb[,-88]),type="prob")$yes



# ROC plot for train and test set
par(mfrow=c(1,2))
par(pty="s")

roc(response=as.factor(train_xgb$target),predictor=p_train_prob,percent = T,plot = T,col="#69b3a2",print.auc=T,
    legacy.axes=T,xlab="False Positive percentage",ylab="True Positive percentage",lwd=5,main="Train Set")

roc(response=as.factor(test_xgb$target),predictor=xgb.prob,percent = T,plot = T,col="navyblue",print.auc=T,
    legacy.axes=T,xlab="False Positive percentage",ylab="True Positive percentage",lwd=5,main="Test Set")

auc1[4]=0.874
auc2[4]=0.785

#Cumulative gain chart for test set
test_xgb$target=as.factor(test_xgb$target)
area_mod=performance(prediction(sort(xgb.prob,decreasing=F),test_xgb$target[order(xgb.prob,decreasing=F)])
                     ,measure = "auc")@y.values[[1]] #Area under curve

x=unlist(slot(performance(prediction(sort(xgb.prob,decreasing=F),test_xgb$target[order(xgb.prob,decreasing=F)])
                          ,"tpr","rpp"),'x.values'))
y=unlist(slot(performance(prediction(sort(xgb.prob,decreasing=F),test_xgb$target[order(xgb.prob,decreasing=F)])
                          ,"tpr","rpp"),'y.values'))
par(mfrow=c(1,1))
par(pty="m")
plot(x,y,type="l",col="green",lwd=4,main="Cumulative gain chart for XGBoost model",font.main=2,xlab="Rate of Positive Prediction",
     ylab="Cumulative proportion of target")
abline(a=0,b=1,col="red")
text(0.8,0.2,"Area under model\n curve is 0.7852",font = 2,col="steelblue4")
text(0.1,0.85,"best line->")
text(0.8,0.69,"<-base line")
lines(x=c(0,0.2166667,1),y=c(0,1,1),lwd=2,col="blue")