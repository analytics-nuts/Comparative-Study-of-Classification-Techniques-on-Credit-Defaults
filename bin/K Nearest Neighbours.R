##FItting K Nearest Neighbor classifier ##

#Preprocessing the data
# min-max normalization of quantitative features
f=function(x)
{
  return((x-min(x))/(max(x)-min(x)))
}

quanti.norm=quanti

setDF(quanti.norm)#Converting into data.frame

for(i in 1:14){
  quanti.norm[,i]=f(quanti.norm[,i]) #Normalization
}

#Dummy encoding for factor variables
quali.dummy=dummy.data.frame(quali)

#Merging the normalized data and encoded dummies
target=recode_factor(target,'0'="no",'1'="yes")

data.knn=cbind(quanti.norm,quali.dummy,target)

setDT(data.knn)#Converting into data.table

#Test train split in 80:20 ratio
set.seed(666) #For Reproducibility

ind=sample(nrow(data.knn),nrow(data.knn)*0.8,replace = F)
train.knn=data.knn[ind,]
test.knn=data.knn[-ind,]

trainy=train.knn$target

model.list=list()#empty list
v=numeric()

set.seed(222)
for(i in 1:30){
  
  model.list[[i]]=knn3(train.knn[,-88],trainy,k=i)
  tab=table(prediction=predict(model.list[[i]],test.knn[,-88],type = "class"),truth=test.knn$target)
  v[i]=sum(diag(tab)/sum(tab))
  
}

plot(1:30,v,type="both",xlab="k",ylab="accuracy")
abline(v=19)
which.max(v)

model.knn=knn3(train.knn[,-88],trainy,k=19)#Best model in terms of accuracy

#Prediction
#Test set
set.seed(666)
table(prediction=predict(model.knn,test.knn[,-88],type = "class"),truth=test.knn$target)
err2[3]=(826+238)/6000

knn.prob=predict(model.knn,test.knn[,-88],type="prob")[,2] #Probalities of "yes"

#Training set
set.seed(666)
table(prediction=predict(model.knn,train.knn[,-88],type = "class"),truth=train.knn$target)
err1[3]=(3405+836)/24000

##Ploting ROC curve and AUC for test and train set
par(mfrow=c(1,2))
par(pty="s")

roc(train.knn$target,predict(model.knn,train.knn[,-88],type="prob")[,2],plot=T,col="#69b3a2",print.auc=T,legacy.axes=TRUE,
    percent = T,xlab="False Positive percentage",ylab="True Positive percentage",lwd=5,main="Train Set")


roc(test.knn$target,knn.prob,plot=T,col="navyblue",print.auc=T,legacy.axes=TRUE,percent = T,
    xlab="False Positive percentage",ylab="True Positive percentage",lwd=5,main="Test Set")

auc1[3]=0.814
auc2[3]=0.751

#Cumulative gain chart for test set
area_mod=performance(prediction(sort(knn.prob,decreasing=F),test.knn$target[order(knn.prob,decreasing=F)])
                     ,measure = "auc")@y.values[[1]] #Area under curve

x=unlist(slot(performance(prediction(sort(knn.prob,decreasing=F),test.knn$target[order(knn.prob,decreasing=F)])
                          ,"tpr","rpp"),'x.values'))
y=unlist(slot(performance(prediction(sort(knn.prob,decreasing=F),test.knn$target[order(knn.prob,decreasing=F)])
                          ,"tpr","rpp"),'y.values'))
par(mfrow=c(1,1))
par(pty="m")
plot(x,y,type="l",col="green",lwd=4,main="Cumulative gain chart for KNN model",font.main=2,xlab="Rate of Positive Prediction",
     ylab="Cumulative proportion of target")
abline(a=0,b=1,col="red")
text(0.8,0.2,"Area under model\n curve is 0.7506",font = 2,col="steelblue4")
text(0.1,0.85,"best line->")
text(0.8,0.69,"<-base line")
lines(x=c(0,0.2166667,1),y=c(0,1,1),lwd=2,col="blue")