## Classification Evaluation for the above six models##
classification.eval=data.frame(Model=c("Logistic","LDA","KNN","XGBoost","SVM","ANN"),Train.Error=err1,
                               Validation.Error=err2, Train.AUC=auc1, Validation.AUC=auc2)
classification.eval


## Sorting Smoothing Method for estimating real probability of default ##
par(mfrow=c(1,1))
par(pty="m")

#For logistic model

sort=order(pred.logit) #Index for sorted predicted probabilities
def.logit=as.numeric(test.logit$target)-1
sorted_target=def.logit[sort]
sorted_pred.logit=pred.logit[sort]

#Setting the smoothing parameter n=50
P1=numeric()

for(i in 1:6000){
  P1[i]=0
  for(k in max(i-50,1):min(i+50,6000)){
    P1[i]=P1[i]+sorted_target[k]
  }
  P1[i]=P1[i]/(2*50+1)
}

summary(lm(P1~sorted_pred.logit))#Fitting linear regression line

#Scatter plot
plot(sorted_pred.logit,P1,type="p",pch=1,xlab = "Predicted Probability of Logit Model(x)",ylab="True Probability(Y)",
     main="Scatter Plot for Logistic Regression",font.main=2)
lines(sorted_pred.logit,0.0044150+0.9723708*sorted_pred.logit,type="l",col="blue",lwd=2)
text(0.2,0.7,"Y=0.0044150+0.9723708*x\n R^2=0.929",col="green")


#For Linear Discriminent Analysis

sort=order(pred.lda.prob) #Index for sorted predicted probabilities
def.lda=as.numeric(test.logit$target)-1
sorted_target=def.lda[sort]
sorted_pred.lda=pred.lda.prob[sort]

#Setting the smoothing parameter n=50
P2=numeric()

for(i in 1:6000){
  P2[i]=0
  for(k in max(i-50,1):min(i+50,6000)){
    P2[i]=P2[i]+sorted_target[k]
  }
  P2[i]=P2[i]/(2*50+1)
}

summary(lm(P2~sorted_pred.lda))#Fitting linear regression line

#Scatter plot
plot(sorted_pred.lda,P2,type="p",pch=1,xlab = "Predicted Probability of LDA Model(x)",ylab="True Probability(Y)",
     main="Scatter Plot for LDA",font.main=2)
lines(sorted_pred.lda,0.0642664+0.7706105*sorted_pred.lda,type="l",col="blue",lwd=2)
text(0.2,0.7,"Y=0.0642664+0.7706105*x\n R^2=0.909",col="green")

#For KNN model 

sort=order(knn.prob) #Index for sorted predicted probabilities
def.knn=as.numeric(test.knn$target)-1
sorted_target=def.knn[sort]
sorted_pred.knn=knn.prob[sort]

#Setting the smoothing parameter n=50
P3=numeric()

for(i in 1:6000){
  P3[i]=0
  for(k in max(i-50,1):min(i+50,6000)){
    P3[i]=P3[i]+sorted_target[k]
  }
  P3[i]=P3[i]/(2*50+1)
}

summary(lm(P3~sorted_pred.knn))#Fitting linear regression line

#Scatter plot
plot(sorted_pred.knn,P4,type="p",pch=1,xlab = "Predicted Probability of KNN Model(x)",ylab="True Probability(Y)",
     main="Scatter Plot for KNN",font.main=2)
lines(sorted_pred.knn, 0.0302506+0.8448061 *sorted_pred.knn,type="l",col="blue",lwd=2)
text(0.2,0.7,"Y=0.0302506+0.8448061*x\n R^2=0.9112",col="green")

#For XGBOOST model 

sort=order(xgb.prob) #Index for sorted predicted probabilities
def.xgb=as.numeric(test_xgb$target)-1
sorted_target=def.xgb[sort]
sorted_pred.xgb=xgb.prob[sort]

#Setting the smoothing parameter n=50
P4=numeric()

for(i in 1:6000){
  P4[i]=0
  for(k in max(i-50,1):min(i+50,6000)){
    P4[i]=P4[i]+sorted_target[k]
  }
  P4[i]=P4[i]/(2*50+1)
}

summary(lm(P4~sorted_pred.xgb))#Fitting linear regression line

#Scatter plot
plot(sorted_pred.xgb,P4,type="p",pch=1,xlab = "Predicted Probability of XGBOOST Model(x)",ylab="True Probability(Y)",
     main="Scatter Plot for XGBOOST",font.main=2)
lines(sorted_pred.xgb, 0.0041041+0.9381054*sorted_pred.xgb,type="l",col="blue",lwd=2)
text(0.2,0.7,"Y= 0.0041041+0.9381054*x\n R^2=0.9341",col="green")

#For Support Vector Machine

sort=order(pred.svm.prob$yes) #Index for sorted predicted probabilities
def.svm=as.numeric(test.svm$target)-1
sorted_target=def.svm[sort]
sorted_pred.svm=pred.svm.prob$yes[sort]

#Setting the smoothing parameter n=50
P5=numeric()

for(i in 1:6000){
  P5[i]=0
  for(k in max(i-50,1):min(i+50,6000)){
    P5[i]=P5[i]+sorted_target[k]
  }
  P5[i]=P5[i]/(2*50+1)
}

summary(lm(P5~sorted_pred.svm))#Fitting linear regression line

par(mfrow=c(1,1))
par(pty="m")
plot(sorted_pred.svm,P5,type="p",pch=1,xlab = "Predicted Probability of SVM Model(x)",ylab="True Probability(Y)",
     main="Scatter Plot for Support Vector Machine",font.main=2)
lines(sorted_pred.svm,-0.012319+1.000582*sorted_pred.svm,type="l",col="blue",lwd=2)
text(0.3,0.7,"Y= -0.012319+1.000582*x\n R^2=0.8845",col="green")

#For ANN

sort = order(pred.ann.prob) #Index for sorted predicted probabilites 
def.ann = as.numeric(test.ann$target)-1
sorted_target =def.ann[sort]
sorted_pred.ann = pred.ann.prob[sort]

#Setting the smoothing parameter n=50 

P6=numeric()

for(i in 1:6000){
  P6[i]=0
  for(k in max(i-50,1):min(i+50,6000) )  {
    P6[i]=P6[i]+sorted_target[k]
  } 
  P6[i] =P6[i]/(2*50+1)
}
summary(lm(P6~sorted_pred.ann))

#Fitting Linear regression Line
plot(sorted_pred.ann, P6, type ="p", pch = 1, xlab="Predicted Probability of ANN Model(x)" , ylab ="True Probability(Y)",
     main = "Scatter Plot for ANN ", font.main = 2)
lines(sorted_pred.ann, 0.0033099+0.9379403*sorted_pred.ann , type = "l",col= "blue", lwd=2)
text(0.3,0.7,"Y= 0.0033099+0.9379403*x\n R^2=0.9401",col="green")



#Model evaluation table ##
Prediction.eval=data.frame(Model=c("Logistic","LDA","KNN","XGBoost","SVM","ANN"),Intercept=c(0.00441,0.06426,0.0302506,0.0041041,-0.012319,0.0033099),
                           Slope=c(0.9724,0.7706,0.8448061,0.9381054,1.000582,0.9379403),Rsq=c(0.929,0.909,0.9112,0.9341,0.884,0.9401))
Prediction.eval