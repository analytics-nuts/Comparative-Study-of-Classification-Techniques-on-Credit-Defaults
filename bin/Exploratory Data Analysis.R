##Loading the required libraries
library(data.table)
library(ggplot2)
library(psych)
library(GGally)
library(dplyr)
library(cowplot)
library(caret)
library(pROC)
library(ROCR)
library(MASS)
library(dummies)
library(class)
library(xgboost)
library(e1071)
library(nnet)

##Reading the data in R session

credit=fread("default of credit card clients.csv")

##Having a look at the data 
str(credit)
head(credit)

##Droping the unnecessary variable/column
credit=credit[,-1]#There's no use of the column id in our analysis

##Changing the name of the variable PAY_0 to PAY_1
names(credit)
names(credit)[6]="PAY_1"
names(credit)[23] = "target"

##Looking for missing value
sum(is.na(credit))#it is observed that there's no missing values

##Transforming the variables SEX,MARRIAGE,EDUCATION and default payment into factors
df=as.data.frame(credit)
df[c("SEX","MARRIAGE","EDUCATION","target","PAY_1",
     "PAY_2","PAY_3","PAY_4","PAY_5","PAY_6")]= lapply(df[c("SEX","MARRIAGE","EDUCATION",
                                                            "target","PAY_1","PAY_2","PAY_3",
                                                            "PAY_4","PAY_5","PAY_6")]
                                                       ,function(x) as.factor(x))



str(df)
credit=df
rm(df)

##Extended data dictionary
summary(credit)

##Correlation analysis and Correlogram plot

df=credit[,c(-2:-4,-24)]
cor(credit[,c(-2:-4,-24)])
ggpairs(df,aes(color=credit$`default payment next month`),title = "Correlogram")
ggcorr(df,method=c("everything", "pearson"))+ggtitle("Correlation Steps")

##Visualizing the data

ggplot(credit,aes(x=credit$LIMIT_BAL,fill=credit$`default payment next month`))+
  geom_density(alpha=0.6,show.legend = T,color="blue")+
  ggtitle("Density plot oh Credit Amount")+
  xlab("Credit Amount")

#We see Customers with relatively lower credit tend to be defualt for payment

ggplot(credit,aes(x=credit$AGE,fill=credit$`default payment next month`))+
  geom_histogram(show.legend = T,alpha=0.9)+
  ggtitle("AGE for different customers with respect to default")+
  xlab("AGE")
#Customers with age between 20-35 have relatively higher no of defaults

ggplot(credit,aes(x=credit$MARRIAGE,group=credit$`default payment next month`))+
  geom_bar(show.legend = T,fill="lightblue")+
  ggtitle("Default for different marital status")+
  xlab("Marriage")+
  facet_grid(~credit$`default payment next month`)

#No of default is slightly higher for married customers

ggplot(credit,aes(x=credit$SEX,fill=credit$`default payment next month`))+
  geom_bar(aes(y=(..count..)/sum(..count..)), show.legend = T)+
  ggtitle("Default for different gender")+
  xlab("SEX")+
  ylab("proportion")

# Proportion of default is greater for female compared to male.

p=list()#creating empty plot list

for(i in 12:17){
  p[[i]]= ggplot(credit,aes(x=credit[,1],y=credit[,i],color=credit$`default payment next month`))+
    geom_point(show.legend = T)+xlab("Limit_Bal")+ylab(paste0("Bill_Amt",i-11,sep=""))+ggtitle(paste0("Limit_bal vs Bill_amt",i-11,sep=""))
  
}

plot_grid(p[[12]],p[[13]],p[[14]],p[[15]],p[[16]],p[[17]],nrow=3,ncol=2)

#There's a cluster of default cusromers in the lower range of limit bal and Bill amounts

q=list()#creating empty plot list

for(i in 6:11){
  q[[i]]=ggplot(credit,aes(x=credit[,i],y=credit[,1],color=credit$`default payment next month`,palette="jco"))+
    geom_point(show.legend = T)+xlab(paste0("PAY_",i-5,sep=""))+ylab("Limit Bal")+
    ggtitle(paste0("PAY_",i-5,"Vs Limit Bal",sep=""))
}

plot_grid(q[[6]],q[[7]],q[[8]],q[[9]],q[[10]],q[[11]],nrow=3,ncol=2)

#Most of the default customers have delays in repayment status

## Feature Engineering

##Redefining the variables EDUCATIOn,Marriage according to the revised description of the dataset.
credit$EDUCATION = recode_factor(credit$EDUCATION, '4' = "0", '5' = "0", '6' = "0"
                                 ,.default = levels(credit$EDUCATION))
table(credit$EDUCATION)

credit$MARRIAGE = recode_factor(credit$MARRIAGE, '0'="3"
                                ,.default = levels(credit$MARRIAGE))
table(credit$MARRIAGE)

##Partitioning the whole data in quantitative and qualitative parts and defining the target
quanti=credit[,c(-2:-4,-6:-11,-24)]
quali=credit[,c(2:4,6:11)]

target=credit$target
(table(target)/length(target))


all.features=cbind(quanti,quali,target)
head(all.features)

##Classification models accuracy comparison
err1=numeric()
err2=numeric() #Creating empty vectors for further comparisons
auc1=numeric()
auc2=numeric()