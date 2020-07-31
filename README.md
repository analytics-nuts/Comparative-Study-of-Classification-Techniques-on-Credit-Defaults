# **Comparative Study of Classification Techniques on Credit Defaults**
### *A study of six data mining techniques into the Credit Card Defaulter Data -Taiwan (2005)*

#### *by,  Kousik Somodder,  Sagarnil Bose &  Shreyashi Saha*


![](images/banner.jpg)


## **Introduction**

This project is an attempt toward a comparative study among six different machine- learning models for their accuracy in 
predicting the target class as well as their accuracy for representing the real probability of an individual belonging to the actual 
class from the perspective of risk management using the *Sorting Smoothing Method*. In this project, we summoned the **“Default of credit card clients, Taiwan 2005”** 
dataset available at [UCI machine-learning repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients).


## **Objective**

The sole purpose of this project is to compare the six machine-learning models, namely *Logistic regression, Discriminant analysis, K-nearest neighbor, Support vector machine, XGBOOST and Artificial neural network* based on their accuracy of classification performane and effectiveness in representing the real probability of an individual belonging to the actual class.

 For assessment of classification accuracy, different measures such as classification error rate from confusion matrix, ROC curve and area under curve (AUC) are employed. For the assessment of accuracy of predicted probabilities, we’ll use the scatter plot of real probabilities of default(Y) vs predicted probability of default(x) from each of the six techniques. Then fit a linear regression line Y=A+Bx from the scatter plot and decide the best predicting model for which A is closest to 0 and B is closest to 1 and Rsq is highest.
 
 
## **About The Data Set**

Our dataset *‘Default of credit card clients’* consists of informations about transactions from April 2005 to September 2005 of 30000 clients who were credit holders in a bank in Taiwan. This dataset has binary response variable ‘default.payment.next.month’ that takes the value 1 if the corresponding client has default payment and 0 otherwise. Out of 30000 clients 6636(22.12%) were with default payment. There are 23 other independent or explanatory variables:

* 	LIMIT_BAL: Amount of the given credit(NT dollar), it includes both the individual consumer credit as well as the person’s family credit
* 	SEX: 1=male and 2= female
* 	EDUCATION: 1= graduate school, 2= university,3=high-school , 4=others
* 	MARRIAGE: Marital status. 1=married,2=single, 3= others
* 	AGE: Age of the client
* 	PAY_1-PAY_6: History of past payments from April to September 2005. Like PAY_1=The repayment status in September, …., PAY_6=The repayment status of April 2005. The scaling of the status is as follows -2= no transactions history,-1=paid duly,0=revolving ,1=payment delay for one month ,2= payment delay for 2 months ,….,9=payment delay for 9 months or more.
*	 BILL_AMT1-BILL_AMT6: Amount of bill statement (NT dollar). BILL_AMT1=amount of bill statement in September  ,…., BILL_AMT6= amount of bill statement in April 2005.
* 	PAY_AMT1-PAY_AMT6:Amount of previous payment(NT dollar).PAY_AMT1=amount paid in September ,…., PAY_AMT6=amount paid in April 2005.

 
## **Content List**

1.	Loading required packages into the session
1.	Reading the data into the session
1.	Having a look at the data, its structure and summary
1.	Visualization
1.	Feature engineering 
1.	Data preprocessing and test-train split of the data
1.	Model fitting 
1.	Prediction on training and test set and computing error rate and AUC
1.	Plotting ROC curves and cumulative lift charts
1.	Sorting smoothing method 
1.	Scatter plot and linear regression line fitting and comparison study for the models
1.	Conclusion 

 
 ## **Let’s get started!**
 
 #### **_Loading Packages_**
 
 We’ll load some packages into the session first, required in this project. Such as *data.table, dplyr* for data importing and wrangling, *ggplot2, cowplot, pROC,ROCR* for visualization of data and other plotting, *caret* for models training and several other packages, using `library()` function. If the package is not installed then it has to be installed using `install.packages("package name")`. In our case, we have our packages installed, we just need bring them into our session.
 
 ```Rscript
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

 ```
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
