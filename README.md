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


 
 We’ll load some packages into the session first, required in this project. Such as *data.table, dplyr* for data importing and wrangling, *ggplot2, cowplot, pROC,ROCR* for visualization of data and diagonistic plotting, *caret* for models training and several other packages, using `library()` function. If the package is not installed then it has to be installed using `install.packages("package name")`. In our case, we have our packages installed, we just need bring them into our session.
 
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
 
 We’ll read the data ‘Default of credit card client’ in as a csv file into an object named as credit.
 
 ```{r}
##Reading the data in R session
credit=fread("default of credit card clients.csv")
```
Let’s take a look at how the first few rows look like as well as the structures of the variables.
```{r}
##Having a look at the data 
str(credit)
head(credit)
```  


```{r}
Classes ‘data.table’ and 'data.frame':	30000 obs. of  25 variables:
 $ ID                        : int  1 2 3 4 5 6 7 8 9 10 ...
 $ LIMIT_BAL                 : int  20000 120000 90000 50000 50000 50000 500000 100000 140000 20000 ...
 $ SEX                       : int  2 2 2 2 1 1 1 2 2 1 ...
 $ EDUCATION                 : int  2 2 2 2 2 1 1 2 3 3 ...
 $ MARRIAGE                  : int  1 2 2 1 1 2 2 2 1 2 ...
 $ AGE                       : int  24 26 34 37 57 37 29 23 28 35 ...
 $ PAY_0                     : int  2 -1 0 0 -1 0 0 0 0 -2 ...
 $ PAY_2                     : int  2 2 0 0 0 0 0 -1 0 -2 ...
 $ PAY_3                     : int  -1 0 0 0 -1 0 0 -1 2 -2 ...
 $ PAY_4                     : int  -1 0 0 0 0 0 0 0 0 -2 ...
 $ PAY_5                     : int  -2 0 0 0 0 0 0 0 0 -1 ...
 $ PAY_6                     : int  -2 2 0 0 0 0 0 -1 0 -1 ...
 $ BILL_AMT1                 : int  3913 2682 29239 46990 8617 64400 367965 11876 11285 0 ...
 $ BILL_AMT2                 : int  3102 1725 14027 48233 5670 57069 412023 380 14096 0 ...
 $ BILL_AMT3                 : int  689 2682 13559 49291 35835 57608 445007 601 12108 0 ...
 $ BILL_AMT4                 : int  0 3272 14331 28314 20940 19394 542653 221 12211 0 ...
 $ BILL_AMT5                 : int  0 3455 14948 28959 19146 19619 483003 -159 11793 13007 ...
 $ BILL_AMT6                 : int  0 3261 15549 29547 19131 20024 473944 567 3719 13912 ...
 $ PAY_AMT1                  : int  0 0 1518 2000 2000 2500 55000 380 3329 0 ...
 $ PAY_AMT2                  : int  689 1000 1500 2019 36681 1815 40000 601 0 0 ...
 $ PAY_AMT3                  : int  0 1000 1000 1200 10000 657 38000 0 432 0 ...
 $ PAY_AMT4                  : int  0 1000 1000 1100 9000 1000 20239 581 1000 13007 ...
 $ PAY_AMT5                  : int  0 0 1000 1069 689 1000 13750 1687 1000 1122 ...
 $ PAY_AMT6                  : int  0 2000 5000 1000 679 800 13770 1542 1000 0 ...
 $ default payment next month: int  1 1 0 0 0 0 0 0 0 0 ...
 - attr(*, ".internal.selfref")=<externalptr> 
```  

```{r}
 LIMIT_BAL SEX EDUCATION MARRIAGE AGE PAY_0 PAY_2 PAY_3 PAY_4 PAY_5 PAY_6 BILL_AMT1 BILL_AMT2
1:     20000   2         2        1  24     2     2    -1    -1    -2    -2      3913      3102
2:    120000   2         2        2  26    -1     2     0     0     0     2      2682      1725
3:     90000   2         2        2  34     0     0     0     0     0     0     29239     14027
4:     50000   2         2        1  37     0     0     0     0     0     0     46990     48233
5:     50000   1         2        1  57    -1     0    -1     0     0     0      8617      5670
6:     50000   1         1        2  37     0     0     0     0     0     0     64400     57069

 BILL_AMT3 BILL_AMT4 BILL_AMT5 BILL_AMT6 PAY_AMT1 PAY_AMT2 PAY_AMT3 PAY_AMT4 PAY_AMT5 PAY_AMT6
1:       689         0         0         0        0      689        0        0        0        0
2:      2682      3272      3455      3261        0     1000     1000     1000        0     2000
3:     13559     14331     14948     15549     1518     1500     1000     1000     1000     5000
4:     49291     28314     28959     29547     2000     2019     1200     1100     1069     1000
5:     35835     20940     19146     19131     2000    36681    10000     9000      689      679
6:     57608     19394     19619     20024     2500     1815      657     1000     1000      800

   default payment next month
1:                          1
2:                          1
3:                          0
4:                          0
5:                          0
6:                          0
```
The column ‘id’ has no role to play in our analysis. Hence, it is omitted.  

 ```{r}
 ##Droping the unnecessary variable/column
credit=credit[,-1]#There's no use of the column id in our analysis
 ```  
 
Let’s inspect for missing values and converting the variables SEX, EDUCATION, MARRIAGE, and default.payment.next.month in factor variables

```{r}
##Looking for missing value
sum(is.na(credit))#it is observed that there's no missing values

##Changing the name of the variable PAY_0 to PAY_1 and default payment to target
names(credit)
names(credit)[6]="PAY_1"

##Transforming the variables SEX,MARRIAGE,EDUCATION and default payment variables into factors

df=as.data.frame(credit)
df[c("SEX","MARRIAGE","EDUCATION","default payment next month")]=lapply(df[c("SEX","MARRIAGE","EDUCATION","default payment next month")]
                                                                        ,function(x) as.factor(x))

credit=df
rm(df)
```

The overall summary of the variables, quantitative as well as qualitative.


```{r}
##Extended data dictionary
summary(credit)
```
 
 ```{r}
   LIMIT_BAL       SEX       EDUCATION MARRIAGE       AGE            PAY_1             PAY_2             PAY_3             PAY_4        
 Min.   :  10000   1:11888   0:  468   3:  377   Min.   :21.00   Min.   :-2.0000   Min.   :-2.0000   Min.   :-2.0000   Min.   :-2.0000  
 1st Qu.:  50000   2:18112   1:10585   1:13659   1st Qu.:28.00   1st Qu.:-1.0000   1st Qu.:-1.0000   1st Qu.:-1.0000   1st Qu.:-1.0000  
 Median : 140000             2:14030   2:15964   Median :34.00   Median : 0.0000   Median : 0.0000   Median : 0.0000   Median : 0.0000  
 Mean   : 167484             3: 4917             Mean   :35.49   Mean   :-0.0167   Mean   :-0.1338   Mean   :-0.1662   Mean   :-0.2207  
 3rd Qu.: 240000                                 3rd Qu.:41.00   3rd Qu.: 0.0000   3rd Qu.: 0.0000   3rd Qu.: 0.0000   3rd Qu.: 0.0000  
 Max.   :1000000                                 Max.   :79.00   Max.   : 8.0000   Max.   : 8.0000   Max.   : 8.0000   Max.   : 8.0000  
     PAY_5             PAY_6           BILL_AMT1         BILL_AMT2        BILL_AMT3         BILL_AMT4         BILL_AMT5     
 Min.   :-2.0000   Min.   :-2.0000   Min.   :-165580   Min.   :-69777   Min.   :-157264   Min.   :-170000   Min.   :-81334  
 1st Qu.:-1.0000   1st Qu.:-1.0000   1st Qu.:   3559   1st Qu.:  2985   1st Qu.:   2666   1st Qu.:   2327   1st Qu.:  1763  
 Median : 0.0000   Median : 0.0000   Median :  22382   Median : 21200   Median :  20089   Median :  19052   Median : 18105  
 Mean   :-0.2662   Mean   :-0.2911   Mean   :  51223   Mean   : 49179   Mean   :  47013   Mean   :  43263   Mean   : 40311  
 3rd Qu.: 0.0000   3rd Qu.: 0.0000   3rd Qu.:  67091   3rd Qu.: 64006   3rd Qu.:  60165   3rd Qu.:  54506   3rd Qu.: 50191  
 Max.   : 8.0000   Max.   : 8.0000   Max.   : 964511   Max.   :983931   Max.   :1664089   Max.   : 891586   Max.   :927171  
   BILL_AMT6          PAY_AMT1         PAY_AMT2          PAY_AMT3         PAY_AMT4         PAY_AMT5           PAY_AMT6       
 Min.   :-339603   Min.   :     0   Min.   :      0   Min.   :     0   Min.   :     0   Min.   :     0.0   Min.   :     0.0  
 1st Qu.:   1256   1st Qu.:  1000   1st Qu.:    833   1st Qu.:   390   1st Qu.:   296   1st Qu.:   252.5   1st Qu.:   117.8  
 Median :  17071   Median :  2100   Median :   2009   Median :  1800   Median :  1500   Median :  1500.0   Median :  1500.0  
 Mean   :  38872   Mean   :  5664   Mean   :   5921   Mean   :  5226   Mean   :  4826   Mean   :  4799.4   Mean   :  5215.5  
 3rd Qu.:  49198   3rd Qu.:  5006   3rd Qu.:   5000   3rd Qu.:  4505   3rd Qu.:  4013   3rd Qu.:  4031.5   3rd Qu.:  4000.0  
 Max.   : 961664   Max.   :873552   Max.   :1684259   Max.   :896040   Max.   :621000   Max.   :426529.0   Max.   :528666.0  
 default payment next month
 0:23364                   
 1: 6636           
 ```  
 
#### **_Bivariate analysis_**

Now we’ll scrutinize the correlations between the quantitative variables and will check if there are high correlation between some of the features. We employ correlation step plot.

```Rscript
##Correlation analysis and Correlogram plot

df=credit[,c(-2:-4,-24)] #Considering only quantitative variables
#cor(credit[,c(-2:-4,-24)])
#ggpairs(df,aes(color=credit$`default payment next month`),title = "Correlogram")
ggcorr(df,method=c("everything", "pearson"))+ggtitle("Correlation Steps")

```

![plot 1](images/plot_2)

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
