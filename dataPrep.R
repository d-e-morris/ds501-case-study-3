library(ggplot2)
library(rpart)
library(rpart.plot)
library(caret)
library(dplyr)
library(class)
library(ISLR)
# This file is the data processing portion of the project. The variables will
# be explored and the data will be prepped.

# DATA SOURCE: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients#

SplitData = function(data, trainPer=0.7,testPer=0.2,valPer=0.1) {
  trainInd = sort(sample(nrow(data),nrow(data)*0.7))
  train = data[trainInd,]
  testAndVal = data[-trainInd,]
  testInd = sort(sample(nrow(testAndVal),nrow(testAndVal)*(testPer/(testPer+valPer))))
  test = testAndVal[testInd,]
  val = testAndVal[-testInd,]
  return(list("train"=train,"test"=test,"val"=val))
}

#load the data
rawData = data.frame(read.csv("data/cs3_data.csv", header=TRUE))
targetFact = factor(rawData$target)



# The dataset used is customer information around credit worthiness. It is used 
# to determine if a customer is likely to default on. The data has 30000 data
# data points. It is all integer values and does not have any missing values.
# the dataset consists of 23 raw features. The features are
# 1. Combined credit limit of the individual and the family (supplementary)
# 2. Gender
# 3. Education (1 = Grad school; 2= University; 3= high school; 4 = other)
# 4. Marital status (1=married; 2=single; 3=other)
# 5. Customer age in years
# 6 - 11. Payment history over the past 6 months (6 = most recent, 11= oldest)
# 12 - 17. Bill statement (12 = most recent)
# 18 - 23. Amount paid for last month's bill

# The response is whether or not the customer defaulted and the goal of this
# project is also to classify if a given customer will default. There may be
# an aim to assign a "probability of default" but only as a later step. 
# Classifying a customer as likely to default or not is best solved using
# supervised classification techniques like KNN, Decision Trees, Logistic
# Regressions, and perhaps association rules if appropriate features can be 
# generated.

# The problem is interesting to me because the credit worthiness is somewhat of 
# a black box to me and diving into the data to understand what influences credit
# worthiness seems like a great way to learn about what a lender would consider.
# I am very interested to see what features will have the greatest predictive 
# value. In addition, this is interesting because it solves a real world problem
# and it would be cool, as an independent project, to see if there is any data
# around the effectiveness of intervention methods to keep identified customers 
# from defaulting. 

# Before looking at the data some initial thoughts around what might be useful
# as an indicator for default would be an increasing bill statement or customers
# with a small ratio between the bill statement and the amount paid. In addition,
# things like the ratio between the bill statement and credit limit may also
# provide predictive value. The descriptive values may also have predictive power
# on their own but I believe care will need to be taken with the categorical
# variables that have the catchall of "other" as this may, in conjunction with
# other features, have predictive value but may not be obvious as surface level.

# Start by getting general summary information of the data.
summary(train)
#Plot the various data columns to see if there are trends
# for limit balance the data is largely skewed to the bottom wiht the vast majority having less than 250,000 in a limit
# for education most of have high school or college level education. What is interesting is that the data 
# also seems to imply that the other category also has values of 5 and 6 (maybe some kind of error or extra info)
# for marriage most people are singly followed closely behind by single, other only has a few
# for age it is a skewed distribution but probably follows the general population breakdown for taiwan with 
# most people being 40 or younger
# the data for the pay status also has some inconsistencies that will need to be addressed. Mostly the fact that -2 shows up in the data
# but does not correspond to a value
# bill amount also shows an expected curve that mostly matches the curve for credit limit
# pay amount interestingly shows a much smaller curve


# Look at the response most of the people did not default. About 22% of the customers did so a prediction accuracy greater than 88%
# is the minimum threshold for success. 

# Moving on lets take a look and see if any of the variables are highly correlated.
#cormat = round(cor(train),2)
#melted_cormat = melt(cormat)
#ggplot(data=melted_cormat, aes(x=Var1, y=Var2, fill=value)) + geom_tile()
# looking at the data it is clear that the various group fields are highly correlated
# which makes sense, you're likely to have the same amount of credit card balance each
# month and the same with payments. Looking at correlation with the response
# the only thing that seems relatively correlated is the most recent payment status
# and limit balance. The rest are only weakly correlated at best.

# Move on to feature engineering to better prepare the data. To start, something should
# be done to address the errors in the data. There are three options as to what to do.
# the data could simply be left in there and assume it has some specific meaning not 
# addressed by the creators of the dataset. However, a better choice would be to either
# remove the erroneous elements or attempt to reclassify them. Given my limited ability
# to seek clarification from the original creators and the relatively large amount of 
# samples availabe. It is probably best to simply filter those values out.

# filter out the extra categories on education
rawData = rawData[rawData$EDUCATION<5,]
rawData = rawData[rawData$PAY_1>-2,]
rawData = rawData[rawData$PAY_2>-2,]
rawData = rawData[rawData$PAY_3>-2,]
rawData = rawData[rawData$PAY_4>-2,]
rawData = rawData[rawData$PAY_5>-2,]
rawData = rawData[rawData$PAY_6>-2,]

#convert the categorical variables into one hot encoding
rawData$GRAD = rawData$EDUCATION==1
rawData$UNI = rawData$EDUCATION==2
rawData$HIGH = rawData$EDUCATION==3
rawData$Mar = rawData$MARRIAGE==1
rawData$Sin = rawData$MARRIAGE==2


# check how that affects correlations
#cormat = round(cor(train),2)
#melted_cormat = melt(cormat)
#ggplot(data=melted_cormat, aes(x=Var1, y=Var2, fill=value)) + geom_tile()
# There were no significant changes in the correlations.

# From this point on the rawData will be left alone and changes will only be made
# in the training data which means transforms will need to also be made in the other
# buckets of data later
# The next phase of feature engineering is to start generating synthetic features
# based on some of the previous hypotheses that might affect the response. The
# first synthetic feature is to look at the ratio between the balance and the limit
# for each month
rawData$RATIO_BILL_1 = rawData$BILL_AMT1/rawData$LIMIT_BAL
rawData$RATIO_BILL_2 = rawData$BILL_AMT2/rawData$LIMIT_BAL
rawData$RATIO_BILL_3 = rawData$BILL_AMT3/rawData$LIMIT_BAL
rawData$RATIO_BILL_4 = rawData$BILL_AMT4/rawData$LIMIT_BAL
rawData$RATIO_BILL_5 = rawData$BILL_AMT5/rawData$LIMIT_BAL
rawData$RATIO_BILL_6 = rawData$BILL_AMT6/rawData$LIMIT_BAL
#cormat = round(cor(rawData),2)
#melted_cormat = melt(cormat)
#ggplot(data=melted_cormat, aes(x=Var1, y=Var2, fill=value)) + geom_tile()
# While not strong there is a small amount of positive correlation with a high ratio

# Another feature to look at is the deltas between the months of each economic category
rawData$delta_PAY_1 = rawData$PAY_1 - rawData$PAY_2
rawData$delta_PAY_2 = rawData$PAY_2 - rawData$PAY_3
rawData$delta_PAY_3 = rawData$PAY_3 - rawData$PAY_4
rawData$delta_PAY_4 = rawData$PAY_4 - rawData$PAY_5
rawData$delta_PAY_5 = rawData$PAY_5 - rawData$PAY_6

rawData$delta_BILL_1 = rawData$BILL_AMT1 - rawData$BILL_AMT2
rawData$delta_BILL_2 = rawData$BILL_AMT2 - rawData$BILL_AMT3
rawData$delta_BILL_3 = rawData$BILL_AMT3 - rawData$BILL_AMT4
rawData$delta_BILL_4 = rawData$BILL_AMT4 - rawData$BILL_AMT5
rawData$detla_BILL_5 = rawData$BILL_AMT5 - rawData$BILL_AMT6
#cormat = round(cor(rawData),2)
#no strong correlations value

# try looking to see if the change in ratio bill over the 6 month period is predictive
rawData$delta_ratio = rawData$RATIO_BILL_1 - rawData$RATIO_BILL_6

#cormat = round(cor(train),2)
# while the global correlation is low within specific populations the correalation is high

rawData = rawData %>% mutate(sumPayStatus = PAY_1+PAY_2+PAY_3+PAY_4+PAY_5+PAY_6, 
                             countPayStatusDelinq = sum((PAY_1>0)+(PAY_2>0)+(PAY_3>0)+(PAY_4>0)+(PAY_5>0)+(PAY_6>0)))
saveRDS(rawData,file="./data/processed.Rda")