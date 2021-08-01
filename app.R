library(shiny)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(caret)
library(dplyr)
library(class)
library(ISLR)
library(PerformanceAnalytics)
data = readRDS(file="./data/processed.Rda")
targetFact = readRDS(file="./data/targetFact.Rda")

ui <- fluidPage(
    
    titlePanel("DS501 - Case Study 3: Predicting Credit Default"),
    
    fluidRow("The data set contains selected non-identifying basic financial information for customer of a large Taiwanese credit card issuer.
             The intent of the data is to use that information to classify whether or not a given customer will default next month. The data
             was collected over a six-month period in 2005 from April to September and the response variable indicates the default status in
             October of 2005. The data contains the following information:", HTML("<ul><li>Limit Balance - The credit limit of the customer</li>
             <li>Gender - The customer's gender</li><li>Education - The highest level of education received ranging from graduate to high school
             and extra category for other</li><li>Marital Status - The customer's current marital status</li><li>Age - The customer's age as of 
             October 2005</li><li>Payment History - An indicator variable for the customer's payment status for each of the prior six months</li>
             <li>Bill Statement - The customer's bill statement (i.e. balance due) for each month</li><li>Amount Paid - The amount the customer
             paid each month</li><li>Default Status - The binary response variable indicating if the customer defaulted in the month of October
            </li></ul>"),
             p("The objective of this project is to classify whether a given customer will default or not based on a set of input features. 
               As the goal is to classify a user into one of two groups, using machine learning classification methods is the focus of this
               project. The project will employ three different supervised classification techniques and even allow the user to combine the 
               outputs to create a rudimentary ensemble method.")),
            p("The data's raw features were used to create several synthetic features that are hypothesized to aid in predictive value. The
              categorical variables like education level and marital status were converted into their one-hot encoding as this way they can
              serve as indicator variables instead of having the relationship between each identifier explored. For example, the nominal value
              and difference between the categorical values has no predictive value and can actually hinder the predictive capability of the
              model because it can potentially find false releationships. In addition, summary information about the number of missed payment
              or sum of the payment status was created as it was hypothesized that a person who is more likely to default would have more missed
              payments and longer periods of missed payments. These engineered features are included in the synthetic features secture of the side
              bar."),
            p("Three classification models were explored as well as a rudimentary ensemble method. The three models were included so that their
              performance could be compared and contrasted as well as when they are all combined in some way. Each model takes the selected
              features in as an input without any further modification to them to allow them to be compared more easily. The output of each
              model is represented in its corresponding confusion matrix. The use of a confusion matrix was selected as it is able to convey
              a lot of information about the performance of the data. The decision tree also includes a graphical representation of the decision
              tree, but the other two models do not as it was not feasible to visually represent their structure due to the high dimensionality.
              The confusion matrix strikes a good balance of quickly conveying meaningful information while also allowing the reader to dig deeper.
              Other representations like receiver operator curves were considered but were not used as I felt it did not convey the necessary
              information as well. One important aspect of the data is that the distribution of data between the two classes is heavily skewed
              and it is important to reference the information free accuracy rate when considering the accuracy of each model. In addition,
              the confusion matrix allows one to see where in the algorithm is getting things incorrect via the false positive and false negative
              break downs."),
    
    sidebarPanel(
        
        #Feature Selection
        checkboxGroupInput("rawFeatures","Original Features to Use:",
                           choices = c("Balance Limit"="LIMIT_BAL",
                             "Gender"="SEX",
                             "Education"="EDUCATION",
                             "Marriage Status"="MARRIAGE",
                             "Age" = "AGE",
                             "Payment Status - September"="PAY_1",
                             "Payment Status - August"="PAY_2",
                             "Payment Status - July"="PAY_3",
                             "Payment Status - June"="PAY_4",
                             "Payment Status - May"="PAY_5",
                             "Payment Status - April"="PAY_6",
                             "Bill Statement - September"="BILL_AMT1",
                             "Bill Statement - August"="BILL_AMT2",
                             "Bill Statement - July"="BILL_AMT3",
                             "Bill Statement - June"="BILL_AMT4",
                             "Bill Statement - May"="BILL_AMT5",
                             "Bill Statement - April"="BILL_AMT6",
                             "Payment Amount - September"="PAY_AMT1",
                             "Payment Amount - August"="PAY_AMT2",
                             "Payment Amount - July"="PAY_AMT3",
                             "Payment Amount - June"="PAY_AMT4",
                             "Payment Amount - May"="PAY_AMT5",
                             "Payment AMount - April"="PAY_AMT6"),
                           selected = c("LIMIT_BAL",
                                        "SEX",
                                        "EDUCATION",
                                        "MARRIAGE",
                                        "AGE",
                                        "PAY_1",
                                        "PAY_2",
                                        "PAY_3",
                                        "PAY_4",
                                        "PAY_5",
                                        "PAY_6",
                                        "BILL_AMT1",
                                        "BILL_AMT2",
                                        "BILL_AMT3",
                                        "BILL_AMT4",
                                        "BILL_AMT5",
                                        "BILL_AMT6",
                                        "PAY_AMT1",
                                        "PAY_AMT2",
                                        "PAY_AMT3",
                                        "PAY_AMT4",
                                        "PAY_AMT5",
                                        "PAY_AMT6")),
        
        #Synthetic Feature Selection
        checkboxGroupInput("synthFeatures","Synthetic Features to Use:",
                         choices = c("Education - One hot encoding"="edHot",
                                     "Marriage - One hot encoding"="marHot",
                                     "Bill-to-Limit Ratio"="billratio",
                                     "Sum of Payment Status"="sumPayStatus",
                                     "Count of Months with Delinquent Patment"="countPayStatusDelinq"),
                         selected = c("edHot",
                                      "marHot",
                                      "billratio",
                                      "sumPayStatus",
                                      "countPayStatusDelinq")),
  
        p(strong("Model Selection")),      
        #Setup for KNN
        checkboxInput("knnEnable","k Nearest Neighbors",value=TRUE),
        conditionalPanel(condition = "input.knnEnable",
                         numericInput("knnK","Number of Neighbors",3,min=1,max=100,step=1)),
        
        #Setup for Logistic Regression
        checkboxInput("logEnable","Logistic Regression",value=TRUE),
        conditionalPanel(condition = "input.logEnable",
                         numericInput("logThresh","Classification Threshold",0.5,min=0.01,max=0.99,step=0.01)),
        
        #Setup for Decision Tree
        checkboxInput("treeEnable","Classification Tree",value=TRUE),
        
        #Setup for Combined
        checkboxInput("ensEnable","Ensemble of kNN and Logistic",value=TRUE),
        conditionalPanel(condition = "input.ensEnable",
                         numericInput("knnWeight","kNN Weight",0.33,min=0,max=1,step=0.01),
                         numericInput("logWeight","Logistic Weight",0.33,min=0,max=1,step=0.01),
                         numericInput("treeWeight","Classification Tree Weight",0.34,min=0,max=1,step=0.01),
                         numericInput("ensThresh","Ensemble Threshold",0.5,min=0.01,0.99,step=0.01))
        
    ),
    
    mainPanel(
        h3("K Nearest Neighbors"),
        p("The k nearest neighbors alogrithm is a supervised, instance-based machine learning algorithm that determines 
          the classification of a given input based on the classifications of known data points that are most similar to it in terms of some measure.
          Normally, the measure of similarity uses the euclidean distance from the data point of interest to all the other
          points. Then the top k points are chosen. However, other measures of similarity can be used, for example, manhattan
          distances or a custom function can be used. Once the top number of points are located, the majority classification
          of known top points is assigned to the unknown point. The number of top points to use is a hyper parameter that is
          often tuned or searched in the training phase of model development. For this project a basic implementation of k
          nearest neighbors was used where the euclidean distance is used to identify the top k closest points and then it is 
          classified according to the vote."),
        verbatimTextOutput("kNNOutput"),
        h3("Logistic Regression"),
        p("The logistic regression is a regression algorithm that is used to the determine the log-odds of a binary dependent variable. The model
          assumes a linear relationship between the log-odds of the response and outputs a value between 0 and 1. The logistic regression is fit 
          using the same least sum of squares procedure as a simple linear regression which tries to minimize the overall error. The least squares
          error determines the coefficients of the function that minimizes the error. A logistic 
          regression is great for mapping the relationship between multivariate inputs and a binary response when the probability that outcome is 
          an important function."),
        p("One advantage of working withn the regression family is that complex relationships within the data can be easily modeled and interpretted. For example,
          conditional inputs like considering the ratio of balance to credit limit for only those customers that are married can be done by simply multiplying the
          one-hot married indicator variable with the ratio of credit limit. In this scenario, when the one-hot indicator is not true (i.e. zero) it will not
          contribute to the response even if the ratio were non-zero. While this implementation does not implement such complex modeling it is something that
          could be explored in future work."),
        verbatimTextOutput("LogisticOutput"),
        h3("Classification Tree"),
        p("A decision tree is another simple model that is used in supervised classification. The tree is constructed recursively partition the data that, 
          at each iteration, maximizes the homogeneity of the training data set. In each iteration the alrogithm starts by sorting the training data on 
          each feature based. Once each feature is sorted, a split is determined for each feature that minimizes the diversity (or maximizes the homogeneity)
          of the response variable for each feature. This is accomplished by checking the homogeneity of the feature at a value that is in between eachs succesive
          pair of points. The boundary that maximizes the homogeneity is used for that particular feature. Then, the feature that has the greatest homogeneity is
          selected for that iteration and the data is split into new partitions. The process then recurses and iterates on the newly created partitions and is
          repeated until a full tree is reached with complete homogeneity in each leaf node or some other stopping condition is reached. In the case of the
          method used for rpart, the algorithm terminates either when uniformity is reached or when the complexity of creating a new level drops below a certain
          threshold. Implementing the complexity parameter is a penalty system used to prevent a decision tree from overfitting to the training data in effort
          to focus the tree on learning a general structure as opposed to merely memorizing the data."),
        plotOutput("treePlot"),
        verbatimTextOutput("treeOutput"),
        h3("Ensemble Method"),
        p("Ensemble methods take on many forms and can be used effectively to increase generalized learning while keeping overfitting low. The core concept
          of ensembling is combining many algorithms and models together to create one prediction that is better than the sum of its parts. One example of
          ensembling is explored here which takes three different classification techniques and uses simple weighted average voting with a threshold to assign
          an output. While this method is simple and straightforward it is not optimal and there are many different techniques that could perform better. However,
          this technique has been used because of its simplicity in demonstating the concept of ensembling."),
        p("A more common ensembling method is the use of boosting or bootstrapping/bagging decision trees, creating random forests by pruning or feature limitations,
          among others. Boosting is an iterative model where each successive decision tree attempts to build upon the errors of the previous tree to increase the
          accuracy. Bootstrapping and bagging is a combined technique where many trees are trained on randomly sampled (with replacement) subsets of the training
          data. The outputs of this bag of trees is then combined in some manner to create an output. A simple method of the combining output could be simple voting
          but other algoritms or strategies could be employed. Another form of ensembling could be using one algorithm to select which subsequent algoritm should be
          used to make the prediction. This method works under the assumption that certain algorithms strengths can be leveraged where it fits the data best.
          Classifiers or clustering methods can be used to perform the selection. Ensembling is a wide and exciting field of machine learning and there are many
          great techniques. Zhi-Hua Zhou's Ensemble Methods: Foundations and Algorithms is a fantastic introduction and overview of the field."),
        verbatimTextOutput("ensembleOutput")
    )
    
    
)

SplitData = function(data, trainPer=0.7,testPer=0.2,valPer=0.1) {
  trainInd = sort(sample(nrow(data),nrow(data)*0.7))
  train = data[trainInd,]
  testAndVal = data[-trainInd,]
  testInd = sort(sample(nrow(testAndVal),nrow(testAndVal)*(testPer/(testPer+valPer))))
  test = testAndVal[testInd,]
  val = testAndVal[-testInd,]
  return(list("train"=train,"test"=test,"val"=val))
}


server <- function(input, output, session) {
  # Create instructions
  output$Information = renderText({if(length(input$rawFeatures)==0){typeof(input$rawFeatures)} else {input$rawFeatures}})
  
  # Create filtered list of the data to use
  
  rawData = reactive({
    input$rawFeatures 
  })
  
  selectedFeatures = reactive({
    newFeatures = rawData()
    if("edHot" %in% input$synthFeatures) {
      newFeatures = c(newFeatures,"GRAD")
      newFeatures = c(newFeatures,"UNI")
      newFeatures = c(newFeatures,"HIGH")
    }
    if("marHot" %in% input$synthFeatures) {
      newFeatures = c(newFeatures,"Mar")
      newFeatures = c(newFeatures,"Sin")
    }
    if("billratio" %in% input$synthFeatures) {
      newFeatures = c(newFeatures,"RATIO_BILL_1")
      newFeatures = c(newFeatures,"RATIO_BILL_2")
      newFeatures = c(newFeatures,"RATIO_BILL_3")
      newFeatures = c(newFeatures,"RATIO_BILL_4")
      newFeatures = c(newFeatures,"RATIO_BILL_5")
      newFeatures = c(newFeatures,"RATIO_BILL_6")
    }
    if("sumPayStatus" %in% input$synthFeatures) {
      newFeatures = c(newFeatures,"sumPayStatus")
    }
    if("countPayStatusDeling" %in% input$synthFeatures) {
      newFeatures = c(newFeatures,"countPayStatusDelinq")
    }
    return(newFeatures)
  })
  
  selectedData = reactive({
    data %>% select(one_of(c(selectedFeatures(),"target","ID")))
  })
  
  # Create the render for kNN
  output$kNNOutput = renderPrint(
  {
    if(length(selectedFeatures())!=0 & input$knnEnable){
      splits=SplitData(selectedData(),train=0.7,test=0.3,valPer=0)
      train = splits$train
      test = splits$test
      trainFact = targetFact[train$ID]
      testFact = targetFact[test$ID]
      knnpred = knn((train %>% select(!ID & !target)),(test %>% select(!ID & !target)),
                    trainFact,k=input$knnK,prob=TRUE)
      confusionMatrix(knnpred,testFact)
    } else {
      if(!input$knnEnable){
        "K Nearest Neighbors Not Enabled"
      } else if(length(selectedFeatures())==0) {
        "No Features Selected"
      }
    }
  })
  
  # Create the render for Logistic Regression
  output$LogisticOutput = renderPrint({
    if(length(selectedFeatures())!=0 & input$logEnable){
      splits=SplitData(selectedData(),train=0.7,test=0.3,valPer=0)
      train = splits$train
      test = splits$test
      trainFact = targetFact[train$ID]
      testFact = targetFact[test$ID]
      glm.fit = glm(target ~ ., data=(train %>% select(!ID)), family=binomial)
      glm.probs = predict(glm.fit,(test %>% select(!ID & !target)),type="response")
      glm.pred = factor(ifelse(glm.probs>input$logThresh,1,0))
      confusionMatrix(glm.pred,testFact)
    } else {
      if(!input$logEnable){
        "Logistic Regression Not Enabled"
      }else if(length(selectedFeatures())==0) {
        "No Features Selected"
      }
    }
  })
  
  output$treePlot = renderPlot({
    if(length(selectedFeatures()) != 0 & input$ensEnable) {
      splits=SplitData(selectedData(),train=0.7,test=0.3,valPer=0)
      train = splits$train
      test = splits$test
      trainFact = targetFact[train$ID]
      testFact = targetFact[test$ID]
      tree = rpart(target ~ ., data=(train %>% select(!ID)), method="class")
      rpart.plot(tree)
    } else {
      if(!input$ensEnable){
        "Classification Tree is Not Enabled"
      } else if (length(selectedFeatures())==0) {
        "No Features Selected"
      }
    }
  })
  
  output$treeOutput = renderPrint({
    if(length(selectedFeatures()) != 0 & input$ensEnable) {
      splits=SplitData(selectedData(),train=0.7,test=0.3,valPer=0)
      train = splits$train
      test = splits$test
      trainFact = targetFact[train$ID]
      testFact = targetFact[test$ID]
      tree = rpart(target ~ ., data=(train %>% select(!ID)), method="class")
      p = predict(tree,(test %>% select(!ID)), type="class")
      confusionMatrix(p,testFact)
    } else {
      if(!input$ensEnable){
        "Classification Tree is Not Enabled"
      } else if (length(selectedFeatures())==0) {
        "No Features Selected"
      }
    }
  })
  
  output$ensembleOutput = renderPrint({
    if(length(selectedFeatures()) != 0 & input$ensEnable) {
      splits=SplitData(selectedData(),train=0.7,test=0.3,valPer=0)
      train = splits$train
      test = splits$test
      trainFact = targetFact[train$ID]
      testFact = targetFact[test$ID]
      knnpred = knn((train %>% select(!ID & !target)),(test %>% select(!ID & !target)),
                    trainFact,k=input$knnK,prob=TRUE)
      glm.fit = glm(target ~ ., data=(train %>% select(!ID)), family=binomial)
      glm.probs = predict(glm.fit,(test %>% select(!ID & !target)),type="response")
      glm.pred = ifelse(glm.probs>input$logThresh,1,0)
      tree = rpart(target ~ ., data=(train %>% select(!ID)), method="class")
      p = predict(tree,(test %>% select(!ID)), type="class")
      #c("Length of knn: ",length(knnpred),"\nLength of log: ",length(glm.pred))
      combined = factor(ifelse((input$knnWeight*(as.integer(knnpred)-1)+
                                 input$logWeight*glm.pred +
                                  input$treeWeight*(as.integer(p)-1)) > input$ensThresh,1,0))
      confusionMatrix(combined,testFact)
    } else {
      if(!input$ensEnable){
        "Ensemble Method is Not Enabled"
      } else if (length(selectedFeatures())==0) {
        "No Features Selected"
      }
    }
    
  })
  
  observeEvent(input$knnWeight, {
    if(!input$knnEnable){
      updateNumericInput(session,"knnWeight",value=0)
    }
    total = input$knnWeight + input$logWeight + input$treeWeight
    if (total != 1){
      updateNumericInput(session, "logWeight", value=input$logWeight/total)
      updateNumericInput(session, "treeWeight", value=input$treeWeight/total)
    }
  })
  
  observeEvent(input$knnEnable, {
    if(!input$knnEnable){
      updateNumericInput(session,"knnWeight",value=0)
    }
    total = input$knnWeight + input$logWeight + input$treeWeight
    if (total != 1){
      updateNumericInput(session, "logWeight", value=input$logWeight/total)
      updateNumericInput(session, "treeWeight", value=input$treeWeight/total)
    }
  })
  
  observeEvent(input$logWeight, {
    if(!input$logEnable){
      updateNumericInput(session,"logWeight",value=0)
    }
    total = input$knnWeight + input$logWeight + input$treeWeight
    if (total != 1){
      updateNumericInput(session, "knnWeight", value=input$knnWeight/total)
      updateNumericInput(session, "treeWeight", value=input$treeWeight/total)
    }
  })
  
  observeEvent(input$logEnable, {
    if(!input$logEnable){
      updateNumericInput(session,"logWeight",value=0)
    }
    total = input$knnWeight + input$logWeight + input$treeWeight
    if (total != 1){
      updateNumericInput(session, "knnWeight", value=input$knnWeight/total)
      updateNumericInput(session, "treeWeight", value=input$treeWeight/total)
    }
  })
  
  observeEvent(input$treeWeight, {
    if(!input$treeEnable){
      updateNumericInput(session,"treeWeight",value=0)
    }
    total = input$knnWeight + input$logWeight + input$treeWeight
    if (total != 1){
      updateNumericInput(session, "logWeight", value=input$logWeight/total)
      updateNumericInput(session, "knnWeight", value=input$knnWeight/total)
    }
  })
  
  observeEvent(input$treeEnable, {
    if(!input$treeEnable){
      updateNumericInput(session,"treeWeight",value=0)
    }
    total = input$knnWeight + input$logWeight + input$treeWeight
    if (total != 1){
      updateNumericInput(session, "logWeight", value=input$logWeight/total)
      updateNumericInput(session, "knnWeight", value=input$knnWeight/total)
    }
  })
}




shinyApp(ui, server)
