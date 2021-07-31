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
        verbatimTextOutput("kNNOutput"),
        h3("Logistic Regression"),
        verbatimTextOutput("LogisticOutput"),
        h3("Classification Tree"),
        plotOutput("treePlot"),
        verbatimTextOutput("treeOutput"),
        h3("Ensemble Method"),
        verbatimTextOutput("ensembleOutput")
    ),
    
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
               project. The project will employ two different supervised classification techniques and even allow the user to combine the 
               outputs to create a rudimentary ensemble method."))
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
      #c("Length of knn: ",length(knnpred),"\nLength of log: ",length(glm.pred))
      combined = factor(ifelse(input$knnWeight*(as.integer(knnpred)-1)+input$logWeight*glm.pred > input$ensThresh,1,0))
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
