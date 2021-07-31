# From here it is time to start running some basic tests of various classification
# algorithms. Let's start with a basic decision tree

splits = SplitData(rawData)
train = splits$train
test = splits$test
tree = rpart(target ~ ., data=(train %>% select(!ID)), method="class",cp=0.005)
t = predict(tree,(train %>% select(!ID)),type="class")
trainFact = targetFact[train$ID]
confusionMatrix(t,trainFact)
p = predict(tree,(test %>% select(!ID)), type="class")
testFact = targetFact[test$ID]
confusionMatrix(p,testFact)

knnpred = knn((train %>% select(!ID & !target)),(test %>% select(!ID & !target)),trainFact,k=10,prob=TRUE)
confusionMatrix(knnpred,testFact)

glm.fit = glm(target ~ ., data=(train %>% select(!ID)), family=binomial)
glm.probs = predict(glm.fit,type="response")
glm.pred = factor(ifelse(glm.probs>0.5,1,0))
confusionMatrix(glm.pred,trainFact)

combined = factor(ifelse((glm.probs*(as.integer(t)-1))>0.3,1,0))
confusionMatrix(combined,trainFact)

maxAcc = 0
pos = 0
for (i in seq(0.01,1,0.01)){
  glm.probs = predict(glm.fit,(test %>% select(!ID)),type="response")
  glm.pred = factor(ifelse(glm.probs>i,1,0))
  acc = confusionMatrix(glm.pred,testFact)$overall["Accuracy"]
  if (acc > maxAcc) {
    maxAcc = acc
    pos = i
  }
}
pos
maxAcc

maxAcc = 0
ipos = 0
jpos = 0
glm.probs = predict(glm.fit,(test %>% select(!ID)),type="response")
glm.pred = factor(ifelse(glm.probs>0.1,1,0))
for (i in seq(0.01,1,0.01)){
  output = factor(ifelse((glm.probs*(as.integer(p)-1)*attr(knnpred,"prob"))>i,1,0))
  acc = confusionMatrix(output,testFact)$overall["Accuracy"]
  if(acc > maxAcc) {
    maxAcc = acc
    ipos = i
    jpos = 0
  }
}

pos
maxAcc

trainset = train %>% select(LIMIT_BAL, RATIO_BILL_1, RATIO_BILL_2, RATIO_BILL_3, RATIO_BILL_4, RATIO_BILL_5, RATIO_BILL_6)
testset = test %>% select(LIMIT_BAL, RATIO_BILL_1, RATIO_BILL_2, RATIO_BILL_3, RATIO_BILL_4, RATIO_BILL_5, RATIO_BILL_6)

allmaxAcc = 0
allpos = 0
filtmaxAcc = 0
filtpos = 0
for(i in 1:100){
  knnpred = knn((train %>% select(!ID & !target)),(test %>% select(!ID & !target)),trainFact,k=i,prob=TRUE)
  knnpredset = knn(trainset, testset, trainFact, k=i)
  allacc = confusionMatrix(knnpred, testFact)$overall["Accuracy"]
  acc = confusionMatrix(knnpredset,testFact)$overall["Accuracy"]
  if (allacc > allmaxAcc){
    allmaxAcc = allacc
    allpos = i
  }
  if (acc > filtmaxAcc) {
    filtmaxAcc = acc
    filtpos = 1
  }
}