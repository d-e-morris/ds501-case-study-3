splits = SplitData(rawData)
train = splits$train
test = splits$test
trainFact = targetFact[train$ID]
testFact = targetFact[test$ID]


tree = rpart(target ~ ., data=(train %>% select(!ID)), method="class")
p = predict(tree,(test %>% select(!ID)), type="class")
print(confusionMatrix(p,testFact))

treeNoPay1 = rpart(target ~ ., data=(train %>% select(!ID & !PAY_1)), method="class")
pNoPay1 = predict(treeNoPay1,(test %>% select(!ID & !PAY_1)), type="class")
print(confusionMatrix(pNoPay1,testFact))

# let try making it better by first filtering
trainPay1 = train %>% filter(PAY_1 > 1.5)
testPay1 = test %>% filter(PAY_1 > 1.5)
testFactPay1 = targetFact[testPay1$ID]
treePay1 = rpart(target ~ ., data=(trainPay1 %>% select(!ID & !PAY_1)),method="class")
pPay1 = predict(treePay1,(testPay1 %>% select(!ID & !PAY_1)),type="class")
print("Confusion Matrix on Pay 1 greater than 1.5")
print(confusionMatrix(pPay1,testFactPay1))

trainNoPay1 = train %>% filter(PAY_1 < 1.5)
testNoPay1 = test %>% filter(PAY_1 < 1.5)
testFactNoPay1 = targetFact[testNoPay1$ID]
treeNoPay1 = rpart(target ~ ., data=(trainNoPay1 %>% select(!ID & !PAY_1)),method="class")
pNoPay1 = predict(treeNoPay1,(testNoPay1 %>% select(!ID & !PAY_1)),type="class")
print("Confusion Matrix on Pay 1 less than 1.5")
print(confusionMatrix(pNoPay1,testFactNoPay1))

combinedPay1 = factor(c(as.integer(pPay1),as.integer(pNoPay1)))
combineTest = factor(c(as.integer(testFactPay1),as.integer(testFactNoPay1)))
print(confusionMatrix(combinedPay1,combineTest))