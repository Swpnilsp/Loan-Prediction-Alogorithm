library(rpart)
library(ggplot2)
library(GGally)
library(randomForest)
library(leaps)
library(gam)
library(dplyr)
library(nnet)
library(caret)
library(verification)
library(maptree)

setwd('/Users/swapnilpatil/Study/MS-Bana/Projects/Loan Prediction')
train<-read.csv('train.csv')

head(train)
dim(train)
colSums(is.na(train))
str(train)

##
train$Loan_Status<-ifelse(train$Loan_Status=='Y',1,0)
train$Loan_Status<-as.factor(train$Loan_Status)
## Categorical Variables
table(train$Gender)
train$Gender<-as.character(train$Gender)
train$Gender[train$Gender=='']<-'Unkown'
train$Gender<-as.factor(train$Gender)
table(train$Gender)
table(train$Married)
table(train$Dependents)

## it is safe to assume that blank dependents are 0 dependets
train$Dependents[train$Dependents=='']<-0
train$Dependents<-as.character(train$Dependents)
train$Dependents<-as.factor(train$Dependents)
table(train$Dependents)



table(train$Married,train$Dependents)
table(train$Education)
table(train$Self_Employed)
table(train$Property_Area)
table(train$Loan_Status)
table(train$Credit_History)
train$Credit_History<-as.factor(train$Credit_History)
table(train$Credit_History)
table(train$Credit_History,train$Loan_Status)
train$Credit_History<-as.numeric(train$Credit_History)
train$Credit_History[is.na(train$Credit_History)]<-3
train$Credit_History<-as.factor(train$Credit_History)
table(train$Credit_History)
table(train$Credit_History,train$Loan_Status)
table(train$Credit_History,train$Education)
table(train$Credit_History,train$Self_Employed)
train$Credit_History<-as.factor(train$Credit_History)

## Quantitative Variables
summary(train$ApplicantIncome)
summary(train$CoapplicantIncome)
summary(train$Loan_Amount_Term)
summary(train$Loan_Amount)
summary(train$Credit_History)


### to check employment
ggplot(data = train,(aes(x = Self_Employed,y = ApplicantIncome)))+geom_boxplot()
ggplot(data = train,(aes(x = Self_Employed,y = CoapplicantIncome)))+geom_boxplot()
train$Self_Employed[train$Self_Employed=='']<-'No'
train$Self_Employed<-as.character(train$Self_Employed)
train$Self_Employed<-as.factor(train$Self_Employed)
table(train$Self_Employed)


## Changing Married status
train$Married[train$Married=='']<-'No'
train$Married<-as.character(train$Married)
train$Married<-as.factor(train$Married)
table(train$Married)

completeTrain<-train[(complete.cases(train)),]
dim(train)

## Adding total income
train$TotalIncome<-train$ApplicantIncome+train$CoapplicantIncome

### EDA
ggplot(data = train,aes(x = Gender,fill=Loan_Status))+geom_bar(stat = 'count',position = 'fill')
ggplot(data = train,aes(x = Married,fill=Loan_Status))+geom_bar(stat = 'count',position = 'fill')
ggplot(data = train,aes(x = Dependents,fill=Loan_Status))+geom_bar(stat = 'count',position = 'fill')
ggplot(data = train,aes(x = Education,fill=Loan_Status))+geom_bar(stat = 'count',position = 'fill')
ggplot(data = train,aes(x = Self_Employed,fill=Loan_Status))+geom_bar(stat = 'count',position = 'fill')
ggplot(data = train,aes(x = Credit_History,fill=Loan_Status))+geom_bar(stat = 'count',position = 'fill')
ggplot(data = train,aes(x = Property_Area,fill=Loan_Status))+geom_bar(stat = 'count',position = 'fill')


ggplot(data = train,aes(Loan_Status,ApplicantIncome))+geom_boxplot()
ggplot(data = train,aes(Loan_Status,CoapplicantIncome))+geom_boxplot()
ggplot(data = train,aes(Loan_Status,LoanAmount))+geom_boxplot()
ggplot(data = train,aes(Loan_Status,Loan_Amount_Term))+geom_boxplot()
ggplot(data = train,aes(Loan_Status,TotalIncome))+geom_boxplot()

### Cost function to decide threshold probability
cost1<-function(r,pi)
{
  return(sum((r==pi)/length(pi)))
}

#### Logistic Regression
log.reg.full<-glm(data = train[,-1],Loan_Status~.,family = binomial)
summary(log.reg.full)

## Probit Link
log.reg.reduced<-glm(data = train[,-1],Loan_Status~Married+Credit_History+Property_Area,family = binomial)
summary(log.reg.reduced)
predicted.p<-predict(log.reg.reduced,type = 'response')


## Logit Link
log.reg.reduced.l<-glm(data = train[,-1],Loan_Status~Married+Credit_History+Property_Area,family = binomial(link="logit"))
summary(log.reg.reduced.l)
predicted.l<-predict(log.reg.reduced.l,type = 'response')


## Clog Link Full
log.reg.reduced.cf<-glm(data = train[,-c(1,9,10)],Loan_Status~.,binomial(link="cloglog"))
summary(log.reg.reduced.cf)
predicted.cf<-predict(log.reg.reduced.cf,type = 'response')

## Clog Link reduced
log.reg.reduced.cr<-glm(data = train[,-1],Loan_Status~Married+Credit_History+Property_Area,binomial(link="cloglog"))
summary(log.reg.reduced.cr)
predicted.cr<-predict(log.reg.reduced.cr,type = 'response')


roc.plot(x = train$Loan_Status == "1", pred = cbind(predicted.p,predicted.l,predicted.cr,predicted.cf),
         legend = TRUE)$roc.vol



# CART
rpart.full<-rpart(data = train[,-1],Loan_Status~.,method = "class")
summary(rpart.full)
plot(rpart.full)
text(rpart.full)

predicted<-predict(rpart.full,type = 'prob')

roc.plot(x = train$Loan_Status == "1", pred = predicted[,2])$roc.vol



## Cart Reduced
rpart.reduced<-rpart(data = train[,-1],Loan_Status~Married+Credit_History+Property_Area,method = "class")
summary(rpart.reduced)
plot(rpart.reduced)
text(rpart.reduced)

predicted<-predict(rpart.reduced,type = 'prob')

roc.plot(x = train$Loan_Status == "1", pred = predicted[,2])$roc.vol


### rf Model
rf.full<-randomForest(data=train[,-c(1,9,10)],Loan_Status~.,ntree=1000)
summary(rf.full)
predicted.rf<-predict(rf.full,type = 'prob')
roc.plot(x = train$Loan_Status == "1", pred = predicted.rf[,2])$roc.vol

### rf Reduced
rf.r<-randomForest(data=trainComplete[,-1],Loan_Status~Married+Credit_History+Property_Area,ntree=20000)
summary(rf.r)
predicted.r<-predict(rf.r,type = 'prob')
roc.plot(x = train$Loan_Status == "1", pred = predicted.r[,2])$roc.vol

## NNet
ntrain<-train[,-c(1,9,10)]
ntrain$ApplicantIncome<-(ntrain$ApplicantIncome-min(ntrain$ApplicantIncome))/(max(ntrain$ApplicantIncome)-min(ntrain$ApplicantIncome))
ntrain$CoapplicantIncome<-(ntrain$CoapplicantIncome-min(ntrain$CoapplicantIncome))/(max(ntrain$CoapplicantIncome)-min(ntrain$CoapplicantIncome))
#ntrain$LoanAmount<-(ntrain$LoanAmount-min(ntrain$LoanAmount))/(max(ntrain$LoanAmount)-min(ntrain$LoanAmount))
#ntrain$Loan_Amount_Term<-(ntrain$Loan_Amount_Term-min(ntrain$Loan_Amount_Term))/(max(ntrain$Loan_Amount_Term)-min(ntrain$Loan_Amount_Term))
ntrain$TotalIncome<-(ntrain$TotalIncome-min(ntrain$TotalIncome))/(max(ntrain$TotalIncome)-min(ntrain$TotalIncome))
summary(ntrain)

str(ntrain)

## NNet Full
nnmodel<-nnet(data=ntrain,Loan_Status~.,size = 10,maxit=10000,decay=0.008)
predicted.nn<-nnmodel$fitted.values
roc.plot(x = ntrain$Loan_Status == "1", pred = predicted.nn)$roc.vol

## NNet Reduced
nnmmodel.red<-nnet(data=ntrain,Loan_Status~Married+Credit_History+Property_Area,size = 8,maxit=100000,decay=0.008)
predicted.nred<-nnmmodel.red$fitted.values
roc.plot(x = ntrain$Loan_Status == "1", pred = predicted.nred)$roc.vol
predict(nnmmodel.red)

## Cross validation to decide optimum size
l<-nrow(ntrain)
cv.size<-NULL
cv.roc<-NULL
for (i in 1:10)
{
  set.seed(i)
  cv.sample<-sample(l,0.8*l,replace = F)
  cv.train<-ntrain[cv.sample,]
  cv.test<-ntrain[-cv.sample,]
  cv.nn<-nnet(data=cv.train,Loan_Status~Married+Credit_History+Property_Area,size = i,maxit=10000,decay=0.008)
  cv.predicted<-predict(cv.nn,cv.test)
  cv.rp<-roc.plot(x = cv.test$Loan_Status == "1", pred = cv.predicted)
  cv.size[i]<-i
  cv.roc[i]<-as.numeric(cv.rp$roc.vol[2])
}
  
## Decay should be 0.008
plot(cv.size,cv.roc)

## NNET bootstrapped
boot.sample<-sample(l,5*l,replace = T)
boot.data<-ntrain[boot.sample,]
boot.nnet<-nnet(data=boot.data,Loan_Status~.,size = 8,maxit=100000,decay=0.008)
boot.train<-predict(boot.nnet,boot.data)
boot.test<-predict(boot.nnet,ntrain)
roc.plot(x = boot.data$Loan_Status == "1", pred = boot.train)$roc.vol
roc.plot(x = ntrain$Loan_Status == "1", pred = boot.test)$roc.vol

## Definint cutoff probability
accuracy<-NULL
p<-NULL
for ( i in 1:1000)
{
  prob<-i/1000
  pred<-ifelse(boot.test<prob,0,1)
  cm<-confusionMatrix(pred,ntrain$Loan_Status)
  accuracy[i]<-as.numeric(cm$overall[1])
  p[i]<-prob
}
## 0.576
View(cbind(p,accuracy))

## Definint cutoff probability for LogReg
accuracy<-NULL
p<-NULL
for ( i in 1:1000)
{
  prob<-i/1000
  pred<-ifelse(predicted.cr<prob,0,1)
  cm<-confusionMatrix(pred,train$Loan_Status)
  accuracy[i]<-as.numeric(cm$overall[1])
  p[i]<-prob
}

plot(p,accuracy)
View(cbind(p,accuracy))

### 0.109

### GBM boosting
fitControl <- trainControl( method = "repeatedcv", number = 5, repeats = 10)
fit <- train(Loan_Status~., data = completeTrain[,-1], method = "gbm", trControl = fitControl,verbose = FALSE)
predicted= predict(fit,type= "prob")[,2] 
roc.plot(x = completeTrain$Loan_Status == "1", pred = predicted)$roc.vol

## XGBoost
TrainControl <- trainControl( method = "repeatedcv", number = 5, repeats = 4)
model<- train(Loan_Status~., data = completeTrain[,-1], method = "xgbTree", trControl = TrainControl,verbose = FALSE)
predicted <- predict(model,type= "prob")[,2]
roc.plot(x = completeTrain$Loan_Status == "1", pred = predicted)$roc.vol



### Testing the model
test<-read.csv('test.csv')
colSums(is.na(test))
table(test$Gender)
test$Gender<-as.character(test$Gender)
test$Gender[test$Gender=='']<-'Unkown'
test$Gender<-as.factor(test$Gender)
table(test$Gender)
table(test$Married)
table(test$Dependents)
test$Dependents[test$Dependents=='']<-0
test$Dependents<-as.character(test$Dependents)
test$Dependents<-as.factor(test$Dependents)
table(test$Dependents)

table(test$Education)
table(test$Self_Employed)
table(test$Credit_History)
table(test$Property_Area)

### to check employment
ggplot(data = test,(aes(x = Self_Employed,y = ApplicantIncome)))+geom_boxplot()
ggplot(data = test,(aes(x = Self_Employed,y = CoapplicantIncome)))+geom_boxplot()
test$Self_Employed[test$Self_Employed=='']<-'Yes'
test$Self_Employed<-as.character(test$Self_Employed)
test$Self_Employed<-as.factor(test$Self_Employed)
table(test$Self_Employed)
str(test)
table(test$Credit_History)
test$Credit_History[is.na(test$Credit_History)]<-2
test$Credit_History<-test$Credit_History+1
test$Credit_History<-as.factor(test$Credit_History)
table(test$Credit_History)

## Transformation
test$TotalIncome<-test$ApplicantIncome+test$CoapplicantIncome
str(test)
summary(test)
ntest<-test[,-c(1,9,10)]

ntest$ApplicantIncome<-(ntest$ApplicantIncome-min(ntest$ApplicantIncome))/(max(ntest$ApplicantIncome)-min(ntest$ApplicantIncome))
ntest$CoapplicantIncome<-(ntest$CoapplicantIncome-min(ntest$CoapplicantIncome))/(max(ntest$CoapplicantIncome)-min(ntest$CoapplicantIncome))
#ntrain$LoanAmount<-(ntrain$LoanAmount-min(ntrain$LoanAmount))/(max(ntrain$LoanAmount)-min(ntrain$LoanAmount))
#ntrain$Loan_Amount_Term<-(ntrain$Loan_Amount_Term-min(ntrain$Loan_Amount_Term))/(max(ntrain$Loan_Amount_Term)-min(ntrain$Loan_Amount_Term))
ntest$TotalIncome<-(ntest$TotalIncome-min(ntest$TotalIncome))/(max(ntest$TotalIncome)-min(ntest$TotalIncome))
summary(ntest)


predictedStatus<-predict(log.reg.reduced.cr,test,type = 'response')
summary(predictedStatus)
predictedStatus<-ifelse(predictedStatus>0.6,'Y','N')
test$pred<-predictedStatus
write.csv(test,'SubmissionSwapnil.csv')
table(test$pred)

sub<-as.data.frame(cbind(test$Loan_ID,predictedStatus))
dim(sub)
