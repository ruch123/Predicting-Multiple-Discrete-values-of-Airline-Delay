rm(list=ls())
d<-read.csv(file.choose(),header=T,stringsAsFactors=T)
# to use a small subset of data ,pertaining to only "dec" month of 1987.
d = subset(d, Month == "12")
str(d)
# to remove coloumns with missing values, month, year coloumns, and coloumns with constant values
d=d[ ,-20:-29]
d=d[ ,-14]
d=d[ ,-11]
d=d[ ,-1]
d=d[ ,-1]
d=d[ ,-7]
str(d)
# to replace NAs with 0.

d <- na.omit(d)
d[is.na(d)] <- 0
str(d)

d$Origin<-as.numeric(d$Origin)
d$Dest<-as.numeric(d$Dest)
str(d)
##neural networks

library(nnet)
library(caret)
install.packages("RCurl")
library(RCurl)
install.packages("Metrics")
library(Metrics)
attach(d)
Delay <- cut(ArrDelay,breaks=c(-50,0, 15,30,45,1000),labels=c("On_Time", "Almost_On_time", 
                                                              "Short_Delay","Much_delay", "Uncertain_Delay"),right=FALSE)
d<-cbind(Delay,d)
split <- floor(nrow(d)/2)
dTrain <- d[0:split,]
dTest <- d[(split+1):nrow(d),]

DelayModel <- multinom(Delay~DepDelay+Distance+Origin+Dest+DayOfWeek+DayofMonth, data=dTrain, maxit=100, trace=T)
length(Delay)
length(DepDelay)

##To find the most influential variable in Delayodel

mostImportantVariables <- varImp(DelayModel)
mostImportantVariables$Variables <- row.names(mostImportantVariables)
mostImportantVariables <- mostImportantVariables[order(-mostImportantVariables$Overall),]
print(head(mostImportantVariables))



##Next we predict Delay using the predict function on the testing data set. There are
##two ways to compute predictions, class or probs:
preds1 <- predict(DelayModel, type="probs", newdata=dTest)
head(preds1)
preds2 <- predict(DelayModel, type="class", newdata=dTest)
head(preds2)

##Choosing which of the two predictions will depend on your needs. If you just want your
##Delay predictions, use class, if you need to do anything more complex, like measure
##the conviction of each prediction, use the probs option (every row will add up to 1). 


##To check the accuracy of the model, we call the postResample function from caret. For 
##numeric vectors, it uses the mean squared error and R-squared and for factors, the 
##overall agreement rate and Kappa:

postResample(dTest$Delay,preds2)

##lets do some simple repeated cross validation to get a more comprehensive mean accuracy
##score and understand convergence. The code below will iterate through all the data to
##give every variable a chance of being test and train data sets. The first time around 
##we set maxit to only 100:
totalAccuracy <- c()
cv <- 10
cvDivider <- floor(nrow(d) / (cv+1))

for (cv in seq(1:cv)){
  # assign chunk to data test
  dataTestIndex <- c((cv * cvDivider):(cv * cvDivider + cvDivider))
  dataTest <- d[dataTestIndex,]
  # everything else to train
  dataTrain <- d[-dataTestIndex,]
  
  DelayModel <- multinom(Delay~., data=dataTrain, maxit=100, trace=T) 
  
  pred <- predict(DelayModel, newdata=dataTest, type="class")
  
  #  classification error
  cv_ac <- postResample(dataTest$Delay, pred)[[1]]
  print(paste('Current Accuracy:',cv_ac,'for CV:',cv))
  totalAccuracy <- c(totalAccuracy, cv_ac)
}
mean(totalAccuracy)

##The mean accuracy of 0.64 is much lower than the accuracy that we got with the original
##simple split. We notice that the log output never prints the word converged.
##This means the model never reaches the lowest error or global minima and therefore
##isn't the best fit. 

##Let's try this again and let the model converge by setting the maxit to a large number

totalAccuracy <- c()
cv <- 10
cvDivider <- floor(nrow(d) / (cv+1))

for (cv in seq(1:cv)) {
  # assign chunk to data test
  dataTestIndex <- c((cv * cvDivider):(cv * cvDivider + cvDivider))
  dataTest <- d[dataTestIndex,]
  # everything else to train
  dataTrain <- d[-dataTestIndex,]
  
  DelayModel <- multinom(Delay~., data=dataTrain, maxit=1000, trace=T) 
  
  pred <- predict(DelayModel, newdata=dataTest, type="class")
  
  #  classification error
  cv_ac <- postResample(dataTest$Delay, pred)[[1]]
  print(paste('Current Accuracy:',cv_ac,'for CV:',cv))
  totalAccuracy <- c(totalAccuracy, cv_ac)
}
mean(totalAccuracy)
##The score using the repeated cross validation code is better than the original simple
##split of 0.9304 and we let each loop converge. The point of using the repeated cross
##validation code isn't that it will return a higher accuracy score (and it doesn't always) 
##but that it will give a much more accurate score as it uses all of the data.
