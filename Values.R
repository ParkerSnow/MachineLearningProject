# Title     : MachineLearningProject
# Objective : Explore the dataset for valuable information
# Created by: Parker
# Created on: 3/14/2021


train_final <- read.csv('train_final.csv')
install.packages("ggcorrplot")
library(ggplot2)
library(ggcorrplot)



numeric <- subset(train_final,select = c(age,fnlwgt,education.num,capital.gain,capital.loss,hours.per.week,income.50K))


corr <- round(cor(numeric), 1)
ggcorrplot(corr)

func <- function(){
theData <- subset(train_final)

p <- ggplot(data=theData,aes(x=income.50K)) + geom_bar()
print(p)
p1 <- ggplot(data=theData,aes(x=income.50K,fill=workclass)) + geom_bar(position=position_dodge())
print(p1) 
p2 <- ggplot(data=theData,aes(x=income.50K,fill=education)) + geom_bar(position=position_dodge())
print(p2) 
p3 <- ggplot(data=theData,aes(x=income.50K,fill=marital.status)) + geom_bar(position=position_dodge())
print(p3) 
p4 <- ggplot(data=theData,aes(x=income.50K,fill=occupation)) + geom_bar(position=position_dodge())
print(p4) 
p5 <- ggplot(data=theData,aes(x=income.50K,fill=relationship)) + geom_bar(position=position_dodge())
print(p5) 
p6 <- ggplot(data=theData,aes(x=income.50K,fill=race)) + geom_bar(position=position_dodge())
print(p6) 
p7 <- ggplot(data=theData,aes(x=income.50K,fill=sex)) + geom_bar(position=position_dodge())
print(p7) 
p8 <- ggplot(data=theData,aes(x=income.50K,fill=native.country)) + geom_bar(position=position_dodge())
print(p8) 
}
func()

l <- ggplot(data=train_final,aes(x=hours.per.week,fill=income.50K,color=as.factor(income.50K))) + geom_histogram(position="stack",alpha=.5,binwidth = 25)
print(l)

moreData = subset(train_final,capital.gain < 40000)

with(data = moreData, plot((capital.gain - capital.loss)^2,age,col=as.factor(income.50K)))

