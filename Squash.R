#loading required libraries
library(caret)
library(kernlab)
library(ggplot2)
library(cvTools)
library(caTools)

#import data
sqAct1 <- read.table("Squash1PlayerActivity.txt")
sqAct2 <- read.table("Squash2PlayerActivity.txt")
sqPos1 <- read.table("squash1positions.txt")
sqPos2 <- read.table("squash2positions.txt")

nameHeadAct <- c("Frames","Seconds","StrokePlayer","StrokeType","StrokeOutcome","ForehandBackhand","StrokeX","StrokeY")

#Change header names
sqAct1 <- sqAct1[2:132,]
head(sqAct1)
colnames(sqAct1) <- nameHeadAct

#Manipulate data by changing char to num 
sqAct1$StrokeType <- factor(sqAct1$StrokeType, levels = c("S","LL","C","CS","DL","DC","LOL","LOC","KL","KC","VLL","VC","VCS","VDL","VDC","VBN","VBO","VBR","VKL","VKC","BN","BO","BR","BS","COS"),labels = c("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25"))

sqAct1$StrokeOutcome <- factor(sqAct1$StrokeOutcome, levels = c("i","l","s","t","n"), labels = c("1","2","3","4","5"))

sqAct1$ForehandBackhand <- factor(sqAct1$ForehandBackhand, levels = c("f","b"), labels = c("1","2"))

#shifting 4th column to the last column
sqAct1 <- sqAct1[, c(1,2,3,5,6,7,8,4)]

#randomizing the data
sqAct1 <- sqAct1[sample(nrow(sqAct1)),]
str(sqAct1)

#transforming each column to numeric data type from factor type
sqAct1[nameHeadAct] <- lapply(sqAct1[nameHeadAct], as.numeric)

#visualizing transformed data
str(sqAct1)
head(sqAct1)

#-----------------------------------------------------------------------------#

#Data manipulation of player 2 
sqAct2 <- sqAct2[2:132,]
head(sqAct2)

#changing header names for better understanding
colnames(sqAct2) <- nameHeadAct
sqAct2$StrokeType <- factor(sqAct2$StrokeType, levels = c("S","LL","C","CS","DL","DC","LOL","LOC","KL","KC","VLL","VC","VCS","VDL","VDC","VBN","VBO","VBR","VKL","VKC","BN","BO","BR","BS","COS"),labels = c("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25"))

#Manipulate data by changing char to num 
sqAct2$StrokeOutcome <- factor(sqAct2$StrokeOutcome, levels = c("i","l","s","t","n"), labels = c("1","2","3","4","5"))

sqAct2$ForehandBackhand <- factor(sqAct2$ForehandBackhand, levels = c("f","b"), labels = c("1","2"))

#moving 4th column to the last
sqAct2 <- sqAct2[, c(1,2,3,5,6,7,8,4)]

#randomizing the data
sqAct2 <- sqAct2[sample(nrow(sqAct2)),]
str(sqAct2)

#making data numeric from factor
sqAct2[nameHeadAct] <- lapply(sqAct2[nameHeadAct], as.numeric)

#visualizing the data
str(sqAct2)
head(sqAct2)

#-----------------------------------------------------------------------------#

#player's position data manipulation
head(sqPos1)
sqPos1 <- sqPos1[2:13729,]

#changing header names
namePos <- c("Frames","Seconds","X1","Y1","X1Cam","Y1Cam","X2","Y2","X2Cam","Y2Cam","Phase")
colnames(sqPos1) <- namePos
head(sqPos1)

#randomizing data
sqPos1 <- sqPos1[sample(nrow(sqPos1)),]
str(sqPos1)

#making data to numeric data
sqPos1[namePos] <- lapply(sqPos1[namePos], as.numeric)

#visualize the data
str(sqPos1)
head(sqPos1)

sqPos1 <- sqPos1[1:2000,]
row.names = NULL
write.csv(a, "squash1Position.csv")
a = read.csv("squash1Position.csv")
a= a[,2:12]

#------------------------------------------------------------------------------#

# 2nd player's Position
head(sqPos2)
sqPos2 <- sqPos2[2:13729,]

#changing column names
namePos <- c("Frames","Seconds","X1","Y1","X1Cam","Y1Cam","X2","Y2","X2Cam","Y2Cam","Phase")
colnames(sqPos2) <- namePos
head(sqPos2)

#Randomizing data
sqPos2 <- sqPos2[sample(nrow(sqPos2)),]
str(sqPos2)

#transform data to numeric
sqPos2[namePos] <- lapply(sqPos2[namePos], as.numeric)

#visualizing the data
str(sqPos2)
head(sqPos2)

#------------------------------------------------------------------------------#

#loading data to some another variables to be implemented in algorithm
Act1 <- sqAct1
Act2 <- sqAct1

#only 1500 rows are taken, this is done to make processing faster if whole data is used at a time them it will take approximately 2 or 3 hours to process
Pos1 <- sqPos1[1:1500,]
Pos2 <- sqPos2[1:1500,]


#gausspr is a function in kernlab library this function is used implement gaussian process.
# Here data and the field on which process is implementing are provided to the function.
gp1 <- gausspr(StrokeType~.,data=Act1,var=0.1)

# printing the model which is processed
gp1

# predicting the gp with the dataset
pred1 <- predict(gp1,Act1[,-8])

#plot the predicted result
plot(pred1, type="l", ylab="Activity prediction")


#Gaussian process is implemented on the 2nd  player activity data
gp2 <- gausspr(StrokeType~.,data=Act2,var=0.1)
gp2
# predicting the gp with data
pred2 <- predict(gp2,Act2[,-8])

#ploting the predicted result
plot(pred2, type="l", ylab="Activity prediction")

#------------------------------------------------------------------------------#


#Gaussian process on the position data of player 1
gp3 <- gausspr(Phase~.,data=Pos1,var=0.1)
gp3

# predicting the gp with position
pred3 <- predict(gp3,Pos1[,-11])

#ploting the result
plot(pred3, type="l", ylab="Position prediction")

#Gaussian process on 2nd player position data
gp4 <- gausspr(Phase~.,data=Pos2,var=0.1)
gp4

# predicting the gp
pred4 <- predict(gp4,Pos1[,-11])

#ploting the result
plot(pred4, type="l", ylab="Position prediction")

#==============================================================================#
#analyzing the activities of both players using plots
plot(pred1, type="l", ylab="Activity prediction")
lines(pred2, type="l", ylab="Activity prediction", col= "blue")

### Player 1 has few more types of shots during the game than te other player, as it clearly seen on from the plot that black lines are going higher.

#------------------------------------------------------------------------------#

#Analyzing the positions changed by players during the game
plot(pred4, type="l", ylab="Position prediction")
lines(pred3, type = "l", col= "green")

### Player 2 is occupying more area of the court by frequently moving in the court as compared with the 1st player. This shows that 1st player is more active during the game.
