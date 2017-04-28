# Clearing all the variables in the workspace
rm(list = ls(all = TRUE))

# Reading in the data into R; the summary of the dataframe and the dependent variable
emails <- read.csv("emails.csv", stringsAsFactors = FALSE)
str(emails)
summary(emails)
table(emails$spam)

# Finding the biggest and the smallest email
max(nchar(emails$text))
which.min(nchar(emails$text))
emails$text[1992]

# Building the corpus and preprocessing it by converting it to lowercase, removing the 
# punctuation, the english stop-words and finally stemming the words
library(tm)
library(SnowballC)
corpusEmails <- Corpus(VectorSource(emails$text))
corpusEmails <- tm_map(corpusEmails, content_transformer(tolower))
corpusEmails <- tm_map(corpusEmails, removePunctuation)
corpusEmails <- tm_map(corpusEmails, removeWords, stopwords("english"))
corpusEmails <- tm_map(corpusEmails, stemDocument)

# Building a document term matrix from the corpus
dtmEmails <- DocumentTermMatrix(corpusEmails)

# Limiting the matrix to words that occur in atleast 5% of the documents
sparseDtmEmails <- removeSparseTerms(dtmEmails, 0.95)

# Building a dataframe from the sparse word matrix
emailsSparse <- as.data.frame(as.matrix(sparseDtmEmails))

# Making the variable names of emailSparse valid
colnames(emailsSparse) <- make.names(colnames(emailsSparse)) 

# Importing the dependent variable into the new dataframe
emailsSparse$spam <- emails$spam

# Converting the dependent variable to a factor
emailsSparse$spam <- as.factor(emailsSparse$spam)

# Splitting the dataframe into train and test. Training set will have 70% of the data.
library(caTools)
set.seed(123)
split <- sample.split(emailsSparse$spam, SplitRatio = 0.7)
emailsTrain <- subset(emailsSparse, split == TRUE)
emailsTest <- subset(emailsSparse, split == FALSE)



# Logistic regression model
spamLog <- glm(spam ~ ., data = emailsTrain, family = "binomial")
summary(spamLog)

# Predicting probabilties on training set
predLogTrain <- predict(spamLog)
table(emailsTrain$spam)
length(subset(predLogTrain, predLogTrain < 0.00001))
length(subset(predLogTrain, predLogTrain > 0.99999))
length(subset(predLogTrain, predLogTrain >= 0.00001 & predLogTrain <= 0.99999))
table(emailsTrain$spam, predLogTrain > 0.5)
(3052 + 954) / nrow(emailsTrain) # Training set accuracy with a threshold of 0.5

library(ROCR)
predictionTrainLog <- prediction(predLogTrain, emailsTrain$spam)
as.numeric(performance(predictionTrainLog, "auc")@y.values) # Training set auc
# All these values indicate that the logistic model is severely overfit

# Predicting probabilties on testing set
predLogTest <- predict(spamLog, newdata = emailsTest)
table(emailsTest$spam, predLogTest > 0.5)
(1258 + 376) / nrow(emailsTest) # Testing set accuracy with a threshold of 0.5

predictionTestLog <- prediction(predLogTest, emailsTest$spam)
as.numeric(performance(predictionTestLog, "auc")@y.values) # Testing set auc



# Decision tree model
library(rpart)
library(rpart.plot)
spamCART <- rpart(spam ~ ., data = emailsTrain, method = "class")
prp(spamCART)

# Predicting probabilties on training set
predCARTTrain <- predict(spamCART)
# Keeping only the probabilities of predictions
predCARTTrain <- predCARTTrain[,2]
table(emailsTrain$spam, predCARTTrain > 0.5)
(2885 + 894) / nrow(emailsTrain) # Training set accuracy with a threshold of 0.5

predictionTrainCART <- prediction(predCARTTrain, emailsTrain$spam)
as.numeric(performance(predictionTrainCART, "auc")@y.values) # Training set auc

# Predicting probabilties on testing set
predCARTTest <- predict(spamCART, newdata = emailsTest)
# Keeping only the probabilities of predictions
predCARTTest <- predCARTTest[,2]
table(emailsTest$spam, predCARTTest > 0.5)
(1228 + 386) / nrow(emailsTest) # Testing set accuracy with a threshold of 0.5

predictionTestCART <- prediction(predCARTTest, emailsTest$spam)
as.numeric(performance(predictionTestCART, "auc")@y.values) # Testing set auc



# Random forest model
library(randomForest)
set.seed(123)
spamRF <- randomForest(spam ~ ., data = emailsTrain)

# Predicting probabilities on training set
predRFTrain <- predict(spamRF, type = "prob")
# Keeping only the probabilities of predictions
predRFTrain <- predRFTrain[,2]
table(emailsTrain$spam, predRFTrain > 0.5)
(3013 + 912) / nrow(emailsTrain) # Training set accuracy with a threshold of 0.5

predictionTrainRF <- prediction(predRFTrain, emailsTrain$spam)
as.numeric(performance(predictionTrainRF, "auc")@y.values) # Training set auc

# Predicting probabilties on testing set
predRFTest <- predict(spamRF, newdata = emailsTest, type = "prob")
# Keeping only the probabilities of predictions
predRFTest <- predRFTest[,2]
table(emailsTest$spam, predRFTest > 0.5)
(1290 + 387) / nrow(emailsTest) # Testing set accuracy with a threshold of 0.5

predictionTestRF <- prediction(predRFTest, emailsTest$spam)
as.numeric(performance(predictionTestRF, "auc")@y.values) # Testing set auc

# These values indicate that the Random forest model has the best performance on the testing
# set and that the Logistic regression model is overfitting the most among the three models