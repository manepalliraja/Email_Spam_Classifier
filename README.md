An email spam classifier built using a publicly available dataset. Built and cleaned the corpus as well as built the models in R, using the "tm" package for text mining and "rpart" and "randomForest" packages for Decision trees and Random forests respectively. The models were then compared to a logistic regression model built from the same data to evaluate which model generalized to the testing data the best.

The dataset contains just two fields:
  text: The text of the email.
  spam: A binary variable indicating if the email was spam.

The "ham" messages in this dataset come from the inbox of former Enron Managing Director for Research Vincent Kaminski, one of the inboxes in the Enron Corpus. One source of spam messages in this dataset is the SpamAssassin corpus, which contains hand-labeled spam messages contributed by Internet users. The remaining spam was collected by Project Honey Pot, a project that collects spam messages and identifies spammers.
