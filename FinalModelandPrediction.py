import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt

#########################################################################
# Load the user data from
user = pd.read_csv('E:\Fall 2014\Social Media Mining\Processed Features\user_features.csv').set_index('user_id')
# Load the buisness data
business = pd.read_csv('E:\Fall 2014\Social Media Mining\Processed Features\Business_features.csv').set_index('business_id')
# Load the chcekin data
checkin = pd.read_csv('E:\Fall 2014\Social Media Mining\Processed Features\Checkin_features.csv').set_index('business_id')
# Join the checkin and buisness data on business id
business = business.join(checkin)
#####################################
#Load the review data
review = pd.read_csv('E:\Fall 2014\Social Media Mining\Processed Features\\review_training_features.csv')
#Join user and review data
review = review.join(user, on='user_id', rsuffix= '_user')
# Join user and business data
review = review.join(business, on='business_id', rsuffix= '_bus')
review = review.drop(['business_id','user_id', 'city','open'],axis = 1)
review = review.set_index('review_id')
review = review.fillna(0)
##############################################
# Extract training and validation data
review_train = review._slice(slice(0,50000),0)
review_valid = review_train.ix[:int(len(review_train)*0.1),:]
review_train = review_train.ix[set(review_train.index) - set(review_valid.index),:]
print 'Size of training set %i' % len(review_train)
print 'Size of validation set %i' % len(review_valid)
review_without_useful = review_train.drop(['votes_useful'], axis=1)
######################################
#Root mean sqaure log error
def rsmle(train,test): # Define the Root Mean Square Logarithm Error
    return np.sqrt(np.mean((pow(np.log(test+1) - np.log(train+1),2))))
################################################
# Set up the gradient boosting regressor
gradientboost = GradientBoostingRegressor(n_estimators=400, max_depth= 7, random_state=7)
#Remove no of useful votes

print 'Time to train the regressor'
gradientboost.fit(review_without_useful, review_train['votes_useful'])
print 'Training complete time for predictions!'
cap_result = gradientboost.predict(review_valid.drop(['votes_useful'], axis=1))
print 'The rmsle is' + str(rsmle(cap_result, review_valid['votes_useful']))

###########################################
# Set up random forest regressor

#randomForest = RandomForestRegressor(n_estimators= 400, max_depth=7, random_state= 7)
#print 'Time to train randomForest regressor!'
#randomForest.fit(review_without_useful, review_train['votes_useful'])
#print 'Training complete'

#cap_result = randomForest.predict(review_valid.drop(['votes_useful'],axis = 1))
#print 'The rmsle is ' + str(rsmle(cap_result, review_valid['votes_useful']))


