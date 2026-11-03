2-4 pages

# Kaggle Model Write-up
### Jothi Gupta
### AIPI 520


## Overview
The data was challenging for two main reasons: there was so much of it, there was a lot of missing features, and it was therefore hard to decide what data to use and how. I began by (1) reviewing each data source and understanding how they fit together (2) cleaned the data frame (3) did some feature engineering (4) tested a baseline logistic regression model (5) viewed the correlation matrix and decided to test a regularization model (6) compared L1 to L2 regularization with varying lambdas and reported the best model (7) tried random forest, but L2 regularization ended up still winning out. 

## General Observations
- location, account has billing city, shipping zip code, concert has city
- in testing data, has the account id and the label of whether or not they renewed the subscription (0 or 1)
- few accounts are marked for "multiple subscriptions"
- fewer than 5% of the training data have the label of 1, so it's generally uncommon to renew the subscription
- shipping zip code / city is 98% empty so may have to rely on billing city
- tickets_all is kind of incomplete

Takeaways from EDA:
- since label 1 is rare, a classifier predictor based on accuracy will be misleading, rely on precision, recall, PR-AUC
- some repeated account IDs in subscription/ticket tables, aggregate into one row if represent real history
- since shipping zip code / city is 98% incomplete, we'll have to do location based on the billing zip code

## Approach Overview
1. Cleaning up account df, remove columns with over 70% missing 
2. Checking for duplicate accounts (there are none)
3. Cleaning: 
    - cleaned account IDs by converting them into text and trimming spaces, raising any errors for missing, blank, duplicate IDs
    - standardizing billing cities by lowercasing, trimming, collapsing extra spaces, treatying empty strings as missing
    - converting donation amount to numeric values, raising error for invalid entries
    - converting donation count to integers, checking first for fractional counts
    - converts first-donation dates to datetime, preserving missing dates
    - dropgs selected columns with >70% missing, shipping zip, billing zip, shipping city, relationship
    - returns cleaned df without deleting accounts / filling in missing
4. Clean up the other data frames:
    - starting with subscriptions, doing a similar cleaning process to the account df
5. Compared subscriptions, account, and train / test dataframes
    - evaluated overlap between
6. Feature engineering:
    - added given features to train and testing dataframes:
        - 
7. Started with a baseline logistic regression model:
8. Lot's of correlated features (makes sense, if got subscription in 2012, probs got one in 2013) but attempted a second regularization model (compared L1 and L2 and found ideal penalty term size)
9. reported L2 regularization as the best model

## Reflecting on choices
Feature selection:
Algorithm selection: 
## Data Pipeline
## The L2 Model
## Validation Approach