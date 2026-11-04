
# Kaggle Model Write-up
### Jothi Gupta
### AIPI 520

## Overview
The data was challenging for two main reasons: there was so much of it and there were a lot of missing features. It was therefore hard to decide what data to use and how. I began by (1) reviewing each data source and understanding how they fit together (2) cleaned the data frame (3) did some feature engineering (4) tested a baseline logistic regression model (5) viewed the correlation matrix and decided to test a regularization model (6) compared L1 to L2 regularization with varying lambdas and reported the best model (7) tried random forest, but L2 regularization ended up still winning out. Given the high AUROC of the baseline logistic regression model (0.96290), I decided to continue pursuing linear models before moving to something more complex as the model was already performing so successfully. I suspected this was the case because the output was understandably predictable. It made sense that people who purchased subscriptions in the past would purchase subscriptions again, and if the concerts were in their city they were more likely to get an annual pass. These assumptions made sense, thus the success of a simple model also made sense.

## General Observations
- Geographical data
  - each account has a billing city, shipping zip code
  - each concert lists a city
- Training data
  - the y contains the account id and the label of whether or not they renewed their subscription (1 for yes 0 for no)
  - output categorical data 
- The data is skewed negative
  - fewer than 5% of the training data has a label of 1
  - it will be important to either stratify split or create synthetic data so we don't miss out on the false negatives
- Empty data points
  - few accounts are marked for "multiple subscriptions"
  - shipping zip code / city is 98% empty so may have to rely on billing city

## Takeaways from EDA
- Since label 1 is rare, a classifier predictor based on accuracy will be misleading, rely on precision, recall, PR-AUC
- Some repeated account IDs in subscription/ticket tables, aggregate into one row if represent real history
- Since shipping zip code / city is 98% incomplete, we'll have to do location based on the billing zip code

## Data Pipeline
1. Exploring the data (inspect for quality) `preliminary_eda.py`
2. Cleaning up account dataframe `data_clean.py`
    - checked for duplicate accounts (there are none)
    - cleaned account IDs by converting them into text and trimming spaces, raising any errors for missing, blank, duplicate IDs
    - standardizing billing cities by lowercasing, trimming, collapsing extra spaces, treatying empty strings as missing
    - converting donation amount to numeric values, raising error for invalid entries
    - converting donation count to integers, checking first for fractional counts
    - converts first-donation dates to datetime, preserving missing dates
    - dropgs selected columns with >70% missing, shipping zip, billing zip, shipping city, relationship
    - returns cleaned df without deleting accounts / filling in missing (flags outliers)
3. Clean up the other data frames:
    - starting with subscriptions, doing a similar cleaning process to the account df
4. Compared subscriptions, account, and train / test dataframes
    - evaluated overlap between
5. Feature engineering `engineer_features.py`
    - combined account information with subscription history into one row
    - added given features to train and testing dataframes:
        
  | Feature group | Features | Meaning |
|---|---|---|
| Geography | `billing.city` | City text standardized to lowercase with extra whitespace removed. |
| Donation amounts and counts | `amount.donated.2013`, `amount.donated.lifetime`, `no.donations.lifetime` | Donation amounts for 2013 and lifetime, and lifetime donation count. |
| Donation indicators | `has_donated`, `donated_in_2013` | Binary flags indicating whether lifetime donation count or 2013 donation amount is greater than zero. Missing source values not filled in. |
| Average donation | `average_donation` | Lifetime donation amount divided by lifetime donation count. Missing when the count is zero or unavailable. |
| Donor tenure | `donor_tenure_years` | Reference year (default: 2014) minus first-donation year. Missing for accounts without a positive donation count or with a future first-donation year. |
| Subscription history | `has_subscription_history`, `total_subscribed_seasons` | Whether the account has historical subscription records and the number of subscribed seasons. Both are zero when no history exists. |
| Subscription recency | `seasons_since_last_subscription` | Seasons elapsed since the latest subscription, relative to 2013–14. Zero means subscribed in 2013–14; missing means no history. |
| Recent subscription streak | `recent_subscription_streak` | Consecutive subscribed seasons ending in 2013–14. Zero if the account did not subscribe in 2013–14. |
| Longest subscription streak | `longest_subscription_streak` | Longest consecutive run of subscribed seasons anywhere in the historical records. Zero when no history exists. |
| Recent-season indicators | `subscribed_2011_12`, `subscribed_2012_13`, `subscribed_2013_14` | Separate binary flags indicating subscription in each specified season. |
| Latest subscription details | `latest_subscription_seats`, `latest_package`, `latest_location`, `latest_price_level` | Seat count, package, location, and categorical price-level code from the latest historical subscription record. |

6.Split labeled data, reserved 20% for validation
7. Preprocess within training folds
8. Train and compare models
    - Evaluated unpenalized logistic regression model `baseline_model.py`
    - Tuned L1 and L2 regularization `penalty_model.py`
    - Compared random forest to ideal regularized model `random_forest_model.py`
9. Evaluate and generate submissions  
    - L1 won in AUROC scoring

## Reflecting on choices
Feature selection: I wanted to produce features to account for someone's history both in donation and subscription. So I produced a series of features documenting their most recent consecutive streak, how much they donated, their average donation, etc. Thinking through the data realistically, figuring out how often the patron engaged with the organization was important.

Algorithm selection: I started with a logistic regression model to get a baseline. Surprisingly, the logistic regression model did quite well but given the number of features I engineered and the correlation matrix showing some overlap, I assumed regularization could help. So I then compared L1 and L2 regularization at various lambdas and chose the best one. This ended up being my most successful model, with an AUROC close to 0.98 on the testing data. I planned to do more complex modeling but given this success, it seemed sufficient. I tried a random forest model just in case, and this ended up performing worse than the L1 regularized model, so I stuck with that. 

## Validation Approach
I used a stratified 80/20 train-validation split to preserve the proportion of subscribers to non-subscribers in both subsets. Within the 80% validation set, I used a five-fold stratified cross-validation to compare models and tune hyperparameters, including regularized strength. Each model was trained on four folds and evaluated on the remaining fold. Regularized models and random forest candidates were chose using AUROC and evaluated on the remaining 20% validation subset. A fixed random seed of 42 was used for reproducibility purposes.