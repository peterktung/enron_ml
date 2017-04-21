1. The Enron dataset contains financial data (salary, bonus, stocks, etc) as well as email metadata (number of emails sent/received) of Enron employees. We want to see such data could be used to identify persons-of-interest (poi). From the initial data analysis, the relationship between individual feature and whether an employee is a poi is unclear. Therefore, this would be a problem that we could use machine learning techniques to help us find any possible relationships between the financial and email data and poi's.  
  There were two outliers that was removed through manual inspection as they were not real people ("TOTAL", and "THE TRAVEL AGENCY IN THE PARK"). The data distribution also suggest that there were possible poi outliers, but given that there are only 18 poi's vs 125 non poi's, we left these entries in the dataset.  The fact that the poi is an outlier could also be a factor that could be leveraged is our machine learning algorithm. Every feature had some missing data, whether they are zero's or NaN values. We found that some financial features had mostly zero values (director_fees, restricted_stock_deferred, loan_advances) so they were removed as they provide no predictive power. Below is a table that list the number of entries that have zero or NaN value for each feature.
  
  | feature                   | number of zeros and NaN's |
  |---------------------------|---------------------------|
  | bonus                     | 62                        |
  | deferral_payments         | 105                       |
  | deferred_income           | 95                        |
  | director_fees             | 127                       |
  | exercised_stock_options   | 42                        |
  | expenses                  | 49                        |
  | from_messages             | 57                        |
  | from_poi_to_this_person   | 69                        |
  | from_this_person_to_poi   | 77                        |
  | loan_advances             | 140                       |
  | long_term_incentive       | 78                        |
  | other                     | 52                        |
  | restricted_stock          | 34                        |
  | restricted_stock_deferred | 126                       |
  | salary                    | 49                        |
  | shared_receipt_with_poi   | 57                        |
  | to_messages               | 57                        |
  | total_payments            | 20                        |
  | total_stock_value         | 18                        |
  
  After cleaning up the data, we were left with 143 entries and 16 features.  
  
2. Two features were engineered from the 16 remaining features. One was "bonus to salary ratio", the other was "exercised stock to salary" ratio. The reasoning here is that poi's would make an abnormal amount of money either through bonuses or through selling stocks when the stock price was artifically inflated. These amounts would be much higher in comparson to their normal salary. We then used we used SelectKBest to choose help select our features. We added SelectKBest to a pipeline on top of our final classifier model (DecisionTreeClassifer) and with the help of GridSearchCV to tune the optimal number of features to use `k`, we end up with the following (`k = 3`):
  
  | feature                 | feature score | feature importance |  
  |-------------------------|---------------|--------------------|  
  | salary                  |        11.196 |              0.370 |  
  | exercised_stock_options |        13.714 |              0.376 |  
  | total_stock_value       |        14.691 |              0.254 |  
  
  Note that while both salary and exercised_stock_options were selected, the engineered "exercised_stock_salary_ratio" was not used. Adding the engineered feature in the decision reduced both the precision and recall score significantly (0.400 to 0.334 and 0.340 to 0.294 respectively). This suggests that the ratio between the salary and the exercised stock option was not as good of an indicatior of poi's as we originally thought. Finally, we did not scale our features as our final model was a tree based classifier so "distance" does not affect the outcome.  
  
3. The final model used was the DecisionTreeClassifier. Others there were tried included LogisticRegression, SVM, and RandomForestClassifier. Logistic regression had the poorest performance while SVM had a good precision call, it has a very low recall score. Overall, the tree based classifiers had the best performance, but oddly enough, the random forest classifier did not perform as well as the base decision tree classifier. The scores we got for all the algorithms we tried:
  
  | Classifier               | Accuracy | Precision | Recall | F1-score |
  |--------------------------|----------|-----------|--------|----------|
  | Logistic Regression      | 0.709    | 0.109     | 0.165  | 0.131    |
  | SVM                      | 0.866    | 0.444     | 0.032  | 0.006    |
  | Random Forest Classifier | 0.859    | 0.443     | 0.216  | 0.290    |
  | Decision Tree Classifier | 0.820    | 0.400     | 0.340  | 0.367    |
  
4. Parameter tuning means we adjust the values of the hyperparameters of a particular machine learning algorithm.  This is done so that we don't overfit our model with the testing data. To tune our hyperparameters, we used a grid search strategy where we assign a range of values for each parameter we are looking to tune. Grid search then finds the combination of values that results in a model with the highest accuracy.  For the decision tree classifier, we tuned the parameters that affect the size of the tree (max_depth, and max_leaf_nodes) as well as the parameters that affect the sample size in the leaf nodes (min_samples_split, and min_samples_leaf). Here are the actual values we tested for each parameter and what was chosen by GridSearchCV.

  | Parameter          | Values tested            | Value chosen |
  |--------------------|--------------------------|--------------|
  | min_samples_split  | [2, 3, 4, 5, 6, 7, 8, 9] | 2            |
  | min_samples_leaf   | [1, 2, 3, 4]             | 1            |
  | max_depth          | [None, 2, 4, 6, 10]      | 6            |
  | mas_max_leaf_nodes | [None, 2, 4, 6, 10, 20]  | 20           |
 ddd 
5. Validation is the way we can make sure our hyperparameters are tuned properly before the model is tested against a held-out dataset. Validation is done by splitting a dataset into two, one for training and the other for validation. If we do not separate the distinction between training vs. validation vs. test datasets, we could end up using the test set to validate our model and further tune our model based on the results of the test dataset.  If this happen, we will introduce data leakage and end up creating a model that is unrealistic in terms of its predictive power.  
  To validate our model in this project, we used a stratified shuffle split strategy that splits the data randomly multiple times while keeping the ratio of poi's and non-poi's in each training and validation set the same.  We do this because our dataset is small and we have much more non-poi's than poi's in our dataset.  
  
6. We used both precision and recall as our metrics to evaluate the performance of our models. Precision is the percentage of true positives (poi's) of all the records that were predicted as positives (true + false positives). Recall is the percentage of true positives of all the records that were actual postives (true positves + false negatives). In other words we took a look at how many individuals we predicted are poi's were in fact poi's (precision), and how many poi's were missed in our predictions (recall).  Our final model has a precision score of 0.400 and a recall score of 0.340.  This means that if we had predicted 100 poi's, our model wrongly accused 60 people as pois. Also if there were 100 poi's in reality, we would not have caught 66 of them. Both are not particularly good results!
