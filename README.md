# Biostat626-Midterm1

## The first part of this midterm project is a binary classification problem.

1. For binary classification in this problem, activity<=3 will be classified to class 1, and others will be classified to class 0. Since we only have the class label for training dataset, to understand the performance of each model, first do the train-test split on the training data. Using such train_train data to train models and use the train_test data to check the accuracy.
2. The first approach for this problem is the **Logistic Regression**. Utilize the cv.glmnet() to find the min lambda, refit the model with this smallest lambda value and predict on the train_test set. Get the accuracy as 1.
3. The second approach is the **Random Forest**. The accuracy is 0.992, which is lower than logistic regression. 
4. I choose the LR as the final model to do the prediction on test data.


## The second part of this midterm project is a multi-class classification problem.
1. Similar to the first part, activity<=6 will be labeled as the original class, and activity > 6 will be labeled as 7. Also do the train-test split on the new labeled train dataset and get multi_train_train and multi_train_test. Use the previous one for training and the last one for testing the accuracy.
2. First is my baseline mode, **Logistic Regression**. Setting max_iteraton to avoid early shutdown. Get accuracy of 0.9416.
3. Second model for this problem is the **Random Forest**. I got an accuracy of 0.9791.
4. Since the prediction on test data is based on the model which trained on the sub-train set, I try to fit the whole training set on the RF model and use such a model to predict on test data. However, the leaderboard shows that the accuracy of this “whole” model is worse than the previous “sub” model.
5. The third approach is tuning the hyper-parameter of Random Forest. I try with **n_tree as 100 and 800** and get accuracies of 0.9752 and 0.9782, respectively.
6. Third approach is **Adaboosting** and gets an accuracy of 0.9651.
7. Since models above have similar accuracy scores, I try with feature selection to delete those highly collinear features(collinearity > 0.75).
8. Fit **selected-features multi_train set on LR** again. Get accuracy of 0.9542, better than before; fit on RF with n_tree=100, get accuracy of 0.9725, slightly lower than before.
9. Fit full_features training set and selected_features training set on **KNN with K=3**. Get accuracy of 0.9704 and 0.9516, respectively. Also fit the full_features training set on **K=5**, get acc=0.9643
10. Fit selected_features training set and full_features training set on **SVM with linear kernel**, get acc = 0.9669 and 0.9834, respectively.
11. Write a majority vote function to ensemble results from many models’ prediction results. Also checking how many different labels between two prediction results.
12. Fit full-features training set on **Logistic Regression** again with regularization with **Lasso regression**, get acc=0.9813
13. Fit full_features training set on **SVM with radial kernel and polynomial kernel with degree=2**, get acc=0.976 and 0.9734
14. **Ensemble prediction results** from **9** models: full_features fitting on RF, RF with n_tree=100, RF with n_tree=800, KNN with K=3, SVM with linear, radial and polynomial kernel, LR with Lasso regression, and selected features fitting on RF with n_tree = 100. Using the majority vote to get the final prediction result.
