# Diabetes-Prediction-Model
## Introduction:
This documentation describes a diabetes prediction model implemented in Python using various machine learning algorithms. The model uses a dataset of diabetes patients, including features such as glucose level, insulin, BMI, and blood pressure, to predict the outcome of diabetes (1 for diabetic, 0 for non-diabetic).

## Data Preprocessing:

Importing Libraries: The necessary libraries such as pandas, matplotlib, seaborn, and numpy are imported.

Loading the Dataset: The diabetes dataset is loaded using the pandas library from a CSV file.

Exploratory Data Analysis (EDA): The dataset is analyzed using the Sweetviz library to generate an HTML report, which provides insights into the dataset's structure, missing values, distributions, and relationships between variables.

Handling Missing Values: Although there are no null values in the dataset, certain features have 0 values that need to be replaced. The mean values of the respective features (Insulin, BMI, Blood Pressure, Glucose, and Skin Thickness) are calculated and used to replace the 0 values.

Data Visualization: The correlation between different features is visualized using a heatmap and pair plot. Additionally, a bar chart is plotted to show the count of diabetic and non-diabetic cases.

## Model Building:

Data Splitting: The dataset is split into training and testing sets using the train_test_split function from scikit-learn. The features (x) include columns 0, 1, 5, and 7, while the target variable (y) is the last column.

Feature Scaling: Z-score normalization is applied to standardize the feature values. The StandardScaler from scikit-learn is used to scale both the training and testing data.

Decision Tree Algorithm: A Decision Tree Classifier is implemented to build the prediction model. The algorithm's performance is evaluated by calculating accuracy scores for both the training and testing datasets. The max_depth parameter is varied from 1 to 20, and the results are plotted to identify the optimal depth. Finally, a decision tree model with max_depth=3 is trained on the entire training set and tested on the testing set. The accuracy score and confusion matrix are computed to evaluate the model's performance.

Logistic Regression: Logistic Regression is another algorithm used for prediction. The model is trained and tested on the same training and testing datasets used for the decision tree algorithm. The accuracy score and confusion matrix are computed to evaluate the model's performance.

K-Nearest Neighbors (KNN) Algorithm: The KNN classifier is implemented to build the prediction model. Similar to the previous algorithms, the training and testing datasets are used. The number of neighbors (k) is varied from 1 to 20, and the accuracy scores for different k values are plotted. A KNN model with k=4 is trained and evaluated using accuracy score and confusion matrix.

Support Vector Machine (SVM) Algorithm: The SVM classifier with a linear kernel is implemented to build the prediction model. Feature scaling is applied to the entire dataset. The model is trained and tested on the same training and testing datasets used for previous algorithms. The accuracy score and confusion matrix are computed to evaluate the model's performance.

Cross-Validation: Stratified k-fold cross-validation is used to evaluate the Logistic Regression and SVM models. The accuracy scores and mean absolute error (MAE) are calculated using cross_val_score to assess the models' performance on different splits of the data.
## Error Minimization
Here are some techniques used to avoid errors:

Handling Missing Values: The dataset is checked for missing values, and although there are no null values, certain features have 0 values that need to be replaced. To address this, the mean values of the respective features (Insulin, BMI, Blood Pressure, Glucose, and Skin Thickness) are calculated and used to replace the 0 values. This helps in preventing errors caused by incorrect or missing data.

Feature Scaling: Z-score normalization (StandardScaler) is applied to standardize the feature values. Both the training and testing datasets are scaled using the StandardScaler from scikit-learn. Scaling the features helps avoid errors due to differences in the magnitude or scale of the variables, ensuring that all features contribute equally to the model's predictions.

Hyperparameter Tuning: In the decision tree algorithm, the model's performance is evaluated for different values of the max_depth hyperparameter. By varying the max_depth from 1 to 20 and monitoring the accuracy scores, the optimal depth of 3 is selected. This tuning process helps prevent errors caused by overfitting or underfitting the model to the training data, improving its generalization capabilities.

Cross-Validation: To assess the performance of the logistic regression and SVM models, stratified k-fold cross-validation is used. This technique divides the dataset into multiple subsets or folds and performs training and evaluation on different combinations of these folds. Cross-validation helps estimate the model's performance on unseen data, reducing the risk of error due to overfitting or biased evaluation on a single split of the data.

Evaluation Metrics: Multiple evaluation metrics are used to assess the model's performance, including accuracy score, confusion matrix, mean absolute error (MAE), and root mean squared error (RMSE). By using a combination of metrics, the model's accuracy, precision, recall, and general performance can be evaluated comprehensively, reducing the chance of errors caused by relying solely on a single metric.

Visualization: Various visualizations, such as heatmaps, pair plots, bar charts, and line plots, are created to analyze the dataset, assess correlations, and visualize the model's predictions. Visualizations help in identifying patterns, anomalies, and potential errors, enabling a more comprehensive understanding and interpretation of the data and model results.

By implementing these techniques and strategies, the diabetes prediction model aims to minimize errors caused by missing data, inconsistent scaling, inappropriate hyperparameter settings, and biased evaluation. These steps contribute to a more robust and accurate prediction model.

## Conclusion:
This documentation outlines the implementation of a diabetes prediction model using four different machine learning algorithms: Decision Tree, Logistic Regression, KNN, and SVM. The model's performance is evaluated using accuracy scores, confusion matrices, and cross-validation. Users can choose the best-performing algorithm based on the evaluation metrics to predict diabetes outcomes for new instances.
