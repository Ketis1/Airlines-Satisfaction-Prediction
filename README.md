# Airlines-Satisfaction-Prediction

## Data Analysis

The data analysis section focused on understanding the distributions of the features and their correlation with the target variable ('satisfaction').

### Distribution Histograms

Histograms were plotted for the numerical columns to visualize their distributions. This helped in understanding the spread and patterns within these features.

### Correlation with Satisfaction

The relationship between individual features and the 'satisfaction' target was explored:
- **Box plots**: Used to visualize the relationship between the numerical feature 'Age' and 'satisfaction'.
- **Bar plots**: Used to show the percentage of satisfied customers for different categories within the encoded categorical features ('Gender', 'Customer Type', 'Type of Travel', 'Class', 'Flight Haul', and 'Delay').

### Correlation Matrix

A correlation matrix was computed and visualized as a heatmap to show the pairwise correlations between all numerical variables in the training dataset. This helped in identifying features that are highly correlated with each other and with the target variable 'satisfaction'.

### Outliers

Box plots were used to visualize potential outliers in the numerical features (excluding the target and already encoded categorical columns).

-----------

## Data Preparation

The dataset was obtained from Kaggle. The data is split into two files: `train.csv` and `test.csv`. These files were downloaded using the `kagglehub` library and then loaded into pandas DataFrames named `train_df` and `test_df`.

### Handling Missing Values

Initially, missing values were checked in both `train_df` and `test_df`. It was found that only the 'Arrival Delay in Minutes' column contained missing values in both datasets. To handle these missing values, the rows with missing 'Arrival Delay in Minutes' were dropped from both the training and testing DataFrames using the `.dropna()` method with `inplace=True`.

### Removing Useless Columns

The columns 'Unnamed: 0' and 'id' were deemed unnecessary for the analysis and model training as they are likely just identifiers. These columns were removed from both `train_df` and `test_df` using the `.drop()` method with `axis=1`.

### Categorical Feature Encoding

Several categorical features were encoded into numerical representations using dictionary mapping:
- 'satisfaction': Mapped 'neutral or dissatisfied' to 0 and 'satisfied' to 1.
- 'Gender': Mapped 'Male' to 0 and 'Female' to 1.
- 'Customer Type': Mapped 'disloyal Customer' to 0 and 'Loyal Customer' to 1.
- 'Type of Travel': Mapped 'Business travel' to 0 and 'Personal Travel' to 1.
- 'Class': Mapped 'Eco' to 0, 'Eco Plus' to 1, and 'Business' to 2.
These mappings were applied to the corresponding columns in both `train_df` and `test_df`.

### Creating New Features

Two new features were created:
- 'Flight Haul': Categorized based on 'Flight Distance' into 'short-haul' (<= 1500), 'medium-haul' (<= 3500), and 'long-haul' (> 3500).
- 'Total delay': Calculated as the sum of 'Departure Delay in Minutes' and the difference between 'Arrival Delay in Minutes' and 'Departure Delay in Minutes', which simplifies to 'Arrival Delay in Minutes'.
The 'Total delay' was further categorized into 'Delay' levels: 'on_time' (<= 15), 'minor_delay' (<= 60), 'moderate_delay' (<= 180), and 'major_delay' (> 180). These 'Delay' categories were then mapped to numerical values: 0 for 'on_time', 1 for 'minor_delay', 2 for 'moderate_delay', and 3 for 'major_delay'. The 'Flight Haul' categories were also mapped to numerical values: 1 for 'short-haul', 2 for 'medium-haul', and 3 for 'long-haul'.

### Removing Original Columns

The original columns 'Flight Distance', 'Arrival Delay in Minutes', 'Departure Delay in Minutes', and 'Total delay' were removed from the DataFrames as their information was captured by the new 'Flight Haul' and 'Delay' features.

### Splitting Test Dataset

The original test dataset (`test_df`) was split into two equal parts: an evaluation dataset (`eval_dataset`) and a new test dataset (`test_dataset`). This was done to provide separate datasets for model evaluation during training and for final testing.

### Scaling the Data

Numerical columns in the datasets were scaled using `StandardScaler` from `sklearn.preprocessing`. This process standardizes features by removing the mean and scaling to unit variance. The scaler was fitted on the training data (`train_df`) and then used to transform the numerical columns in `train_df`, `eval_dataset`, and `test_dataset`. Columns that were already encoded categorical variables were excluded from scaling.

### Splitting into Features and Target

Finally, the prepared datasets (`train_df`, `eval_dataset`, and `test_dataset`) were split into feature sets (X_train, X_eval, X_test) and target variable sets (y_train, y_eval, y_test). The target variable for all datasets is 'satisfaction'.

### Saving Prepared Datasets

The prepared datasets (X_train, y_train, X_eval, y_eval, X_test, y_test) were combined back into DataFrames with a 'target' column and saved to Google Drive as CSV files for easy access in subsequent steps. The files were saved to the directory '/content/drive/MyDrive/AirlineSatisfactionDatasetsPrepared' with filenames 'Xy_train.csv', 'Xy_eval.csv', and 'Xy_test.csv'.


-----------


## Model Training

Several machine learning models were trained to predict customer satisfaction. For each model, hyperparameter tuning was performed using `GridSearchCV` with cross-validation to find the best performing parameters based on accuracy. The trained models were then saved to Google Drive.

The following models were trained and tuned:

### Logistic Regression
- **Hyperparameter Tuning**: Performed using `GridSearchCV` to tune 'C', 'penalty', 'solver', and 'max_iter'.

### Decision Tree
- **Hyperparameter Tuning**: Performed using `GridSearchCV` to tune 'max_depth', 'min_samples_split', 'min_samples_leaf', and 'criterion'.

### Random Forest
- **Hyperparameter Tuning**: Performed using `GridSearchCV` to tune 'n_estimators', 'max_depth', 'min_samples_split', and 'min_samples_leaf'.

### Gradient Boosted Trees
- **Hyperparameter Tuning**: Performed using `GridSearchCV` to tune 'n_estimators', 'learning_rate', 'max_depth', 'min_samples_split', and 'min_samples_leaf'.

### LightGBM
- **Hyperparameter Tuning**: Performed using `GridSearchCV` to tune 'n_estimators', 'learning_rate', 'num_leaves', 'max_depth', and 'min_child_samples'. Early stopping was used during training.

### XGBoost
- **Hyperparameter Tuning**: Performed using `GridSearchCV` to tune 'n_estimators', 'learning_rate', 'max_depth', 'min_child_weight', 'gamma', 'subsample', and 'colsample_bytree'.

### KNN
- **Hyperparameter Tuning**: Performed using `GridSearchCV` to tune 'n_neighbors', 'weights', and 'metric'.

### Extra Trees
- **Hyperparameter Tuning**: Performed using `GridSearchCV` to tune 'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', and 'criterion'.

### SGD
- **Hyperparameter Tuning**: Performed using `GridSearchCV` to tune 'loss', 'penalty', 'alpha', 'max_iter', 'learning_rate', and 'eta0'.

### LASSO-LARS
- **Hyperparameter Tuning**: Performed using `GridSearchCV` to tune 'alpha' and 'max_iter'. Note that LassoLars is primarily a regression model, and its use for classification here involved thresholding predictions, making standard classification metrics like ROC AUC not directly applicable.

### Single Layer Perceptron
- **Hyperparameter Tuning**: Performed using `GridSearchCV` to tune 'penalty', 'alpha', 'max_iter', 'tol', and 'eta0'. Note that Perceptron does not have a `predict_proba` method, so ROC AUC was not calculated.

### Neural Network (Keras)
- **Manual Hyperparameter Search**: A manual grid search was performed to explore different combinations of 'hidden_units', 'activation', 'solver', 'l2_reg', and 'lr_schedule'. The best model from this search was saved.
- **KerasClassifier with GridSearchCV**: A Keras model was wrapped using `KerasClassifier` to perform `GridSearchCV` for hyperparameter tuning, specifically tuning 'model__optimizer', 'model__dropout_rate', 'model__learning_rate', and 'model__dense_layer_units'.

------

## Model Comparison and Visualization

To effectively compare the performance of the trained models, several visualizations were generated based on the evaluation metrics. These visualizations helped in understanding the strengths and weaknesses of each model across different metrics and datasets.

- **Heatmap**: A heatmap was created to provide a comprehensive overview of all calculated metrics (Accuracy, Precision, Recall, F1-score, ROC AUC, and MCC) for each model on both the evaluation and test sets. This visualization allowed for quick identification of top-performing models and metrics.
- **Bar Plots for Individual Metrics**: Bar plots were generated for each individual metric, showing the performance of all models on both the evaluation and test datasets. This provided a clear comparison of how each model performed on a specific metric. The models were also sorted by test accuracy to highlight the ranking.
- **Pairwise Scatter Plots**: Pairwise scatter plots were used to visualize the relationships between different evaluation metrics. This helped in understanding if models that perform well on one metric also perform well on others.
- **Confusion Matrix Heatmaps**: For each model and dataset, a heatmap of the confusion matrix was plotted. This visualization provided a detailed breakdown of true positives, true negatives, false positives, and false negatives, offering insights into the types of errors each model made.
- **Radar Charts (Spider Plots)**: Radar charts were generated for individual models to visualize their performance across multiple metrics simultaneously. This provided a radial view of how balanced a model's performance was across different evaluation criteria. Radar charts were also generated for models sorted by test accuracy for comparison.

The visualization plots were saved to Google Drive in PNG format for easy access and inclusion in the README file.

------

## Model Comparison Visualizations

To compare the performance of the trained models, several visualizations were generated:

- **Heatmap**: Provides a comprehensive overview of all evaluation metrics (Accuracy, Precision, Recall, F1-score, ROC AUC, MCC) for each model on both the evaluation and test sets.
  - Heatmap: ![models_metrics_heatmap.png](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/model_metrics_heatmap.png)
  - Sorted Heatmap: ![models_metrics_heatmap_sorted.png](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/model_metrics_heatmap_sorted.png)

- **Bar Plots for Individual Metrics**: Shows the performance of all models on each individual metric for both evaluation and test datasets.
  - Accuracy: ![Accuracy_comparison.png](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/Accuracy_comparison.png)
  - Precision: ![Precision_comparison.png](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/Precision_comparison.png)
  - Recall: ![Recall_comparison.png](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/Recall_comparison.png)
  - F1-score: ![F1-score_comparison.png](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/F1-score_comparison.png)
  - ROC AUC: ![ROC AUC_comparison.png](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/ROC%20AUC_comparison.png)
  - MCC: ![MCC_comparison.png](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/MCC_comparison.png)
  - Sorted Accuracy: ![Accuracy_comparison_sorted.png](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/Accuracy_comparison_sorted.png)


- **Pairwise Scatter Plots**: Visualizes the relationships between different evaluation metrics.
  - Pairwise Metrics: ![pairwise_metrics.png](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/pairwise_metrics.png)

- **Confusion Matrix Heatmaps**: Displays the confusion matrix for each model on both evaluation and test sets.
  - Confusion Matrices:
    - ![best_log_reg_model_Evaluation Set](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_log_reg_model_Evaluation%20Set_confusion_matrix.png)
    - ![best_log_reg_model_Test Set](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_log_reg_model_Test%20Set_confusion_matrix.png)
    - ![best_decision_tree_model_Evaluation Set](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_decision_tree_model_Evaluation%20Set_confusion_matrix.png)
    - ![best_decision_tree_model_Test Set](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_decision_tree_model_Test%20Set_confusion_matrix.png)
    - ![best_random_forest_model_Evaluation Set](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_random_forest_model_Evaluation%20Set_confusion_matrix.png)
    - ![best_random_forest_model_Test Set](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_random_forest_model_Test%20Set_confusion_matrix.png)
    - ![best_gradient_boosted_trees_model_Evaluation Set](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_gradient_boosted_trees_model_Evaluation%20Set_confusion_matrix.png)
    - ![best_gradient_boosted_trees_model_Test Set](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_gradient_boosted_trees_model_Test%20Set_confusion_matrix.png)
    - ![best_lightgbm_model_Evaluation Set](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_lightgbm_model_Evaluation%20Set_confusion_matrix.png)
    - ![best_lightgbm_model_Test Set](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_lightgbm_model_Test%20Set_confusion_matrix.png)
    - ![best_xgboost_model_Evaluation Set](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_xgboost_model_Evaluation%20Set_confusion_matrix.png)
    - ![best_xgboost_model_Test Set](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_xgboost_model_Test%20Set_confusion_matrix.png)
    - ![best_knn_model_Evaluation Set](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_knn_model_Evaluation%20Set_confusion_matrix.png)
    - ![best_knn_model_Test Set](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_knn_model_Test%20Set_confusion_matrix.png)
    - ![best_extra_trees_model_Evaluation Set](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_extra_trees_model_Evaluation%20Set_confusion_matrix.png)
    - ![best_extra_trees_model_Test Set](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_extra_trees_model_Test%20Set_confusion_matrix.png)
    - ![best_sgd_model_Evaluation Set](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_sgd_model_Evaluation%20Set_confusion_matrix.png)
    - ![best_sgd_model_Test Set](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_sgd_model_Test%20Set_confusion_matrix.png)
    - ![best_perceptron_model_Evaluation Set](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_perceptron_model_Evaluation%20Set_confusion_matrix.png)
    - ![best_perceptron_model_Test Set](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_perceptron_model_Test%20Set_confusion_matrix.png)
    - ![best_tf_mlp_model_Evaluation Set](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_tf_mlp_model_Evaluation%20Set_confusion_matrix.png)
    - ![best_tf_mlp_model_Test Set](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_tf_mlp_model_Test%20Set_confusion_matrix.png)

- **Radar Charts (Spider Plots)**: Visualizes a model's performance across multiple metrics in a radial plot.
  - Radar Plots:
    - ![best_tf_mlp_model](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_tf_mlp_model_radar_plot.png)
  - Sorted Radar Plots:
    - ![best_lightgbm_model](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_lightgbm_model_radar_sorted.png)
    - ![best_gradient_boosted_trees_model](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_gradient_boosted_trees_model_radar_sorted.png)
    - ![best_random_forest_model](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_random_forest_model_radar_sorted.png)
    - ![best_xgboost_model](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_xgboost_model_radar_sorted.png)
    - ![best_tf_mlp_model](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_tf_mlp_model_radar_sorted.png)
    - ![best_extra_trees_model](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_extra_trees_model_radar_sorted.png)
    - ![best_decision_tree_model](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_decision_tree_model_radar_sorted.png)
    - ![best_knn_model](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_knn_model_radar_sorted.png)
    - ![best_sgd_model](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_sgd_model_radar_sorted.png)
    - ![best_log_reg_model](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_log_reg_model_radar_sorted.png)
    - ![best_perceptron_model](https://github.com/Ketis1/Airlines-Satisfaction-Prediction/blob/main/files/AirlineSatisfactionEvaluationData/Plots/best_perceptron_model_radar_sorted.png)





