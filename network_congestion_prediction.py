import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib  # For saving models to disk
import os
from datetime import datetime

# -----------------------------------------------------------------------------
# Helper Function: Loading Data
# -----------------------------------------------------------------------------
def load_data(file_path):
    """
    This function reads a CSV file and returns the data as a pandas DataFrame.
    If there is an error (for example, if the file isn't found), it prints an error message.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# -----------------------------------------------------------------------------
# Data Preprocessing Function
# -----------------------------------------------------------------------------
def preprocess_data(data, preprocessor=None, training=True):
    """
    This function prepares (or "preprocesses") the dataset so it can be used by the machine learning models.
    
    For a non-technical explanation:
    - We start by separating out the information we want the computer to learn from ("features") and 
      what we want it to predict ("targets").
    - Our targets in this case are:
         1. Binary Congestion: 0 means "no congestion" and 1 means "congestion exists".
         2. Continuous Congestion Level: A numeric scale indicating how bad the congestion is.
    - We also clean up the data by replacing missing numbers, scaling numeric values, 
      and converting text entries into numbers with one-hot encoding.
      
    When training=True, the function builds a new processor; when False (such as when making predictions),
    it uses an already built processor.
    """
    if training:
        # Remove columns that we do NOT want to use for learning.
        # We drop the columns for our predictions ('Congestion (Binary)' and 'Continuous Congestion Level')
        # as well as the 'Congestion Severity' column since we are no longer using it.
        X = data.drop(columns=['Congestion (Binary)', 'Continuous Congestion Level', 'Congestion Severity'])
        # Target for binary classification: indicates if congestion exists (1) or not (0)
        y_binary = data['Congestion (Binary)']
        # Target for regression: a continuous score showing how severe the congestion is.
        y_continuous = data['Continuous Congestion Level']
    else:
        # When not training, we assume the data has already been cleaned and processed.
        X = data

    # Separate the remaining columns into two groups based on data type:
    # - Categorical columns (text or category) that need encoding.
    # - Numerical columns (numbers) that might need scaling.
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if training:
        # Create a processing "pipeline" for the numerical columns:
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Replace missing numbers with the column mean.
            ('scaler', StandardScaler())  # Standardize numbers to have zero mean and unit variance.
        ])

        # Create a processing "pipeline" for the categorical columns:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Replace missing text with the most common entry.
            ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Convert text to numbers.
        ])

        # Combine the numeric and categorical pipelines into one transformer.
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )
        # Fit the preprocessor to the data & transform it.
        X_preprocessed = preprocessor.fit_transform(X)
        print("Data preprocessing completed.")
        return X_preprocessed, y_binary, y_continuous, preprocessor
    else:
        # For new data (like for making predictions), simply apply the already fitted preprocessor.
        X_preprocessed = preprocessor.transform(X)
        print("Prediction data preprocessing completed.")
        return X_preprocessed, preprocessor

# -----------------------------------------------------------------------------
# Cross-Validation Function: Check Model Performance using Different Data Splits
# -----------------------------------------------------------------------------
def cross_validate_model(model, X, y, cv=5):
    """
    This function checks how well a model performs by splitting the data into parts (folds)
    and training/evaluating on them. This gives us a more reliable idea of performance.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"Cross-validation accuracy scores: {scores}")
    print(f"Mean accuracy: {np.mean(scores) * 100:.2f}%")
    print(f"Standard Deviation: {np.std(scores) * 100:.2f}%")
    return scores

# -----------------------------------------------------------------------------
# Learning Curve Plotting Function: Visual Diagnosis of Model Learning
# -----------------------------------------------------------------------------
def plot_learning_curve(model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    This function plots the learning curve of a model which shows the accuracy on the 
    training data vs. accuracy on cross-validated data. It helps us see if the model is learning well.
    """
    train_sizes, train_scores, val_scores = learning_curve(model, X, y, cv=cv, train_sizes=train_sizes,
                                                           scoring='accuracy')
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, val_scores_mean, 'o-', color='g', label='Cross-validation score')
    plt.title('Learning Curve')
    plt.xlabel('Training Examples')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# -----------------------------------------------------------------------------
# Hyperparameter Tuning Function for the Binary Classification Model
# -----------------------------------------------------------------------------
def tune_rf_classifier(X, y, cv=5):
    """
    This function automatically searches for the best settings (parameters) for the RandomForestClassifier.
    It uses a technique called Grid Search on a set of possible parameter values.
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3, 5]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    print("Best parameters for RandomForestClassifier:", grid_search.best_params_)
    print(f"Best cross-validation score: {grid_search.best_score_ * 100:.2f}%")
    return grid_search.best_estimator_

# -----------------------------------------------------------------------------
# Model Training Functions
# -----------------------------------------------------------------------------
def train_binary_classification(X, y):
    """
    This function trains a RandomForest model to decide if there is a congestion (binary: yes/no).
    "Binary" here means there are only two outcomes.
    """
    rf_clf = RandomForestClassifier(random_state=42, max_depth=10, min_samples_split=5, min_samples_leaf=3)
    print("\nPerforming 5-fold cross-validation for Binary Classification:")
    cross_validate_model(rf_clf, X, y, cv=5)

    # Split the data into two parts: one for training the model and one for testing how well it works.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_clf.fit(X_train, y_train)
    y_pred_rf_train = rf_clf.predict(X_train)
    y_pred_rf_test = rf_clf.predict(X_test)
    print("\nRandom Forest Binary Classification Results:")
    print(f"Training Accuracy: {accuracy_score(y_train, y_pred_rf_train) * 100:.2f}%")
    print(f"Testing Accuracy: {accuracy_score(y_test, y_pred_rf_test) * 100:.2f}%")
    print(classification_report(y_test, y_pred_rf_test))
    return rf_clf

def train_numerical_estimation(X, y):
    """
    This function trains a RandomForest regression model to estimate a continuous congestion level.
    In simple words, it learns to predict a number that indicates how severe the congestion is.
    """
    rf_reg = RandomForestRegressor(random_state=42, max_depth=10, min_samples_split=5, min_samples_leaf=3)
    print("\nPerforming 5-fold cross-validation for Numerical Estimation (RMSE):")
    # For regression, we use RMSE as our performance metric.
    scores = cross_val_score(rf_reg, X, y, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    print(f"Cross-validation RMSE scores: {rmse_scores}")
    print(f"Mean RMSE: {np.mean(rmse_scores):.2f}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_reg.fit(X_train, y_train)
    y_pred_rf_train = rf_reg.predict(X_train)
    y_pred_rf_test = rf_reg.predict(X_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_rf_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_rf_test))
    print(f"\nRandom Forest RMSE - Training: {rmse_train:.2f}, Testing: {rmse_test:.2f}")
    return rf_reg

# -----------------------------------------------------------------------------
# Function to Save the Trained Models and Preprocessor
# -----------------------------------------------------------------------------
def save_models(models, preprocessor, folder_path="saved_models"):
    """
    This function saves the trained models and the preprocessor to disk so that they can be re-used later.
    We save:
      - The binary classification model.
      - The numerical estimation model.
      - And the preprocessor (to transform new data in the same way).
    """
    os.makedirs(folder_path, exist_ok=True)
    joblib.dump(models['binary'], f"{folder_path}/random_forest_binary.joblib")
    joblib.dump(models['numerical'], f"{folder_path}/random_forest_numerical.joblib")
    joblib.dump(preprocessor, f"{folder_path}/preprocessor.joblib")
    print(f"\nModels and preprocessor saved to '{folder_path}'.")

# -----------------------------------------------------------------------------
# Prediction Functions for New Data
# -----------------------------------------------------------------------------
def predict_binary(model, preprocessor, input_data):
    """
    Given new data, this function preprocesses it and returns the binary congestion predictions.
    """
    input_data_preprocessed = preprocessor.transform(input_data)
    return model.predict(input_data_preprocessed)

def predict_continuous(model, preprocessor, input_data):
    """
    Given new data, this function preprocesses it and returns the continuous congestion level predictions.
    A scaling is applied to adjust the output to a range of 0 - 100.
    """
    input_data_preprocessed = preprocessor.transform(input_data)
    predictions = model.predict(input_data_preprocessed)
    scaler = MinMaxScaler(feature_range=(0, 100))
    return scaler.fit_transform(predictions.reshape(-1, 1)).flatten()

# -----------------------------------------------------------------------------
# Simulate a Load Balancer:
# This applies decision logic based on the predictions.
# -----------------------------------------------------------------------------
def simulate_load_balancer(predictions_df, threshold=70):
    """
    For each record in the predictions:
      - If there is congestion (binary value is 1) or the congestion level is above a certain threshold,
        the decision is to use a backup server.
      - Otherwise, the primary server is used.
    This function prints the decision for each record and adds the decision details to the DataFrame.
    """
    decisions = []
    router_assignments = []
    for idx, row in predictions_df.iterrows():
        binary = row['Binary Congestion']
        congestion_level = row['Continuous Congestion Level']
        if binary == 1 or congestion_level >= threshold:
            decision = "Route to Backup Server"
            router = np.random.choice(["Backup Router A", "Backup Router B", "Backup Router C"])
        else:
            decision = "Route via Primary Server"
            router = "Primary Router"
        decisions.append(decision)
        router_assignments.append(router)
        print(f"Record {idx}: {decision} to {router} (Congestion Level: {congestion_level:.2f})")
    predictions_df['Routing Decision'] = decisions
    predictions_df['Router'] = router_assignments
    return predictions_df

# -----------------------------------------------------------------------------
# Main Execution Code
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Step 1: Load the training dataset.
    train_file_path = input("Enter the path to your training dataset (CSV file): ")
    train_data = load_data(train_file_path)
    if train_data is not None:
        # Preprocess the training data:
        # This separates the information (features) from what we want to predict.
        # We extract:
        #  - X_train: the input features (all information except the target columns)
        #  - y_binary_train: target for binary classification (0 or 1 indicating congestion)
        #  - y_continuous_train: target for numerical estimation (a number indicating congestion level)
        #  - preprocessor: the tools used to clean and process the data.
        X_train, y_binary_train, y_continuous_train, preprocessor = preprocess_data(train_data, training=True)

        # Optionally, display a learning curve for the binary classification model.
        plot_lc = input("Do you want to display the learning curve for binary classification? (y/n): ")
        if plot_lc.lower() == "y":
            model_for_lc = RandomForestClassifier(random_state=42, max_depth=10, min_samples_split=5,
                                                  min_samples_leaf=3)
            print("Plotting learning curve for binary classification...")
            plot_learning_curve(model_for_lc, X_train, y_binary_train, cv=5)

        # Optionally, perform hyperparameter tuning for the binary classification model.
        tune_binary = input("Do you want to perform hyperparameter tuning for binary classification? (y/n): ")
        if tune_binary.lower() == "y":
            print("Tuning binary classification model...")
            binary_model = tune_rf_classifier(X_train, y_binary_train, cv=5)
        else:
            print("\nTraining Binary Classification Model...")
            binary_model = train_binary_classification(X_train, y_binary_train)

        # Train the model that predicts the continuous congestion level.
        print("\nTraining Numerical Estimation Model...")
        numerical_model = train_numerical_estimation(X_train, y_continuous_train)

        # Save the trained models and processing tools to disk.
        models = {'binary': binary_model, 'numerical': numerical_model}
        save_models(models, preprocessor)

        # Step 2: Load the dataset for making predictions.
        predict_file_path = input("\nEnter the path to your prediction dataset (CSV file): ")
        predict_data = load_data(predict_file_path)
        if predict_data is not None:
            print("\nMaking Predictions on the Provided Dataset:")
            binary_predictions = predict_binary(binary_model, preprocessor, predict_data)
            continuous_predictions = predict_continuous(numerical_model, preprocessor, predict_data)
            # Create a new DataFrame containing the predictions.
            predictions_df = pd.DataFrame({
                'Binary Congestion': binary_predictions,
                'Continuous Congestion Level': continuous_predictions
            })
            predictions_df.to_csv('predictions.csv', index=False)
            print("\nPredictions saved to 'predictions.csv'")

            # Step 3: Simulate the load balancing decisions based on the predictions.
            print("\nSimulating Load Balancer Decisions Based on Predictions:")
            balanced_predictions_df = simulate_load_balancer(predictions_df, threshold=70)
            # Add a timestamp and flag potential anomalies based on high congestion level.
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            balanced_predictions_df["Timestamp"] = timestamp
            ANOMALY_THRESHOLD = 90  # If the congestion level is above this value, it is flagged.
            balanced_predictions_df["Anomaly"] = balanced_predictions_df["Continuous Congestion Level"].apply(
                lambda x: "Yes" if x > ANOMALY_THRESHOLD else "No"
            )
            balanced_predictions_df.to_csv('balanced_predictions.csv', index=False)
            print("\nLoad balancing decisions saved to 'balanced_predictions.csv'")

            # To keep a historical record, append the current predictions to 'historical_predictions.csv'.
            historical_file = "historical_predictions.csv"
            if os.path.exists(historical_file):
                balanced_predictions_df.to_csv(historical_file, mode='a', header=False, index=False)
            else:
                balanced_predictions_df.to_csv(historical_file, index=False)
            print("\nHistorical records updated in 'historical_predictions.csv'")