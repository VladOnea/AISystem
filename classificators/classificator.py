import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from training.visualization import visualizeDecisionTree
from data.utils import augmentFeatures
import shap
import joblib
import os
import matplotlib.pyplot as plt



def trainDecisionTree(X_train, X_test, y_train, y_test, model_path='decision_tree_model.joblib'):
    # Check if the model already exists
    if not os.path.exists(model_path):
        # Train decision tree
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        
        # Save the model to disk
        joblib.dump(clf, model_path)
    else:
        # Load the model from disk
        clf = joblib.load(model_path)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    predictions = ['DR' if pred == 1 else 'No DR' for pred in y_pred]

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print("Decision Tree Accuracy:", accuracy)
    print("Decision Tree Classification Report:")
    print(classification_report(y_test, y_pred))

    visualizeDecisionTree(clf, feature_names=['Density', 'Tortuosity'], class_names=['No DR', 'DR'])
    
    return clf, accuracy, predictions

def logistic_regression(X_train, X_test, y_train, y_test, model_path='logistic_regression_model.joblib'):
    # Feature scaling for numeric columns
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Check if the model already exists
    if not os.path.exists(model_path):
        # Train the Logistic Regression model
        model = LogisticRegression(max_iter=10000)
        model.fit(X_train_scaled, y_train)
        
        # Save the model to disk
        joblib.dump((model, scaler), model_path)
    else:
        # Load the model from disk
        model, scaler = joblib.load(model_path)
        X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    predictions = ['DR' if pred == 1 else 'No DR' for pred in y_pred]

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    classification_report_str = classification_report(y_test, y_pred)
    
    # Print the results
    print(f"Logistic Regression Accuracy: {accuracy}")
    print("Logistic Regression Classification Report:")
    print(classification_report_str)
    
    # SHAP explainability
    explainer = shap.LinearExplainer(model, X_train_scaled)
    shap_values = explainer.shap_values(X_test_scaled)
    
    # Summary plot
    shap.summary_plot(shap_values, X_test_scaled, feature_names=['Density', 'Tortuosity'])
    
    # Dependence plot for the first feature
    shap.dependence_plot("Density", shap_values, X_test_scaled, feature_names=['Density', 'Tortuosity'])
    
    # Force plot for the first prediction
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values[0], X_test_scaled[0], feature_names=['Density', 'Tortuosity'])
    
    return model, accuracy, predictions

def naive_bayes(X_train, X_test, y_train, y_test, model_path='naive_bayes_model.joblib'):
    # Check if the model already exists
    if not os.path.exists(model_path):
        # Train the Naive Bayes model
        model = GaussianNB()
        model.fit(X_train, y_train)
        
        # Save the model to disk
        joblib.dump(model, model_path)
    else:
        # Load the model from disk
        model = joblib.load(model_path)
    
    # Make predictions
    y_pred = model.predict(X_test)
    predictions = ['DR' if pred == 1 else 'No DR' for pred in y_pred]

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    classification_report_str = classification_report(y_test, y_pred)
    
    # Print the results
    print(f"Naive Bayes Accuracy: {accuracy}")
    print("Naive Bayes Classification Report:")
    print(classification_report_str)
    
    return model, accuracy, predictions

def compareModels(file_path):
    # Load the dataset and augment features
    df = pd.read_excel(file_path)
    df_class_1 = df[df['Label'] == 1]

    # Apply augmentations to class 1 samples and create new rows
    new_rows = []
    for index, row in df_class_1.iterrows():
        augmented_rows = augmentFeatures(row)
        new_rows.extend(augmented_rows)

    # Convert new rows to DataFrame
    new_rows_df = pd.DataFrame(new_rows)

    # Combine with the original DataFrame
    df_combined = pd.concat([df, new_rows_df], ignore_index=True)

    # Save the new dataset if it doesn't already exist
    output_path = 'Updated_Image_Features.xlsx'
    if not os.path.exists(output_path):
        df_combined.to_excel(output_path, index=False)
    else:
        print(f"{output_path} already exists. Using existing file.")

    # Prepare the data
    X = df_combined[['Density', 'Tortuosity']]
    y = df_combined['Label'].astype(int)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train and evaluate models
    decision_tree_model, dt_accuracy, dt_predictions = trainDecisionTree(X_train, X_test, y_train, y_test)
    logistic_regression_model, lr_accuracy, lr_predictions = logistic_regression(X_train, X_test, y_train, y_test)
    naive_bayes_model, nb_accuracy, nb_predictions = naive_bayes(X_train, X_test, y_train, y_test)

    # Compare accuracies
    print(f"Decision Tree Accuracy: {dt_accuracy}")
    print(f"Logistic Regression Accuracy: {lr_accuracy}")
    print(f"Naive Bayes Accuracy: {nb_accuracy}")
    
    if dt_accuracy > lr_accuracy and dt_accuracy > nb_accuracy:
        print("Decision Tree has the highest accuracy.")
    elif lr_accuracy > dt_accuracy and lr_accuracy > nb_accuracy:
        print("Logistic Regression has the highest accuracy.")
    elif nb_accuracy > dt_accuracy and nb_accuracy > lr_accuracy:
        print("Naive Bayes has the highest accuracy.")
    else:
        print("There is a tie between the models.")

    # Create a DataFrame for the predictions
    results_df = pd.DataFrame({
        'ID': df_combined.loc[X_test.index, 'ID'].values,
        'Decision_Tree': dt_predictions,
        'Logistic_Regression': lr_predictions,
        'Naive_Bayes': nb_predictions
    })
    
    results_df.to_excel('Model_Predictions.xlsx', index=False)


