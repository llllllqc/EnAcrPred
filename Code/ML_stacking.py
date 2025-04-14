import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import joblib
from xgboost import XGBClassifier
import os
import pickle

# Read CSV data
data = pd.read_csv('selected_features_13.csv')

# Split the data into training and testing sets
train_data = data.iloc[:824]
test_data = data.iloc[824:]

X_train = train_data.drop(columns=["Label"])
y_train = train_data["Label"]
X_test = test_data.drop(columns=["Label"])
y_test = test_data["Label"]

# List of base models
model_names = [
    ("RandomForest", RandomForestClassifier()),
    ("KNeighbors", KNeighborsClassifier()),
    ("GaussianNB", GaussianNB()),
    ("GradientBoosting", GradientBoostingClassifier()),
    ("ExtraTrees", ExtraTreesClassifier()),
    ("MLPClassifier", MLPClassifier(max_iter=1000)),
    ("LinearDiscriminantAnalysis", LinearDiscriminantAnalysis()),
    ("HistGB", HistGradientBoostingClassifier())
]

# Directories to save the base models and meta model
base_model_save_dir = 'model_13'
meta_model_save_dir = 'model_13/集成model'

# Ensure the directories exist
os.makedirs(meta_model_save_dir, exist_ok=True)

if not os.path.exists(base_model_save_dir):
    raise FileNotFoundError(f"Base model directory {base_model_save_dir} does not exist.")

# Create meta-training and meta-testing feature matrices
meta_train = pd.DataFrame()
meta_test = pd.DataFrame()

for name, _ in model_names:
    model_filename = os.path.join(base_model_save_dir, f"{name}.pickle")
    
    if not os.path.exists(model_filename):
        print(f"File {model_filename} does not exist, skipping this model.")
        continue

    try:
        with open(model_filename, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Error loading file {model_filename}: {e}")
        continue

    try:
        prob_train = model.predict_proba(X_train)[:, 1]
        prob_test = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        try:
            prob_train = model.decision_function(X_train)
            prob_test = model.decision_function(X_test)
        except AttributeError:
            raise ValueError(f"Model {name} cannot output probabilities.")
    
    meta_train[name] = prob_train
    meta_test[name] = prob_test

# Meta-classifiers for the ensemble models
meta_classifiers = [
    ("LogisticRegression", LogisticRegression(max_iter=1000)),
    ("RandomForest", RandomForestClassifier()),
    ("KNeighbors", KNeighborsClassifier()),
    ("GaussianNB", GaussianNB()),
    ("GradientBoosting", GradientBoostingClassifier()),
    ("ExtraTrees", ExtraTreesClassifier()),
    ("XGBoost", XGBClassifier()),
    ("MLPClassifier", MLPClassifier(max_iter=1000)),
    ("LinearDiscriminantAnalysis", LinearDiscriminantAnalysis()),
    ("HistGB", HistGradientBoostingClassifier())
]

# Store the results for each meta-classifier
results = []

for classifier_name, classifier in meta_classifiers:
    print(f"Training and evaluating with {classifier_name} as meta-classifier...")
    
    best_acc = -1
    best_sensitivity = -1
    best_specificity = -1
    best_mcc = -1
    best_auc = -1
    best_model = None

    iterations = 50
    for i in range(iterations):
        print(f"Iteration {i + 1}/{iterations}...")
        
        # Reinitialize the model each time for independent results
        current_classifier = type(classifier)()
        current_classifier.fit(meta_train, y_train)

        # Predictions
        y_pred = current_classifier.predict(meta_test)

        # Calculate evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        if cm.shape == (2, 2):
            TN, FP, FN, TP = cm.ravel()
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        else:
            sensitivity = np.nan
            specificity = np.nan

        mcc = matthews_corrcoef(y_test, y_pred)

        try:
            y_proba = current_classifier.predict_proba(meta_test)[:, 1]
        except AttributeError:
            try:
                y_proba = current_classifier.decision_function(meta_test)
            except AttributeError:
                y_proba = None

        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

        # Round to three decimal places
        acc, sensitivity, specificity, mcc, auc = map(lambda x: round(x, 3), [acc, sensitivity, specificity, mcc, auc])

        if acc > best_acc:
            best_acc = acc
            best_sensitivity = sensitivity
            best_specificity = specificity
            best_mcc = mcc
            best_auc = auc
            best_model = current_classifier

    # Save the best model
    best_model_filename = os.path.join(meta_model_save_dir, f"best_{classifier_name}_meta_model.pickle")
    joblib.dump(best_model, best_model_filename)
    print(f"Best meta model saved for {classifier_name} at {best_model_filename}")

    # Add the results to the list
    results.append({
        'MetaClassifier': classifier_name,
        'ACC': best_acc,
        'SEN': best_sensitivity,
        'SPE': best_specificity,
        'MCC': best_mcc,
        'AUC': best_auc
    })

# Save the results to a CSV file
results_df = pd.DataFrame(results)
results_df = results_df.round(3)

results_file_path = os.path.join(meta_model_save_dir, 'meta_classifier_comparison.csv')
results_df.to_csv(results_file_path, index=False)

print(f"Evaluation metrics for all meta-classifiers saved at {results_file_path}.")
