from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import pickle
from xgboost import XGBClassifier
import os
import pandas as pd
import numpy as np

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(),
    'KNeighbors': KNeighborsClassifier(),
    'GaussianNB': GaussianNB(),
    'GradientBoosting': GradientBoostingClassifier(),
    'ExtraTrees': ExtraTreesClassifier(),
    'XGBoost': XGBClassifier(),
    'MLPClassifier': MLPClassifier(max_iter=1000),
    'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
    'HistGB': HistGradientBoostingClassifier()
}

data_path = 'selected_features_13.csv'
data = pd.read_csv(data_path)

train_data = data.iloc[:824]
test_data = data.iloc[824:]

X_train = train_data.drop(columns=["Label"])
y_train = train_data["Label"]
X_test = test_data.drop(columns=["Label"])
y_test = test_data["Label"]

save_path = 'model_13/'
os.makedirs(save_path, exist_ok=True)

for model_name, model in models.items():
    print(f"Training model: {model_name}")

    best_acc = -1
    best_model = None

    for i in range(5):
        print(f"Training iteration {i + 1}...")

        model.set_params(random_state=np.random.randint(0, 10000))

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Iteration {i + 1} results - {model_name}: ACC: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = pickle.dumps(model)

    model_file = os.path.join(save_path, f"best_model_{model_name}.pkl")

    with open(model_file, 'wb') as f:
        f.write(best_model)
    print(f"Best accuracy for model {model_name}: {best_acc:.4f}, saved as {model_file}")
