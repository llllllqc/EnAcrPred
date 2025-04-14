import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import Lasso
from sklearn.preprocessing import scale
from sklearn.feature_selection import SelectFromModel

warnings.filterwarnings("ignore")

def LASSO_feature_selection(input_csv, output_csv, alpha_value=0.001):
    data = pd.read_csv(input_csv)
    features = data.drop(columns=['Label'])
    labels = data['Label']
    features_scaled = scale(features)
    clf = Lasso(alpha=alpha_value, max_iter=500).fit(features_scaled, labels)
    model = SelectFromModel(clf, prefit=True)
    support = model.get_support()
    selected_cols = features.columns[support]
    selected_features = features[selected_cols]
    selected_features['Label'] = labels.values
    output_csv_path = output_csv.replace('.csv', f'_13.csv')
    selected_features.to_csv(output_csv_path, index=False)
    print(f"Feature matrix shape: {selected_features.shape}")
    print(f"CSV file saved as: {output_csv_path}")

if __name__ == '__main__':
    input_csv = 'Feature_13.csv'
    output_csv = 'selected_features_.csv'
    LASSO_feature_selection(input_csv, output_csv, alpha_value=0.01)
