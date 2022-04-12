DATA_PATH = 'data/test.csv'
BEST_MODEL_NAME = 'SVM'
BEST_MODEL_PATH = 'models/svm.pkl'

BASE_URL = 'http://127.0.0.1:8000'

available_models = ['SVM', 'Logistic Regression', 'Random Forest', 'KNN', 'XGBoost', 'LightGBM']

model_names_mapping = {
    'SVM': 'svm',
    'Logistic Regression': 'logistic_regression',
    'Random Forest': 'random_forest',
    'KNN': 'knn',
    'XGBoost': 'xgboost',
    'LightGBM': 'lightgbm'
}

model_names_reverse_mapping = {
    'svm': 'SVM',
    'logistic_regression': 'Logistic Regression',
    'random_forest': 'Random Forest',
    'knn': 'KNN',
    'xgboost': 'XGBoost',
    'lightgbm': 'LightGBM'
}

model_path_mapping = {
    'svm': 'models/svm.pkl',
    'logistic_regression': 'models/logr.pkl',
    'random_forest': 'models/rf.pkl',
    'knn': 'models/knn.pkl',
    'xgboost': 'models/xgb.pkl',
    'lightgbm': 'models/lgbm.pkl'
}

model_performance_df_path = 'data/performance_results.csv'
