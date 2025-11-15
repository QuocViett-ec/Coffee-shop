"""
Machine Learning Models for Time Series Forecasting
- XGBoost
- LightGBM
- Random Forest
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')


class MLForecaster:
    """
    Machine Learning models for time series forecasting
    """

    def __init__(self, model_type='xgboost'):
        """
        Parameters:
        -----------
        model_type : str
            'xgboost', 'lightgbm', or 'random_forest'
        """
        self.model_type = model_type
        self.model = None
        self.feature_importance = None

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate forecasting metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        mbd = np.mean(y_pred - y_true)

        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'MBD': mbd
        }

    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None, params=None):
        """
        Train XGBoost model

        Parameters:
        -----------
        X_train, y_train : Training data
        X_val, y_val : Validation data (optional)
        params : dict, optional
            XGBoost parameters
        """
        print(f"\n[XGBoost] Training...")

        if params is None:
            params = {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }

        self.model = xgb.XGBRegressor(**params)

        # Train model
        self.model.fit(X_train, y_train)

        # Feature importance
        self.feature_importance = pd.Series(
            self.model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)

        print(f"  ✓ Model trained")
        return self.model

    def train_lightgbm(self, X_train, y_train, X_val=None, y_val=None, params=None):
        """
        Train LightGBM model

        Parameters:
        -----------
        X_train, y_train : Training data
        X_val, y_val : Validation data (optional)
        params : dict, optional
            LightGBM parameters
        """
        print(f"\n[LightGBM] Training...")

        if params is None:
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbose': -1
            }

        self.model = lgb.LGBMRegressor(**params)

        # Train model
        self.model.fit(X_train, y_train)

        # Feature importance
        self.feature_importance = pd.Series(
            self.model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)

        print(f"  ✓ Model trained")
        return self.model

    def train_random_forest(self, X_train, y_train, params=None):
        """
        Train Random Forest model

        Parameters:
        -----------
        X_train, y_train : Training data
        params : dict, optional
            Random Forest parameters
        """
        print(f"\n[Random Forest] Training...")

        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            }

        self.model = RandomForestRegressor(**params)
        self.model.fit(X_train, y_train)

        # Feature importance
        self.feature_importance = pd.Series(
            self.model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)

        print(f"  ✓ Model trained")
        return self.model

    def train(self, X_train, y_train, X_val=None, y_val=None, params=None):
        """
        Train model based on model_type

        Parameters:
        -----------
        X_train, y_train : Training data
        X_val, y_val : Validation data
        params : dict, optional
            Model-specific parameters
        """
        if self.model_type == 'xgboost':
            return self.train_xgboost(X_train, y_train, X_val, y_val, params)
        elif self.model_type == 'lightgbm':
            return self.train_lightgbm(X_train, y_train, X_val, y_val, params)
        elif self.model_type == 'random_forest':
            return self.train_random_forest(X_train, y_train, params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)

    def evaluate(self, X, y, dataset_name='Test'):
        """
        Evaluate model on dataset

        Returns:
        --------
        dict : metrics
        """
        y_pred = self.predict(X)
        metrics = self.calculate_metrics(y, y_pred)

        print(f"\n{dataset_name} Set Performance:")
        print(f"  RMSE: ${metrics['RMSE']:.2f}")
        print(f"  MAE: ${metrics['MAE']:.2f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
        print(f"  R²: {metrics['R2']:.4f}")

        return metrics

    def get_feature_importance(self, top_n=20):
        """
        Get top N most important features

        Parameters:
        -----------
        top_n : int, default 20
            Number of top features to return
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance not available. Train model first.")

        return self.feature_importance.head(top_n)

    def save_model(self, filepath):
        """Save model to file"""
        if self.model is None:
            raise ValueError("No model to save. Train model first.")

        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance
        }, filepath)

        print(f"\n✓ Model saved to {filepath}")

    def load_model(self, filepath):
        """Load model from file"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.model_type = data['model_type']
        self.feature_importance = data.get('feature_importance')

        print(f"\n✓ Model loaded from {filepath}")
        return self.model

    def hyperparameter_tuning(self, X_train, y_train, param_grid=None, cv=3):
        """
        Perform hyperparameter tuning using TimeSeriesSplit cross-validation

        Parameters:
        -----------
        X_train, y_train : Training data
        param_grid : dict
            Parameter grid for GridSearchCV
        cv : int, default 3
            Number of folds for TimeSeriesSplit

        Returns:
        --------
        best_params : dict
            Best parameters found
        """
        print(f"\n[{self.model_type.upper()}] Hyperparameter Tuning...")
        print(f"  CV folds: {cv}")

        # Default parameter grids
        if param_grid is None:
            if self.model_type == 'xgboost':
                param_grid = {
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [50, 100, 200],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                }
            elif self.model_type == 'lightgbm':
                param_grid = {
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [50, 100, 200],
                    'num_leaves': [15, 31, 63]
                }
            elif self.model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5, 10]
                }

        # Create base model
        if self.model_type == 'xgboost':
            base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        elif self.model_type == 'lightgbm':
            base_model = lgb.LGBMRegressor(objective='regression', random_state=42, verbose=-1)
        elif self.model_type == 'random_forest':
            base_model = RandomForestRegressor(random_state=42, n_jobs=-1)

        # Time Series Cross Validation
        tscv = TimeSeriesSplit(n_splits=cv)

        # Grid Search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"\n  ✓ Best parameters found:")
        for param, value in grid_search.best_params_.items():
            print(f"    {param}: {value}")

        print(f"\n  Best CV RMSE: ${np.sqrt(-grid_search.best_score_):.2f}")

        # Train final model with best parameters
        self.train(X_train, y_train, params=grid_search.best_params_)

        return grid_search.best_params_


def compare_ml_models(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train and compare all ML models

    Returns:
    --------
    results : pd.DataFrame
        Comparison of all models
    models : dict
        Trained models
    """
    print("\n" + "="*70)
    print("TRAINING MACHINE LEARNING MODELS")
    print("="*70)

    results = {}
    models = {}

    model_types = ['xgboost', 'lightgbm', 'random_forest']

    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"Model: {model_type.upper()}")
        print(f"{'='*70}")

        # Train model
        forecaster = MLForecaster(model_type=model_type)
        forecaster.train(X_train, y_train, X_val, y_val)

        # Evaluate
        train_metrics = forecaster.evaluate(X_train, y_train, 'Train')
        val_metrics = forecaster.evaluate(X_val, y_val, 'Validation')
        test_metrics = forecaster.evaluate(X_test, y_test, 'Test')

        results[model_type] = test_metrics
        models[model_type] = forecaster

        # Feature importance
        print(f"\nTop 10 Features:")
        top_features = forecaster.get_feature_importance(top_n=10)
        for i, (feat, imp) in enumerate(top_features.items(), 1):
            print(f"  {i}. {feat}: {imp:.4f}")

    # Compare results
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('MAPE')

    print("\n" + "="*70)
    print("ML MODELS COMPARISON")
    print("="*70)
    print(results_df.to_string())
    print("\n" + "="*70)

    best_model = results_df['MAPE'].idxmin()
    print(f"\n🏆 Best ML Model: {best_model.upper()}")
    print(f"   MAPE: {results_df.loc[best_model, 'MAPE']:.2f}%")
    print(f"   RMSE: ${results_df.loc[best_model, 'RMSE']:.2f}")
    print("="*70)

    return results_df, models


if __name__ == "__main__":
    # Test ML models
    print("Testing ML Models...\n")

    # Load data
    X = pd.read_csv('../../data/processed/X.csv', index_col='date', parse_dates=True)
    y = pd.read_csv('../../data/processed/y.csv', index_col='date', parse_dates=True).squeeze()

    # Split
    n = len(X)
    train_size = int(n * 0.8)
    val_size = int(n * 0.1)

    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]

    X_val = X.iloc[train_size:train_size+val_size]
    y_val = y.iloc[train_size:train_size+val_size]

    X_test = X.iloc[train_size+val_size:]
    y_test = y.iloc[train_size+val_size:]

    # Train and compare
    results, models = compare_ml_models(X_train, y_train, X_val, y_val, X_test, y_test)
