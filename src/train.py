import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import lightgbm as lgb
from scipy.stats import randint, loguniform, uniform

from src.preprocessing import load_data, prepare_features
from src.config import MODEL_PATH, RANDOM_STATE


def get_models_and_params():
    """
    Define all regression models and their hyperparameter search spaces.
    Note: This is regression — no class_weight needed.
    """
    models = {
        'Decision Tree': (
            DecisionTreeRegressor(random_state=RANDOM_STATE),
            {
                'max_depth': randint(3, 15),
                'min_samples_leaf': randint(1, 20)
            }
        ),

        # TODO: Add Random Forest Regressor
        "Random Forest": (
            RandomForestRegressor(random_state=RANDOM_STATE),
            {
                "n_estimators": randint(100, 300),
                "max_depth": randint(5, 20),
                "min_samples_split": randint(2, 10)
            }
        ),

        # TODO: Add XGBoost Regressor
        "XGBoost": (
            XGBRegressor(
                random_state=RANDOM_STATE,
                objective="reg:squarederror",
                n_jobs=-1
            ),
            {
                "n_estimators": randint(100, 300),
                "max_depth": randint(3, 10),
                "learning_rate": loguniform(1e-3, 0.3),
                "subsample": uniform(0.6, 0.4)
            }
        ),

        # TODO: Add LightGBM Regressor
        "LightGBM": (
            lgb.LGBMRegressor(
                random_state=RANDOM_STATE
            ),
            {
                "n_estimators": randint(100, 300),
                "num_leaves": randint(20, 60),
                "learning_rate": loguniform(1e-3, 0.3)
            }
        ),

        # TODO: Add Linear Regression (no hyperparameters needed)
        "Linear Regression": (
            LinearRegression(),
            {}
        )
    }

    return models


def tune_and_compare(models, X_train, y_train, X_test, y_test, n_iter=20, cv=5):
    """
    Run RandomizedSearchCV. Use scoring='neg_root_mean_squared_error'.
    """
    results = []
    best_models = {}

    for name, (model, params) in models.items():
        print(f'Tuning {name}...')

        # TODO: RandomizedSearchCV with scoring='neg_root_mean_squared_error'
        if len(params) > 0:
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=params,
                n_iter=n_iter,
                scoring='neg_root_mean_squared_error',
                cv=cv,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=0
            )

            # Fit tuned model
            search.fit(X_train, y_train)
            best_model = search.best_estimator_

        else:
            # Linear Regression (no tuning)
            model.fit(X_train, y_train)
            best_model = model

        # TODO: Fit, predict, compute RMSE, MAE, and R2
        y_pred = best_model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # TODO: Append results and store best model
        results.append({
            "Model": name,
            "Test RMSE": rmse,
            "Test MAE": mae,
            "Test R2": r2
        })

        best_models[name] = best_model

    results_df = pd.DataFrame(results).sort_values('Test RMSE', ascending=True)

    return results_df, best_models


def save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f'Model saved: {path}')


if __name__ == "__main__":
    print("Starting training pipeline...")

    df = load_data()
    print(f"Data loaded: {df.shape}")

    X_train, X_test, y_train, y_test = prepare_features(df)
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    models = get_models_and_params()

    print("\nTraining models...\n")
    results_df, best_models = tune_and_compare(
        models, X_train, y_train, X_test, y_test
    )

    print("Model Comparison:")
    print(results_df)

    best_name = results_df.iloc[0]["Model"]
    best_model = best_models[best_name]

    print(f"Best model: {best_name}")

    save_model(best_model)

    print(" Training completed successfully!")