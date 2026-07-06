import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.preprocessing import load_data, prepare_features
from src.config import MODEL_PATH


print("evaluate.py is running...")

def load_model(path=MODEL_PATH):
    """Load the trained model from disk."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the regression model on the test set.
    Print RMSE, MAE, and R2.
    """
    # TODO: y_pred = model.predict(X_test)
    y_pred = model.predict(X_test)

    # TODO: rmse = np.sqrt(mean_squared_error(y_test, y_pred)
    # TODO: mae  = mean_absolute_error(y_test, y_pred)
    # TODO: r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)


    # TODO: print all three metrics
    print(" Model Evaluation Results")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")

    # TODO: return y_pred
    return y_pred


def plot_actual_vs_predicted(y_test, y_pred):
    """Plot actual vs predicted clicks and residuals."""
    # TODO: Two side-by-side plots:
    # Left: scatter of actual vs predicted with diagonal line
    # Right: residuals scatter with horizontal line at 0
    residuals = y_test - y_pred

    plt.figure(figsize=(12, 5))

    # -----------------------------
    # Plot 1: Actual vs Predicted
    # -----------------------------

    plt.subplot(1,2,1)

    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             color='red', linestyle='--')

    plt.xlabel("Actual Clicks")
    plt.ylabel("Predicted Clicks")
    plt.title("Actual vs Predicted")

    # -----------------------------
    # Plot 2: Residuals
    # -----------------------------
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')

    plt.xlabel("Predicted Clicks")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")




if __name__ == '__main__':
    df = load_data()
    _, X_test, _, y_test = prepare_features(df)
    model = load_model()
    y_pred = evaluate_model(model, X_test, y_test)
    plot_actual_vs_predicted(y_test, y_pred)
    plt.show()


#to  the best model from train.py
model = load_model()
print(f"\nLoaded model: {type(model).__name__}")