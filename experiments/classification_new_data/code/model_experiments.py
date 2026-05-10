from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, root_mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass  # already initialized by the notebook before this import

if gpus:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

from mapie.utils import train_conformalize_test_split
from mapie.regression import ConformalizedQuantileRegressor
from mapie.metrics.regression import regression_coverage_score, regression_mean_width_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
})
plt.style.use('seaborn-v0_8-whitegrid')
import json
import os

from constants import OUTPUT_PATH

from tqdm import tqdm

class EpochTqdm(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, desc="Epochs"):
        super().__init__()
        self.pbar = tqdm(total=total_epochs, desc=desc, unit="epoch")
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # show loss/val_loss nicely (change keys as needed)
        postfix = {}
        if "loss" in logs: postfix["loss"] = f"{logs['loss']:.4f}"
        if "val_loss" in logs: postfix["val_loss"] = f"{logs['val_loss']:.4f}"
        if postfix:
            self.pbar.set_postfix(postfix)
        self.pbar.update(1)
    def on_train_end(self, logs=None):
        self.pbar.close()



class Experiment:
    def __init__(self, X, y, train_size=0.8, test_size=0.1, val_size=0.1, split_type='train-val-test', print_stats=None):
        self.X = X
        self.y = y
        
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
        self.__split_data(train_size, test_size, val_size, split_type)
        if print_stats:
            self.__print_data_summary(train_size, test_size, val_size, split_type)
    
    def __split_data(self, train_size, test_size, val_size, split_type):
        if split_type == 'train-val-test':
            if not np.isclose(train_size + test_size + val_size, 1.0):
                raise ValueError("train_size, test_size, and val_size must sum to 1.0")
            
            if train_size == 1.0:
                self.X_train, self.y_train = self.X, self.y
                return

            X_train, X_temp, y_train, y_temp = train_test_split(
                self.X, self.y, train_size=train_size, random_state=42
            )
            
            self.X_train, self.y_train = X_train, y_train
            
            remaining_size = val_size + test_size
            if np.isclose(remaining_size, 0.0):
                return 

            relative_test_size = test_size / remaining_size
            
            if np.isclose(relative_test_size, 1.0):
                self.X_test, self.y_test = X_temp, y_temp
            elif np.isclose(relative_test_size, 0.0):
                self.X_val, self.y_val = X_temp, y_temp
            else:
                self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
                    X_temp, y_temp, test_size=relative_test_size, random_state=42
                )

        elif split_type == 'train-test':
            if not np.isclose(train_size + test_size, 1.0):
                raise ValueError("train_size and test_size must sum to 1.0")
            
            if train_size == 1.0:
                self.X_train, self.y_train = self.X, self.y
            elif test_size == 1.0:
                self.X_test, self.y_test = self.X, self.y
            else:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    self.X, self.y, train_size=train_size, random_state=42
                )
        
        else:
            raise ValueError(f"Unknown split_type: {split_type}. Must be 'train-val-test' or 'train-test'.")
    
    def __print_data_summary(self, train_size, test_size, val_size, split_type):
        total_samples = len(self.X)
        train_samples = len(self.X_train) if self.X_train is not None else 0
        val_samples = len(self.X_val) if self.X_val is not None else 0
        test_samples = len(self.X_test) if self.X_test is not None else 0

        print(f"\n--- Experiment Data Initialized ---")
        print(f"Total Samples: {total_samples}")
        print(f"Split Type:    '{split_type}'")
        
        if split_type == 'train-val-test':
            print(f"  - Train: {train_size*100:>6.1f}% ({train_samples} samples)")
            print(f"  - Val:   {val_size*100:>6.1f}% ({val_samples} samples)")
            print(f"  - Test:  {test_size*100:>6.1f}% ({test_samples} samples)")
        elif split_type == 'train-test':
            print(f"  - Train: {train_size*100:>6.1f}% ({train_samples} samples)")
            print(f"  - Test:  {test_size*100:>6.1f}% ({test_samples} samples)")
        
        print(f"Total in splits: {train_samples + val_samples + test_samples}")
        print("-----------------------------------\n")


class ClassificationExperiment(Experiment):
    def __init__(self, X, y, satellite_name, labels, train_size=0.8, test_size=0.1, val_size=0.1, 
                 split_type='train-val-test', print_stats=True, type='censored'):
        super().__init__(X, y, train_size, test_size, val_size, split_type, print_stats)
        self.satellite_name = satellite_name
        self.results = {}
        self.results_path = OUTPUT_PATH / f"classification_{type}"
        print(f"Results → {self.results_path}")
        self.labels = labels

        self.__scale_data()

    def __scale_data(self):
        self.ordinal_encoder = OrdinalEncoder(categories=[self.labels])
        self.target_encoder = TargetEncoder(categories=self.labels)


        self.y_train = self.ordinal_encoder.fit_transform(self.y_train)
        self.y_test = self.ordinal_encoder.transform(self.y_test)

        self.y_train = self.y_train.reshape(-1, )
        self.y_test = self.y_test.reshape(-1, )

    def run_experiment(self):
        if self.X_train is None or self.y_train is None:
            print("Error: Training data is not available. Cannot run experiment.")
            return

        models_to_run = {
            'rf':  RandomForestClassifier(random_state=42, n_jobs=-1),
            'xgb': XGBClassifier(random_state=42, eval_metric='mlogloss',
                                  device='cuda', tree_method='hist'),
            'ada': AdaBoostClassifier(random_state=42),
            'svc': SVC(probability=True, random_state=42)
        }

        os.makedirs(self.results_path, exist_ok=True)

        for name, model in models_to_run.items():
            print(f"\n--- Running Model: {name.upper()} ---")

            model.fit(self.X_train, self.y_train)

            model_results = {}

            if self.X_test is not None:

                test_preds = model.predict(self.X_test)
                test_accuracy = accuracy_score(self.y_test, test_preds)

                # Get dict report for clean JSON saving
                report_dict = classification_report(self.y_test, test_preds, zero_division=0, output_dict=True, target_names=self.labels)

                model_results['test_accuracy'] = test_accuracy
                model_results['test_classification_report'] = report_dict

                self.plot_confusion_matrix(self.y_test, test_preds, name)

            self.results[name] = model_results['test_classification_report']

            print(f"Test Acc - {model_results['test_accuracy']*100:.4f}%")

        metrics_filename = os.path.join(self.results_path, f"metrics_{self.satellite_name}.json")

        try:
            with open(metrics_filename, 'w') as f:
                json.dump(self.results, f, indent=4)
            print("Metrics saved successfully.")
        except Exception as e:
            print(f"Error saving metrics as JSON: {e}")

        return self.results

    def plot_confusion_matrix(self, y_true, y_preds, model_name):
        plot_dir = self.results_path / "plots"
        os.makedirs(plot_dir, exist_ok=True)

        cm = confusion_matrix(y_true, y_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.labels)

        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, colorbar=True, cmap='Blues')
        acc = accuracy_score(y_true, y_preds)
        ax.set_title(f'{self.satellite_name}: {model_name.upper()} Confusion Matrix (Acc: {acc*100:.2f}%)', fontsize=13)
        plt.tight_layout()

        plot_path = plot_dir / f"{self.satellite_name}_{model_name}_confusion_matrix.png"
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()


class RegressionExperiment(Experiment):
    def __init__(self, X, y, satellite, train_size=0.8, test_size=0.1, val_size=0.1, 
                 split_type='train-val-test', print_stats=None, type='censored'):
        super().__init__(X, y, train_size, test_size, val_size, split_type, print_stats)
        self.satellite = satellite
        self.results_path = OUTPUT_PATH / f"ml_experiment_{type}"
        os.makedirs(self.results_path, exist_ok=True)
        print(f"Results → {self.results_path}")

        self.__scale_data()


    def __scale_data(self):
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()

        self.X_train_scaled = self.x_scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.x_scaler.transform(self.X_test)

        self.y_train_scaled = self.y_scaler.fit_transform(self.y_train)
        self.y_test_scaled = self.y_scaler.transform(self.y_test)

        self.y_train = self.y_train.reshape(-1, )
        self.y_train_scaled = self.y_train_scaled.reshape(-1, )
        self.y_test = self.y_test.reshape(-1, )
        self.y_test_scaled = self.y_test_scaled.reshape(-1, )


    def fit_grid_search(self, model, param_grid, scaled=False, model_name="Model"):
        print(f"=== Running {model_name} for {self.satellite} ===")
        
        y_train = self.y_train
        y_test = self.y_test

        if scaled:
            X_train_data = self.X_train_scaled
            X_test_data = self.X_test_scaled
            
        else:
            X_train_data = self.X_train
            X_test_data = self.X_test

        grid_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=40,
            cv=3,
            n_jobs=-1,
            verbose=0,
            random_state=42,
        )

        grid_search.fit(X_train_data, y_train)

        best_model = grid_search.best_estimator_
        test_score = best_model.score(X_test_data, y_test)
        print(f"{model_name} Test R2 Score - {test_score*100:.4f}")
        
        y_preds = best_model.predict(X_test_data)

        self.make_plot(y_test, y_preds, model_name)
        return self.make_result_dict(y_test, y_preds)
    
    def make_result_dict(self, y_true, y_preds):
        return {
            'MAE':  round(float(mean_absolute_error(y_true, y_preds)), 6),
            'MSE':  round(float(mean_squared_error(y_true, y_preds)), 6),
            'RMSE': round(float(root_mean_squared_error(y_true, y_preds)), 6),
            'R2':   round(float(r2_score(y_true, y_preds)), 6),
            'MAPE': round(float(mean_absolute_percentage_error(y_true, y_preds)), 6),
        }

    def make_plot(self, y_test, y_preds, model_name):
        plot_dir = self.results_path / "plots"
        os.makedirs(plot_dir, exist_ok=True)

        mae = mean_absolute_error(y_test, y_preds)
        mape = mean_absolute_percentage_error(y_test, y_preds) * 100
        
        fig, ax = plt.subplots(figsize=(14, 7))
        indices = range(len(y_test))

        ax.scatter(indices, y_test, label='Actual', alpha=0.7, color='blue', s=20)
        ax.scatter(indices, y_preds, label='Predicted', alpha=0.7, color='red', s=20)

        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Soil Moisture (%)')
        ax.set_title(f'{self.satellite}: Actual vs Predicted — {model_name}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        metrics_text = f'MAE: {mae:.4f}\nMAPE: {mape:.2f}%'
        ax.annotate(metrics_text, xy=(0.02, 0.97), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    verticalalignment='top', fontsize=11)

        plot_path = os.path.join(plot_dir, f"{self.satellite}_{model_name}_actual_vs_predicted.png")
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()

    def run_experiment(self):
        results = {}

        # Random Forest
        rf = RandomForestRegressor(random_state=10, n_jobs=-1)
        rf_param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
        }
        results["RandomForest"] = self.fit_grid_search(rf, rf_param_grid, model_name="RandomForest")

        # XGBoost (GPU via tree_method='hist' + device='cuda')
        xgb = XGBRegressor(random_state=10, objective='reg:squarederror',
                            device='cuda', tree_method='hist')
        xgb_param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2],
        }
        results["XGBoost"] = self.fit_grid_search(xgb, xgb_param_grid, model_name="XGBoost")

        # AdaBoost
        ada = AdaBoostRegressor(
            estimator=DecisionTreeRegressor(random_state=10),
            random_state=10
        )
        ada_param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
            'estimator__max_depth': [2, 3, 5],
            'estimator__min_samples_split': [2, 5, 10],
        }
        results["AdaBoost"] = self.fit_grid_search(ada, ada_param_grid, model_name="AdaBoost")

        # SVR (scaled; grid trimmed to practical kernels/ranges)
        svr = SVR()
        svr_param_grid = {
            'kernel': ['linear', 'rbf'],
            'C': [1, 10, 50, 100],
            'gamma': ['scale', 0.01, 0.1],
            'epsilon': [0.05, 0.1, 0.2],
        }
        if self.X_train_scaled is not None:
            results["SVR"] = self.fit_grid_search(svr, svr_param_grid, scaled=True, model_name="SVR")
        
        metrics_filename = self.results_path / f"metrics_{self.satellite}.json"
        try:
            with open(metrics_filename, 'w') as f:
                json.dump(results, f, indent=4)
            print("Metrics saved successfully.")
        except Exception as e:
            print(f"Error saving metrics as JSON: {e}")
        
        return results

    

class ANNExperiment(Experiment):
    def __init__(self, X, y, satellite, train_size=0.8, test_size=0.1, val_size=0.1, 
                 split_type='train-val-test', print_stats=None, type='censored'):
        super().__init__(X, y, train_size, test_size, val_size, split_type, print_stats)
        self.satellite = satellite
        self.results_path = OUTPUT_PATH / f"ann_experiments_{type}"
        os.makedirs(self.results_path, exist_ok=True)
        print(f"Results → {self.results_path}")

        self.__scale_data()

    def __scale_data(self):
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()

        self.X_train_scaled = self.x_scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.x_scaler.transform(self.X_test)
        self.X_val_scaled = self.x_scaler.transform(self.X_val)

        self.y_train_scaled = self.y_scaler.fit_transform(self.y_train)
        self.y_test_scaled = self.y_scaler.transform(self.y_test)
        self.y_val_scaled = self.y_scaler.transform(self.y_val)

        self.y_train = self.y_train.reshape(-1, )
        self.y_train_scaled = self.y_train_scaled.reshape(-1, )
        self.y_test = self.y_test.reshape(-1, )
        self.y_test_scaled = self.y_test_scaled.reshape(-1, )
        self.y_val = self.y_val.reshape(-1, )
        self.y_val_scaled = self.y_val_scaled.reshape(-1, )

    def train_model(self, model, optimizer='adam', epochs=200, batch_size=256, verbose=0):
        self.model = model
        
        # Compile the model
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        progress = EpochTqdm(total_epochs=epochs)

        # Train the model
        self.history = self.model.fit(
            self.X_train_scaled, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_val_scaled, self.y_val),
            verbose=verbose,
            callbacks=[early_stopping, reduce_lr, progress]
        )
        
        # Evaluate on test set
        test_loss, test_mae = self.model.evaluate(self.X_test_scaled, self.y_test, verbose=0)
        print(f"\nTest Loss (MSE): {test_loss:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test_scaled).flatten()
        y_pred_val = self.model.predict(self.X_val_scaled).flatten()

        # Calculate additional metrics
        results_test = self.evaluate_model(self.y_test, y_pred)
        results_val = self.evaluate_model(self.y_val, y_pred_val)

        print(f"\nAdditional Metrics:")
        print(f"MSE: {results_test['MSE']:.4f}")
        print(f"R² Score: {results_test['R2']:.4f}")
        
        return y_pred, results_test, results_val
        

    def evaluate_model(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        return {
            "MAE": float(mae),
            "MSE": float(mse),
            "R2": float(r2)
        }
    
    def plot_line_comparison(self, test_results, val_results, test_preds, model_params):
        test_mae = test_results['MAE']
        val_mae = val_results['MAE']
        test_mse = test_results['MSE']
        val_mse = val_results['MSE']
        
        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(14, 7))
        indices = range(len(self.y_test))

        ax.scatter(indices, self.y_test, label='Actual (Test)', color='blue', alpha=0.7, s=20)
        ax.scatter(indices, test_preds, label='Predicted (Test)', color='red', alpha=0.7, s=20)

        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Soil Moisture (%)')
        ax.set_title(f'{self.satellite}: Actual vs Predicted — {model_params}')

        metrics_text = (f'Test MAE: {test_mae:.4f}  |  Val MAE: {val_mae:.4f}\n'
                        f'Test MSE: {test_mse:.2f}   |  Val MSE: {val_mse:.2f}')
        ax.annotate(metrics_text, xy=(0.02, 0.97), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    verticalalignment='top', fontsize=11, fontfamily='monospace')

        plot_path = self.results_path / "plots"
        os.makedirs(plot_path, exist_ok=True)

        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path / f"{self.satellite}_{model_params}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_experiment(self, model, optimizer='adam', epochs=200, batch_size=256, verbose=0, model_param_string=None):
        # Train model
        y_pred, test_results, val_results = self.train_model(
            model, optimizer, epochs, batch_size, verbose
        )
        
        # Generate plots
        # self.plot_line_comparison(test_results, val_results, y_pred, model_param_string)
        self.plot_prediction_line(self.y_test, y_pred, model_param_string)

        results = {
            "Test": test_results,
            "Val": val_results
        }

        return results

    def plot_prediction_line(self, y_test, y_preds, model_name):
        plot_dir = self.results_path / "plots"
        os.makedirs(plot_dir, exist_ok=True)

        mae = mean_absolute_error(y_test, y_preds)
        mape = mean_absolute_percentage_error(y_test, y_preds) * 100
        r2 = r2_score(y_test, y_preds)

        # Use a square figure to make the identity line 45 degrees visually
        plt.figure(figsize=(10, 10))
        
        # 1. Scatter Plot (Actual vs Predicted)
        plt.scatter(y_test, y_preds, alpha=0.6, edgecolors='k', linewidth=0.5, color='steelblue', label='Prediction')
        
        # Determine axis limits to keep plot square and unified
        data_min = min(y_test.min(), y_preds.min())
        data_max = max(y_test.max(), y_preds.max())
        
        # Add a small buffer (5%) so points aren't on the edge
        buffer = (data_max - data_min) * 0.05
        p_min = data_min - buffer
        p_max = data_max + buffer

        # 2. Identity Line (y = x) -> Perfect Prediction
        plt.plot([p_min, p_max], [p_min, p_max], color='black', linestyle='--', lw=2, label='Identity')

        # 3. Best Fit Line (Linear Regression of the predictions)
        # This helps you see if the model is systematically over/under predicting
        try:
            coeffs = np.polyfit(y_test.flatten(), y_preds.flatten(), 1)
            poly_eqn = np.poly1d(coeffs)
            x_range = np.linspace(p_min, p_max, 100)
            plt.plot(x_range, poly_eqn(x_range), color='darkred', linestyle='-', lw=2, label='Best Fit')
        except Exception as e:
            print(f"Could not fit regression line for plot: {e}")
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{self.satellite}: Prediction Error — {model_name}')

        metrics_text = f'MAE: {mae:.4f}\nMAPE: {mape:.2f}%\nR²: {r2:.4f}'
        plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
                    verticalalignment='top', fontsize=11)

        plt.xlim(p_min, p_max)
        plt.ylim(p_min, p_max)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, f"{self.satellite}_{model_name}_prediction_error.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()



class PredictionIntervalEstimation(Experiment):
    def __init__(self, X, y, satellite, train_size=0.8, test_size=0.1, val_size=0.1, split_type='train-val-test', print_stats=None, type='uncensored'):
        super().__init__(X, y, train_size, test_size, val_size, split_type, print_stats)
        self.results_path = OUTPUT_PATH / f"pi_estimation_{type}"
        os.makedirs(self.results_path, exist_ok=True)
        print(f"Results → {self.results_path}")
        self.satellite = satellite

        self.__scale_data()

    def __scale_data(self):
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()

        self.X_train_scaled = self.x_scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.x_scaler.transform(self.X_test)
        self.X_val_scaled = self.x_scaler.transform(self.X_val)

        self.y_train_scaled = self.y_scaler.fit_transform(self.y_train)
        self.y_test_scaled = self.y_scaler.transform(self.y_test)
        self.y_val_scaled = self.y_scaler.transform(self.y_val)

        self.y_train = self.y_train.reshape(-1, )
        self.y_train_scaled = self.y_train_scaled.reshape(-1, )
        self.y_test = self.y_test.reshape(-1, )
        self.y_test_scaled = self.y_test_scaled.reshape(-1, )
        self.y_val = self.y_val.reshape(-1, )
        self.y_val_scaled = self.y_val_scaled.reshape(-1, )

    def pinball_loss(self, y_true, y_pred, tau):
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(tau * error, (tau - 1) * error))

    def lower_quantile_loss(self, y_true, y_pred, tau=0.025):
        return self.pinball_loss(y_true, y_pred, tau=tau)
    
    def upper_quantile_loss(self, y_true, y_pred, tau=0.975):
        return self.pinball_loss(y_true, y_pred, tau=tau)

    def train_model(self, model, learning_rate, optimizer='adam', epochs=200, batch_size=256, verbose=0, tau_lower=0.025, tau_upper=0.975):
        self.upper_model = tf.keras.models.clone_model(model)
        self.lower_model = tf.keras.models.clone_model(model)

        if isinstance(optimizer, str):
            optimizer_config = {'class_name': optimizer, 'config': {'learning_rate': learning_rate}} # Adam default
        else:
            optimizer_config = optimizer.get_config()
            optimizer_config['class_name'] = optimizer_config['name']
            del optimizer_config['name']

        # 2. Create two new, independent optimizer instances from that config
        upper_optimizer = tf.keras.optimizers.get(optimizer_config.copy())
        lower_optimizer = tf.keras.optimizers.get(optimizer_config.copy())

        loss_lower = lambda y, p: self.lower_quantile_loss(y, p, tau=tau_lower)
        loss_upper = lambda y, p: self.upper_quantile_loss(y, p, tau=tau_upper)

        # compile both models
        self.lower_model.compile(
            optimizer=lower_optimizer,
            loss=loss_lower
        )
        self.upper_model.compile(
            optimizer=upper_optimizer,
            loss=loss_upper
        )


        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        progress = EpochTqdm(total_epochs=epochs, desc="Upper model")
        self.upper_model_history = self.upper_model.fit(
            self.X_train_scaled, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_val_scaled, self.y_val),
            verbose=verbose,
            callbacks=[progress, early_stopping, reduce_lr]
        )
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        progress = EpochTqdm(total_epochs=epochs, desc="Lower model")
        self.lower_model_history = self.lower_model.fit(
            self.X_train_scaled, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_val_scaled, self.y_val),
            verbose=verbose,
            callbacks=[progress, early_stopping, reduce_lr]
        )

        # Predict on both test and validation sets
        y_preds_lower_test = self.lower_model.predict(self.X_test_scaled)
        y_preds_upper_test = self.upper_model.predict(self.X_test_scaled)
        y_preds_lower_val = self.lower_model.predict(self.X_val_scaled)
        y_preds_upper_val = self.upper_model.predict(self.X_val_scaled)

        return (y_preds_lower_test.flatten(), y_preds_upper_test.flatten(),
                y_preds_lower_val.flatten(), y_preds_upper_val.flatten())

    def plot_training_history(self, model_param_string):
        plt.figure(figsize=(12, 4))
        plt.suptitle(f"{self.satellite}: {model_param_string}\nUpper and Lower Model Training Loss")
        plt.subplot(1, 2, 1)
        plt.plot(self.upper_model_history.history['loss'], label='Training Loss')
        plt.plot(self.upper_model_history.history['val_loss'], label='Validation Loss')
        plt.title('Upper Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.lower_model_history.history['loss'], label='Training Loss')
        plt.plot(self.lower_model_history.history['val_loss'], label='Validation Loss')
        plt.title('Lower Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def evaluate_model(self, y_true, y_pred_lower, y_pred_upper):
        
        def picp(y_true_vals, y_pred_lower_vals, y_pred_upper_vals):
            """Prediction Interval Coverage Probability"""
            covered = np.sum((y_true_vals >= y_pred_lower_vals) & (y_true_vals <= y_pred_upper_vals))
            return covered / len(y_true_vals)

        def mpiw(y_pred_lower_vals, y_pred_upper_vals):
            """Mean Prediction Interval Width"""
            return np.mean(y_pred_upper_vals - y_pred_lower_vals)

        return {
            'PICP': round(float(picp(y_true, y_pred_lower, y_pred_upper)), 6),
            'MPIW': round(float(mpiw(y_pred_lower, y_pred_upper)), 6),
        }

    def plot_prediction_interval(self, y_pred_lower_test, y_pred_upper_test, y_pred_lower_val, y_pred_upper_val, model_param_string):
        indices = range(len(self.y_test))

        test_eval_dict = self.evaluate_model(self.y_test, y_pred_lower_test, y_pred_upper_test)
        val_eval_dict = self.evaluate_model(self.y_val, y_pred_lower_val, y_pred_upper_val)

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(indices, self.y_test, 'o', color='blue', markersize=4, label='Actual Soil Moisture (Test Set)')
        ax.plot(indices, y_pred_lower_test, color='red', linestyle='--', label='Lower Bound')
        ax.plot(indices, y_pred_upper_test, color='orange', linestyle='--', label='Upper Bound')
        ax.fill_between(indices, y_pred_lower_test, y_pred_upper_test, color='gray', alpha=0.2, label='95% Prediction Interval')

        test_metrics = f"Test  | PICP: {test_eval_dict['PICP']*100:5.2f}% | MPIW: {test_eval_dict['MPIW']:.4f}"
        val_metrics  = f"Valid | PICP: {val_eval_dict['PICP']*100:5.2f}% | MPIW: {val_eval_dict['MPIW']:.4f}"
        ax.annotate(
            f"{test_metrics}\n{val_metrics}",
            xy=(0.02, 0.97), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
            verticalalignment='top', fontsize=11, fontfamily='monospace'
        )

        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Soil Moisture (%)')
        ax.set_title(f'{self.satellite}: {model_param_string} — Prediction Interval')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plot_dir = self.results_path / "plots"
        os.makedirs(plot_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{self.satellite}_{model_param_string}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def run_experiment(self, model, optimizer='adam', epochs=200, learning_rate=0.01, batch_size=256, verbose=0, model_param_string=None):
        # Unpack all four returned prediction arrays
        y_preds_lower_test, y_preds_upper_test, y_preds_lower_val, y_preds_upper_val = self.train_model(
            model, optimizer=optimizer, epochs=epochs, batch_size=batch_size, verbose=verbose, learning_rate=learning_rate
        )
        
        # self.plot_training_history(model_param_string)
        
        # Pass all four arrays to the plotting function
        self.plot_prediction_interval(
            y_preds_lower_test, y_preds_upper_test,
            y_preds_lower_val, y_preds_upper_val,
            model_param_string
        )

        results_val = self.evaluate_model(self.y_val, y_preds_lower_val, y_preds_upper_val)
        results_test = self.evaluate_model(self.y_test, y_preds_lower_test, y_preds_upper_test)

        results = {
            "val": results_val,
            "test": results_test
        }
        print(f"{model_param_string}: {json.dumps(results, indent=4)}")

        return results


class ConformalRegression:
    def __init__(self, X, y, satellite, train_size=0.8, conf_size=0.1, test_size=0.1, print_splits=True, type='uncensored'):
        self.X = X
        self.y = y
        self.satellite = satellite
        self.train_size = train_size
        self.conf_size = conf_size
        self.test_size = test_size
        self.random_seed = 42
        self.results_path = OUTPUT_PATH / f"conformal_regression_{type}"
        os.makedirs(self.results_path, exist_ok=True)
        print(f"Results → {self.results_path}")
        self.__prepare_data(print_splits)
    
    def __prepare_data(self, print_splits):
        self.y = self.y.reshape(-1, )

        (
            X_train, X_cal, X_test,
            y_train, y_cal, y_test
        ) = train_conformalize_test_split(self.X, self.y, 
                                          train_size=self.train_size, conformalize_size=self.conf_size, 
                                          test_size=self.test_size, random_state=self.random_seed)

        self.X_train = X_train
        self.X_conf = X_cal
        self.X_test = X_test

        self.y_train = y_train.reshape(-1, )
        self.y_conf = y_cal.reshape(-1, )
        self.y_test = y_test.reshape(-1, )

        if print_splits:
            total = len(self.X)
            print(
                "Split sizes:\n"
                f"  Train    - X: {getattr(self.X_train, 'shape', (len(self.X_train),))}, y: {len(self.y_train)} ({len(self.y_train)/total:.1%})\n"
                f"  Conformal- X: {getattr(self.X_conf,  'shape', (len(self.X_conf), ))}, y: {len(self.y_conf)} ({len(self.y_conf)/total:.1%})\n"
                f"  Test     - X: {getattr(self.X_test,  'shape', (len(self.X_test), ))}, y: {len(self.y_test)} ({len(self.y_test)/total:.1%})\n"
                f"  Original data rows: {total}"
            )

    def run_experiment(self):
        models = {
            "QuantileRegressor": QuantileRegressor(),
            "GradientBoostingRegressor": GradientBoostingRegressor(loss="quantile"),
            "HistGradientBoostingRegressor": HistGradientBoostingRegressor(loss="quantile"),
            "LGBMRegressor": LGBMRegressor(device='gpu', objective='quantile', alpha=0.5, verbose=-1),
        }

        results = {}

        for model_string, model in models.items():
            print(f"==========Running {model_string}===========\n\n")
            regressor = ConformalizedQuantileRegressor(
                estimator=model,
                confidence_level=0.95,
                prefit=False
            )
            regressor.fit(self.X_train, self.y_train)
            regressor.conformalize(self.X_conf, self.y_conf)

            _, intervals = regressor.predict_interval(self.X_test, minimize_interval_width=True)

            intervals = intervals.squeeze()
            lower_preds = intervals[:, 0]
            upper_preds = intervals[:, 1]
            self.plot_prediction_interval(lower_preds, upper_preds, model_string)
            results[model_string] = self.evaluate_model(self.y_test, lower_preds, upper_preds)
        
        with open(self.results_path / f"{self.satellite}_metrics.json", "w") as f:
            json.dump(results, f, indent=4)
    
    def evaluate_model(self, y_true, y_pred_lower, y_pred_upper):
        
        def picp(y_true_vals, y_pred_lower_vals, y_pred_upper_vals):
            """Prediction Interval Coverage Probability"""
            covered = np.sum((y_true_vals >= y_pred_lower_vals) & (y_true_vals <= y_pred_upper_vals))
            return covered / len(y_true_vals)

        def mpiw(y_pred_lower_vals, y_pred_upper_vals):
            """Mean Prediction Interval Width"""
            return np.mean(y_pred_upper_vals - y_pred_lower_vals)

        return {
            'PICP': round(float(picp(y_true, y_pred_lower, y_pred_upper)), 6),
            'MPIW': round(float(mpiw(y_pred_lower, y_pred_upper)), 6),
        }

    def plot_prediction_interval(self, y_pred_lower_test, y_pred_upper_test, model_param_string):
        indices = range(len(self.y_test))

        test_eval_dict = self.evaluate_model(self.y_test, y_pred_lower_test, y_pred_upper_test)

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(indices, self.y_test, 'o', color='blue', markersize=4, label='Actual Soil Moisture (Test Set)')
        ax.plot(indices, y_pred_lower_test, color='red', linestyle='--', label='Lower Bound')
        ax.plot(indices, y_pred_upper_test, color='orange', linestyle='--', label='Upper Bound')
        ax.fill_between(indices, y_pred_lower_test, y_pred_upper_test, color='gray', alpha=0.2, label='95% Prediction Interval')

        test_metrics = f"Test  | PICP: {test_eval_dict['PICP']*100:5.2f}% | MPIW: {test_eval_dict['MPIW']:.4f}"
        ax.annotate(
            test_metrics,
            xy=(0.02, 0.97), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
            verticalalignment='top', fontsize=11, fontfamily='monospace'
        )

        plot_dir = self.results_path / "plots"
        os.makedirs(plot_dir, exist_ok=True)

        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Soil Moisture (%)')
        ax.set_title(f'{self.satellite}: {model_param_string} — Conformal Prediction Interval')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{self.satellite}_{model_param_string}.png", dpi=300, bbox_inches='tight')
        plt.close()



class ConformalizedQuantileExperiment(PredictionIntervalEstimation):
    def __init__(self, X, y, satellite, train_size=0.8, test_size=0.1, val_size=0.1, split_type='train-val-test', print_stats=None, type='uncensored'):
        # Reuse parent init for data splitting (Train=Fit, Val=Calibration, Test=Evaluate) and scaling
        super().__init__(X, y, satellite, train_size, test_size, val_size, split_type, print_stats, type)
        self.results_path = OUTPUT_PATH / f"conformal_results_{type}"
        print(f"Results → {self.results_path}")
        os.makedirs(self.results_path, exist_ok=True)

    def __apply_cqr_calibration(self, y_true_cal, y_lower_cal, y_upper_cal, y_lower_test, y_upper_test, alpha=0.05):
        """
        Applies CQR calibration.
        Computes score E_i = max(q_low - y, y - q_high) on calibration set.
        Adjusts test intervals by the (1-alpha) quantile of scores.
        """
        # 1. Calculate non-conformity scores on calibration set
        # We want y to be between q_low and q_high.
        # If y < q_low, error is positive (q_low - y)
        # If y > q_high, error is positive (y - q_high)
        # If inside, error is negative (max of two negatives)
        scores = np.maximum(y_lower_cal - y_true_cal, y_true_cal - y_upper_cal)

        # 2. Compute Q with finite-sample correction: (1-α)(1+1/n) quantile
        n = len(scores)
        q_level = min((1 - alpha) * (1 + 1 / n), 1.0)
        q_hat = np.quantile(scores, q_level, method='higher')
        
        print(f"  > CQR Calibration constant (Q): {q_hat:.4f}")
        
        # 3. Adjust Test predictions
        y_lower_test_cqr = y_lower_test - q_hat
        y_upper_test_cqr = y_upper_test + q_hat
        
        return y_lower_test_cqr, y_upper_test_cqr

    def __apply_split_conformal_calibration(self, y_true_cal, y_pred_cal, y_pred_test, alpha=0.05):
        """
        Applies standard Split Conformal Prediction (Absolute Residuals).
        Used for models that predict mean (SVM) instead of quantiles.
        """
        # 1. Calculate absolute residuals on calibration set
        scores = np.abs(y_true_cal - y_pred_cal)

        # 2. Compute Q with finite-sample correction
        n = len(scores)
        q_level = min((1 - alpha) * (1 + 1 / n), 1.0)
        q_hat = np.quantile(scores, q_level, method='higher')
        
        print(f"  > Split Conformal Calibration constant (Q): {q_hat:.4f}")
        
        # 3. Create intervals for Test
        y_lower_test = y_pred_test - q_hat
        y_upper_test = y_pred_test + q_hat
        
        return y_lower_test, y_upper_test

    def run_linear_cqr(self, alpha=0.05):
        print("\n--- Running Linear CQR (QuantileRegressor) ---")
        # Train Lower Quantile Model
        qr_low = QuantileRegressor(quantile=alpha/2, solver='highs')
        qr_low.fit(self.X_train_scaled, self.y_train)
        
        # Train Upper Quantile Model
        qr_high = QuantileRegressor(quantile=1 - alpha/2, solver='highs')
        qr_high.fit(self.X_train_scaled, self.y_train)
        
        # Predict on Calibration (Val)
        low_cal = qr_low.predict(self.X_val_scaled)
        high_cal = qr_high.predict(self.X_val_scaled)
        
        # Predict on Test
        low_test = qr_low.predict(self.X_test_scaled)
        high_test = qr_high.predict(self.X_test_scaled)
        
        # Apply CQR
        return self.__apply_cqr_calibration(
            self.y_val, low_cal, high_cal, low_test, high_test, alpha
        )

    def run_svm_conformal(self, alpha=0.05):
        print("\n--- Running SVM Split Conformal (SVR) ---")
        # SVM doesn't support Quantile loss natively efficiently.
        # We use standard Split Conformal: Predict Mean -> Calibrate Residuals.
        
        # 1. Train SVR (Mean predictor)
        # Note: SVR can be slow on unscaled Y. 
        # If y is large, consider scaling Y, but for consistency we use y_train (unscaled) here
        # assuming the user handles runtime or data isn't huge.
        svr = SVR(kernel='rbf') 
        svr.fit(self.X_train_scaled, self.y_train)
        
        # 2. Predict
        pred_cal = svr.predict(self.X_val_scaled)
        pred_test = svr.predict(self.X_test_scaled)
        
        # 3. Apply Split Conformal
        return self.__apply_split_conformal_calibration(
            self.y_val, pred_cal, pred_test, alpha
        )

    def run_ann_cqr(self, model_template, epochs=100, alpha=0.05):
        print("\n--- Running ANN CQR ---")
        # 1. Train Quantile ANN (using parent class logic)
        # Returns raw quantile predictions (uncalibrated)
        low_test, high_test, low_cal, high_cal = self.train_model(
            model_template,
            optimizer='adam',
            epochs=epochs,
            batch_size=256,
            verbose=0,
            learning_rate=0.001
        )
        
        # 2. Apply CQR
        return self.__apply_cqr_calibration(
            self.y_val, low_cal, high_cal, low_test, high_test, alpha
        )

    def run_experiment(self, ann_model_template, epochs=100, alpha=0.05):
        results = {}
        
        # 1. Run Models
        svm_low, svm_high = self.run_svm_conformal(alpha)
        lin_low, lin_high = self.run_linear_cqr(alpha)
        ann_low, ann_high = self.run_ann_cqr(ann_model_template, epochs, alpha)
        
        experiments = {
            "SVM_Split_Conformal": (svm_low, svm_high),
            "Linear_CQR": (lin_low, lin_high),
            "ANN_CQR": (ann_low, ann_high)
        }
        
        # 2. Evaluate and Plot
        for name, (y_low, y_high) in experiments.items():
            print(f"Evaluating {name}...")
            
            # Metrics
            metrics = self.evaluate_model(self.y_test, y_low, y_high)
            results[name] = metrics
            
            # Plot
            self.plot_prediction_interval(y_low, y_high, [], [], name) # Passing empty Val lists as we focus on Test result
            
        # 3. Save Results
        metrics_path = self.results_path / f"{self.satellite}_conformal_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=4)
            
        print(f"\nAll Conformal Experiments Completed. Results saved to {metrics_path}")
        return results
    
    # Override plot to be simpler since we iterate models
    def plot_prediction_interval(self, y_pred_lower_test, y_pred_upper_test, _, __, model_name):
        indices = range(len(self.y_test))
        metrics = self.evaluate_model(self.y_test, y_pred_lower_test, y_pred_upper_test)
        
        plt.figure(figsize=(14, 7))
        plt.plot(indices, self.y_test, 'o', color='blue', label='Actual', markersize=4, alpha=0.6)
        plt.plot(indices, y_pred_lower_test, color='red', linestyle='--', label='Lower Bound', linewidth=1)
        plt.plot(indices, y_pred_upper_test, color='orange', linestyle='--', label='Upper Bound', linewidth=1)
        plt.fill_between(indices, y_pred_lower_test, y_pred_upper_test, color='gray', alpha=0.2, label='95% Confidence')

        metrics_text = f"{model_name}\nPICP: {metrics['PICP']*100:.2f}%\nMPIW: {metrics['MPIW']:.4f}"
        plt.annotate(metrics_text, xy=(0.02, 0.98), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8), 
                    verticalalignment='top', fontsize=14, fontfamily='monospace')
        
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.title(f'{self.satellite} - {model_name} Prediction Intervals')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        plot_path = self.results_path / f"{self.satellite}_{model_name}_plot.png"
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
    

    def plot_tuning_comparison(self, results_df):
        """
        Generates an academic-style dual-axis plot comparing Raw vs CQR
        across different Tau configurations.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        x = np.arange(len(results_df))
        width = 0.35
        
        # --- Plot 1: PICP (Coverage) ---
        ax1.bar(x - width/2, results_df['Raw_PICP'] * 100, width, label='Raw QR', color='#4c72b0', alpha=0.8, edgecolor='black')
        ax1.bar(x + width/2, results_df['CQR_PICP'] * 100, width, label='CQR', color='#dd8452', alpha=0.8, edgecolor='black')
        
        # Target Line (95%)
        ax1.axhline(y=95, color='red', linestyle='--', linewidth=2, label='Target (95%)')
        
        ax1.set_ylabel('Coverage Probability (PICP) [%]', fontsize=14)
        ax1.set_title('Coverage Consistency', fontsize=16)
        ax1.set_xticks(x)
        ax1.set_xticklabels(results_df['Tau_Pair'], rotation=90, ha='right', fontsize=12)
        ax1.set_ylim(80, 100)  # Zoom in on high coverage area usually
        ax1.legend(loc='lower right', fontsize=12)
        ax1.grid(axis='y', linestyle='--', alpha=0.4)

        # --- Plot 2: MPIW (Width) ---
        ax2.bar(x - width/2, results_df['Raw_MPIW'], width, label='Raw QR', color='#4c72b0', alpha=0.8, edgecolor='black')
        ax2.bar(x + width/2, results_df['CQR_MPIW'], width, label='CQR', color='#dd8452', alpha=0.8, edgecolor='black')
        
        ax2.set_ylabel('Mean Prediction Interval Width (MPIW)', fontsize=14)
        ax2.set_title('Interval Efficiency (Lower is Better)', fontsize=16)
        ax2.set_xticks(x)
        ax2.set_xticklabels(results_df['Tau_Pair'], rotation=90, ha='right', fontsize=12)
        ax2.legend(loc='upper right', fontsize=12)
        ax2.grid(axis='y', linestyle='--', alpha=0.4)

        plt.suptitle(f"{self.satellite}: Hyperparameter Tuning (Lower/Upper Loss)", fontsize=18)
        plt.tight_layout()
        
        plot_path = self.results_path / f"{self.satellite}_tau_tuning_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    def run_ann_tuning_experiment(self, model_template, lower_taus=[0.01, 0.015, 0.02, 0.025, 0.03, 0.04], epochs=100):
        """
        Runs ANN QR and CQR for various (tau_lower, tau_upper) pairs 
        where the difference is fixed at 0.95.
        """
        tuning_results = []
        alpha = 0.05 # Fixed target alpha

        print(f"\n=== Starting Tau Hyperparameter Tuning for {self.satellite} ===")
        
        for lo in lower_taus:
            hi = round(lo + 0.95, 3) # Maintain 0.95 gap
            pair_name = f"Low:{lo} - High:{hi}"
            print(f"\nRunning configuration: {pair_name}")

            # 1. Train Base Models
            # We use the updated train_model which accepts tau arguments
            pred_lo_test, pred_hi_test, pred_lo_val, pred_hi_val = self.train_model(
                model_template, 
                optimizer='adam', 
                epochs=epochs, 
                learning_rate=0.001,
                verbose=0,
                tau_lower=lo,
                tau_upper=hi
            )

            # 2. Evaluate Raw QR
            raw_metrics = self.evaluate_model(self.y_test, pred_lo_test, pred_hi_test)

            # 3. Apply CQR Calibration
            # We use the Validation set predictions to calibrate
            cqr_lo_test, cqr_hi_test = self._ConformalizedQuantileExperiment__apply_cqr_calibration(
                self.y_val, pred_lo_val, pred_hi_val, pred_lo_test, pred_hi_test, alpha=alpha
            )

            # 4. Evaluate CQR
            cqr_metrics = self.evaluate_model(self.y_test, cqr_lo_test, cqr_hi_test)

            tuning_results.append({
                "Tau_Pair": pair_name,
                "Raw_PICP": raw_metrics['PICP'],
                "Raw_MPIW": raw_metrics['MPIW'],
                "CQR_PICP": cqr_metrics['PICP'],
                "CQR_MPIW": cqr_metrics['MPIW']
            })

        # Convert to DF for easier plotting
        df_results = pd.DataFrame(tuning_results)
        
        # Save JSON
        json_path = self.results_path / f"{self.satellite}_tau_tuning_metrics.json"
        df_results.to_json(json_path, orient='records', indent=4)
        print(f"Tuning metrics saved to {json_path}")

        # Plot
        self.plot_tuning_comparison(df_results)

        return df_results


class TubeLossPredictionInterval(Experiment):
    def __init__(self, X, y, satellite, train_size=0.8, test_size=0.1, val_size=0.1,
                 split_type='train-val-test', print_stats=None, type='uncensored',
                 q=0.95, r=0.5, delta=0, random_state=42):
        super().__init__(X, y, train_size, test_size, val_size, split_type, print_stats)
        self.satellite = satellite
        self.results_path = OUTPUT_PATH / f"tube_loss_{type}"
        os.makedirs(self.results_path, exist_ok=True)
        print(f"Results → {self.results_path}")

        self.q = q       # target coverage probability (default: 0.95)
        self.r = r       # controls PI tube movement (default: 0.5)
        self.delta = delta  # recalibration penalty (default: 0)

        np.random.seed(random_state)
        tf.random.set_seed(random_state)

        self.__scale_data()

    def __scale_data(self):
        self.x_scaler = MinMaxScaler()

        self.X_train_scaled = self.x_scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.x_scaler.transform(self.X_test)
        self.X_val_scaled = self.x_scaler.transform(self.X_val)

        self.y_train = self.y_train.reshape(-1,)
        self.y_test = self.y_test.reshape(-1,)
        self.y_val = self.y_val.reshape(-1,)

    def confidence_loss(self, y_true, y_pred):
        y_true = y_true[:, 0]
        f2 = y_pred[:, 0]
        f1 = y_pred[:, 1]

        c1 = (1 - self.q) * (f2 - y_true)
        c2 = (1 - self.q) * (y_true - f1)
        c3 = self.q * (f1 - y_true)
        c4 = self.q * (y_true - f2)

        loss_part1 = tf.where(y_true > self.r * (f1 + f2), c1, c2)
        loss_part2 = tf.where(f1 > y_true, c3, c4)

        final_loss = tf.where(
            tf.logical_and(y_true <= f2, y_true >= f1),
            loss_part1, loss_part2
        ) + (self.delta * tf.abs(f1 - f2))

        return tf.reduce_mean(final_loss)

    def train_model(self, model, optimizer, num_epochs=100, batch_size=256):
        self.model = tf.keras.models.clone_model(model)
        self.model.compile(optimizer=optimizer, loss=self.confidence_loss)

        progress = EpochTqdm(total_epochs=num_epochs)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

        self.history = self.model.fit(
            self.X_train_scaled, self.y_train,
            validation_data=(self.X_val_scaled, self.y_val),
            epochs=num_epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[progress, early_stopping, reduce_lr]
        )

    def plot_losses(self):
        plt.figure(figsize=(12, 4))
        plt.title('Model Loss')
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def evaluate_model(self, type='test'):
        if type == 'test':
            y_preds = self.model.predict(self.X_test_scaled, verbose=0)
            y_ref = np.stack((self.y_test, self.y_test), axis=1)
        else:
            y_preds = self.model.predict(self.X_val_scaled, verbose=0)
            y_ref = np.stack((self.y_val, self.y_val), axis=1)

        upper_preds = y_preds[:, 0]
        lower_preds = y_preds[:, 1]

        picp = round(float(np.mean((upper_preds > y_ref[:, 0]) * (lower_preds < y_ref[:, 1]))), 6)
        mpiw = round(float(np.mean(upper_preds - lower_preds)), 6)

        return upper_preds, lower_preds, {'PICP': picp, 'MPIW': mpiw}

    def plot_prediction_interval(self, model_param_string, save_fig=False):
        indices = range(len(self.y_test))

        y_pred_upper_test, y_pred_lower_test, test_results = self.evaluate_model()
        _, _, val_results = self.evaluate_model(type='val')

        plt.figure(figsize=(16, 7))
        plt.plot(indices, self.y_test, 'o', color='blue', label='Actual Soil Moisture (Test Set)')
        plt.plot(indices, y_pred_lower_test, color='red', linestyle='--', label='Lower Bound')
        plt.plot(indices, y_pred_upper_test, color='orange', linestyle='--', label='Upper Bound')
        plt.fill_between(indices, y_pred_lower_test, y_pred_upper_test, color='gray', alpha=0.2, label='95% Prediction Interval')

        test_metrics = f"Test  | PICP: {test_results['PICP']*100:5.2f}% | MPIW: {test_results['MPIW']:.4f}"
        val_metrics  = f"Valid | PICP: {val_results['PICP']*100:5.2f}% | MPIW: {val_results['MPIW']:.4f}"
        plt.annotate(f"{test_metrics}\n{val_metrics}", xy=(0.02, 0.98), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                     verticalalignment='top', fontsize=12, fontfamily='monospace')

        plt.xlabel('Sample Index')
        plt.ylabel('Soil Moisture (%)')
        plt.title(f'{self.satellite} — Tube Loss PI: {model_param_string} (r={self.r}, δ={self.delta})')
        plt.legend(fontsize=14)
        plt.grid(True, alpha=0.3)

        if save_fig:
            savepath = self.results_path / "plots"
            os.makedirs(savepath, exist_ok=True)
            plt.savefig(savepath / f"{self.satellite}_{model_param_string}.png", dpi=300, bbox_inches="tight")

        plt.show()

    def plot_prediction_interval_3d(self, feature1_name, feature2_name, model_param_string):
        import plotly.graph_objects as go
        from scipy.interpolate import griddata

        X_original_test = self.X_test
        y_true_test = self.y_test
        y_pred_upper_test, y_pred_lower_test, test_results = self.evaluate_model()
        _, _, val_results = self.evaluate_model(type='val')

        grid_res = 100
        x_grid = np.linspace(X_original_test[feature1_name].min(), X_original_test[feature1_name].max(), grid_res)
        y_grid = np.linspace(X_original_test[feature2_name].min(), X_original_test[feature2_name].max(), grid_res)
        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)

        Z_upper_mesh = griddata(
            (X_original_test[feature1_name], X_original_test[feature2_name]),
            y_pred_upper_test, (X_mesh, Y_mesh), method='cubic'
        )
        Z_lower_mesh = griddata(
            (X_original_test[feature1_name], X_original_test[feature2_name]),
            y_pred_lower_test, (X_mesh, Y_mesh), method='cubic'
        )

        fig = go.Figure()
        fig.add_trace(go.Surface(x=X_mesh, y=Y_mesh, z=Z_upper_mesh, opacity=0.5,
                                  colorscale=[[0, 'orange'], [1, 'orange']], showscale=False, name='Upper Bound'))
        fig.add_trace(go.Surface(x=X_mesh, y=Y_mesh, z=Z_lower_mesh, opacity=0.5,
                                  colorscale=[[0, 'red'], [1, 'red']], showscale=False, name='Lower Bound'))
        fig.add_trace(go.Scatter3d(x=X_original_test[feature1_name], y=X_original_test[feature2_name],
                                    z=y_true_test, mode='markers',
                                    marker=dict(size=3, color='blue'), name='Actual Soil Moisture'))

        test_m = f"Test  | PICP: {test_results['PICP']*100:5.2f}% | MPIW: {test_results['MPIW']:.4f}"
        val_m  = f"Valid | PICP: {val_results['PICP']*100:5.2f}% | MPIW: {val_results['MPIW']:.4f}"

        fig.update_layout(
            title=f'3D Prediction Interval for Soil Moisture<br>{self.satellite} - {model_param_string}',
            scene=dict(xaxis_title=f'Feature: {feature1_name}', yaxis_title=f'Feature: {feature2_name}',
                       zaxis_title='Soil Moisture', aspectratio=dict(x=1.5, y=1.5, z=1)),
            margin=dict(l=0, r=0, b=0, t=40),
            legend=dict(x=0.01, y=0.95),
            annotations=[dict(text=f"<b>Metrics</b><br>{test_m}<br>{val_m}",
                              x=0.01, y=0.98, xref='paper', yref='paper',
                              align='left', showarrow=False,
                              bgcolor='rgba(255, 255, 255, 0.8)', bordercolor='black', borderwidth=1,
                              font=dict(family='monospace', size=12))]
        )
        fig.show()

    def run_experiment(self, model, optimizer, epochs=200, batch_size=256,
                       model_param_string=None, plot_losses=False, return_preds=False,
                       save_fig=False, plot_interval=True):
        self.train_model(model=model, optimizer=optimizer, num_epochs=epochs, batch_size=batch_size)
        if plot_losses:
            self.plot_losses()
        if plot_interval:
            self.plot_prediction_interval(model_param_string=model_param_string, save_fig=save_fig)
        if return_preds:
            _, _, test_results = self.evaluate_model()
            _, _, val_results = self.evaluate_model('val')
            return {
                'test_results': test_results,
                'val_results': val_results
            }