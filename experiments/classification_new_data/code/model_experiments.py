import matplotlib
matplotlib.use('Agg')  # non-interactive backend — must be set before any pyplot import

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, root_mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMRegressor

from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import tensorflow as tf

from mapie.utils import train_conformalize_test_split
from mapie.regression import ConformalizedQuantileRegressor
from mapie.metrics.regression import regression_coverage_score, regression_mean_width_score

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from scipy.spatial.distance import cdist
from cvxopt import matrix, solvers

from constants import OUTPUT_PATH

from tqdm import tqdm

solvers.options['show_progress'] = False

# ── IEEE publication-quality global style (Times New Roman via Nimbus Roman) ──
matplotlib.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Nimbus Roman', 'Times New Roman', 'DejaVu Serif'],
    'font.weight':        'normal',
    'font.size':          9,
    'axes.titlesize':     10,
    'axes.titleweight':   'normal',
    'axes.labelsize':     9,
    'axes.labelweight':   'normal',
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'legend.fontsize':    8,
    'legend.framealpha':  0.85,
    'legend.edgecolor':   '0.7',
    'grid.alpha':         0.3,
    'grid.linestyle':     '--',
    'grid.linewidth':     0.5,
    'lines.linewidth':    1.2,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
})

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
            'rf': RandomForestClassifier(random_state=42),
            'xgb': XGBClassifier(random_state=42, eval_metric='mlogloss'),
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


class RegressionExperiment(Experiment):
    def __init__(self, X, y, satellite, train_size=0.8, test_size=0.1, val_size=0.1, 
                 split_type='train-val-test', print_stats=None, type='censored'):
        super().__init__(X, y, train_size, test_size, val_size, split_type, print_stats)
        self.satellite = satellite
        self.results_path = OUTPUT_PATH / f"ml_experiment_{type}"

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

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train_data, y_train)

        best_model = grid_search.best_estimator_
        test_score = best_model.score(X_test_data, y_test)
        print(f"{model_name} Test R2 Score - {test_score*100:.4f}")
        
        y_preds = best_model.predict(X_test_data)

        self.make_plot(y_test, y_preds, model_name)
        return self.make_result_dict(y_test, y_preds)
    
    def make_result_dict(self, y_true, y_preds):
        result_dict = {}

        result_dict['MAE'] = round(mean_absolute_error(y_true, y_preds), 4)
        result_dict['MSE'] = round(mean_squared_error(y_true, y_preds), 4)
        result_dict['RMSE'] = round(root_mean_squared_error(y_true, y_preds), 4)
        result_dict['R2'] =  round(r2_score(y_true, y_preds), 4)
        result_dict['MAPE'] = round(mean_absolute_percentage_error(y_true, y_preds), 4)

        return result_dict

    def make_plot(self, y_test, y_preds, model_name):
        plot_dir = self.results_path / "plots"
        os.makedirs(plot_dir, exist_ok=True)

        y_true_flat = np.asarray(y_test).flatten()
        y_pred_flat = np.asarray(y_preds).flatten()
        mae  = mean_absolute_error(y_true_flat, y_pred_flat)
        rmse = root_mean_squared_error(y_true_flat, y_pred_flat)
        r2   = r2_score(y_true_flat, y_pred_flat)

        indices = np.arange(len(y_true_flat))

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.scatter(indices, y_true_flat, s=14, alpha=0.7, color='#1f77b4',
                   edgecolors='none', label='Actual', zorder=3)
        ax.scatter(indices, y_pred_flat, s=14, alpha=0.7, color='#d62728',
                   edgecolors='none', label='Predicted', zorder=3)

        metrics_text = f'MAE = {mae:.4f}\nRMSE = {rmse:.4f}\nR² = {r2:.4f}'
        ax.annotate(metrics_text, xy=(0.02, 0.98), xycoords='axes fraction',
                    ha='left', va='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='0.7', alpha=0.9))

        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Soil Moisture (%)')
        ax.set_title(f'{self.satellite}: Actual vs. Predicted Values — {model_name}')
        ax.legend(loc='lower right')
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{self.satellite}_{model_name}_actual_vs_predicted.png")
        plt.close()

    def run_experiment(self):
        results = {}

        # Random Forest
        rf = RandomForestRegressor(random_state=10)
        rf_param_grid = {
            'n_estimators': [100, 200, 500],     
            'max_depth': [None, 5, 10, 20],      
            'min_samples_split': [2, 5, 10],     
            'min_samples_leaf': [1, 2, 4],       
            'max_features': ['sqrt', 'log2']     
        }
        results["RandomForest"] = self.fit_grid_search(rf, rf_param_grid, model_name="RandomForest")

        # XGBoost
        xgb = XGBRegressor(random_state=10, objective='reg:squarederror')
        xgb_param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        results["XGBoost"] = self.fit_grid_search(xgb, xgb_param_grid, model_name="XGBoost")

        # AdaBoost
        ada = AdaBoostRegressor(
            estimator=DecisionTreeRegressor(random_state=10),
            random_state=10
        )
        ada_param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 1.0],
            'estimator__max_depth': [2, 3, 5, None],
            'estimator__min_samples_split': [2, 5, 10]
        }
        results["AdaBoost"] = self.fit_grid_search(ada, ada_param_grid, model_name="AdaBoost")

        # SVR (requires scaled data!)
        svr = SVR()
        svr_param_grid = {
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'epsilon': [0.01, 0.1, 0.2, 0.5]
        }
        if self.X_train_scaled is not None:  # only run if scaled data provided
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

    def train_model(self, model, optimizer='adam', epochs=200, batch_size=32, verbose=0):
        self.model = model
        
        # Compile the model
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        # Train the model
        self.history = self.model.fit(
            self.X_train_scaled, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_val_scaled, self.y_val),
            verbose=verbose
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
        plt.figure(figsize=(14, 7))
        
        # Create index for x-axis (based on the test set)
        indices = range(len(self.y_test))
        
        # Plot both lines (showing test set comparison)
        plt.scatter(indices, self.y_test, label='Actual (Test)', color='blue', alpha=0.7)
        plt.scatter(indices, test_preds, label='Predicted (Test)', color='red', alpha=0.7)
        
        plt.xlabel('Sample Index', fontsize=16)
        plt.ylabel('Values', fontsize=16)
        plt.title(f'{self.satellite}: Actual vs Predicted Values (Test Set).\n Params: {model_params}', fontsize=16)

        # Updated metrics text to show both Test and Val
        metrics_text = (f'Test MAE: {test_mae:.4f}  |  Val MAE:  {val_mae:.4f}\n'
                        f'Test MSE: {test_mse:.2f}  |  Val MSE: {val_mse:.2f}')
        
        plt.annotate(metrics_text, xy=(0.02, 0.98), xycoords='axes fraction', 
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), 
                     verticalalignment='top', fontsize=16)
        
        plot_path = self.results_path / "plots"
        os.makedirs(plot_path, exist_ok=True)

        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plot_path / f"{self.satellite}_{model_params}.png", dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()
    
    def run_experiment(self, model, optimizer='adam', epochs=200, batch_size=32, verbose=0, model_param_string=None):
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

        y_t = np.asarray(y_test).flatten()
        y_p = np.asarray(y_preds).flatten()

        mae  = mean_absolute_error(y_t, y_p)
        mse  = mean_squared_error(y_t, y_p)
        r2   = r2_score(y_t, y_p)
        bias = float(np.mean(y_p - y_t))

        indices = np.arange(len(y_t))

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.scatter(indices, y_t, s=14, alpha=0.7, color='#1f77b4',
                   edgecolors='none', label='Actual (Test)', zorder=3)
        ax.scatter(indices, y_p, s=14, alpha=0.7, color='#d62728',
                   edgecolors='none', label='Predicted (Test)', zorder=3)

        metrics_text = (f'Test MAE={mae:.4f}  MSE={mse:.2f}\n'
                        f'R²={r2:.4f}  Bias={bias:+.4f}')
        ax.annotate(metrics_text, xy=(0.02, 0.98), xycoords='axes fraction',
                    ha='left', va='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='0.7', alpha=0.9))

        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Soil Moisture (%)')
        ax.set_title(f'{self.satellite}: Actual vs. Predicted Values (Test Set)\nParams: {model_name}')
        ax.legend(loc='lower right')
        ax.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, f"{self.satellite}_{model_name}_prediction_error.png")
        plt.savefig(plot_path)
        plt.close()



class PredictionIntervalEstimation(Experiment):
    def __init__(self, X, y, satellite, train_size=0.8, test_size=0.1, val_size=0.1, split_type='train-val-test', print_stats=None, type='censored'):
        super().__init__(X, y, train_size, test_size, val_size, split_type, print_stats)
        self.results_path = OUTPUT_PATH / f"pi_estimation_{type}"
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

    def train_model(self, model, learning_rate, optimizer='adam', epochs=200, batch_size=32, verbose=0, tau_lower=0.025, tau_upper=0.975):
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


        # print("--------- TRAINING UPPER MODEL -----------\n")
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        progress = EpochTqdm(total_epochs=epochs)
        self.upper_model_history = self.upper_model.fit(
            self.X_train_scaled, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_val_scaled, self.y_val),
            verbose=verbose,
            callbacks=[progress, early_stopping]
        )
        # print("--------- TRAINING LOWER MODEL -----------\n")
        progress = EpochTqdm(total_epochs=epochs)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.lower_model_history = self.lower_model.fit(
            self.X_train_scaled, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_val_scaled, self.y_val),
            verbose=verbose,
            callbacks=[progress, early_stopping]
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
        plt.close()

    def evaluate_model(self, y_true, y_pred_lower, y_pred_upper):

        def picp(y_true_vals, y_pred_lower_vals, y_pred_upper_vals):
            """Prediction Interval Coverage Probability"""
            covered = np.sum((y_true_vals >= y_pred_lower_vals) & (y_true_vals <= y_pred_upper_vals))
            return covered / len(y_true_vals)

        def mpiw(y_pred_lower_vals, y_pred_upper_vals):
            """Mean Prediction Interval Width"""
            return np.mean(y_pred_upper_vals - y_pred_lower_vals)

        return {
            'PICP': float(picp(y_true, y_pred_lower, y_pred_upper)),
            'MPIW': float(mpiw(y_pred_lower, y_pred_upper))
        }

    def plot_prediction_interval(self, y_pred_lower_test, y_pred_upper_test, y_pred_lower_val, y_pred_upper_val, model_param_string):
        test_m = self.evaluate_model(self.y_test, y_pred_lower_test, y_pred_upper_test)
        val_m  = self.evaluate_model(self.y_val,  y_pred_lower_val,  y_pred_upper_val)

        idx    = np.arange(len(self.y_test))
        y_true = self.y_test.flatten()
        y_lo   = np.asarray(y_pred_lower_test).flatten()
        y_hi   = np.asarray(y_pred_upper_test).flatten()

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.fill_between(idx, y_lo, y_hi, color='gray', alpha=0.2, label='95% Prediction Interval')
        ax.plot(idx, y_lo, 'r--', lw=1.0, label='Lower Bound')
        ax.plot(idx, y_hi, color='orange', linestyle='--', lw=1.0, label='Upper Bound')
        ax.scatter(idx, y_true, s=12, color='#1f77b4', alpha=0.8,
                   edgecolors='none', label='Actual Soil Moisture (Test Set)', zorder=4)

        txt = (f"Test  | PICP: {test_m['PICP']*100:5.2f}% | MPIW: {test_m['MPIW']:.4f}\n"
               f"Valid | PICP: {val_m['PICP']*100:5.2f}% | MPIW: {val_m['MPIW']:.4f}")
        ax.annotate(txt, xy=(0.02, 0.98), xycoords='axes fraction',
                    ha='left', va='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='0.7', alpha=0.9))

        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Soil Moisture (%)')
        ax.set_title(f'{self.satellite}: {model_param_string}\nPrediction Interval for Soil Moisture')
        ax.legend(loc='upper right', ncol=2)
        ax.grid(True)
        plot_dir = self.results_path / "plots"
        os.makedirs(plot_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{self.satellite}_{model_param_string}.png")
        plt.close()

    def run_experiment(self, model, optimizer='adam', epochs=200, learning_rate=0.01, batch_size=32, verbose=0, model_param_string=None):
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
        # with open(self.results_path / f"{self.satellite}_metrics.json", "w") as f:
        #     json.dump(results, f, indent=4)

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
            # "LGBMRegressor": LGBMRegressor(objective="quantile")
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
            'PICP': float(picp(y_true, y_pred_lower, y_pred_upper)),
            'MPIW': float(mpiw(y_pred_lower, y_pred_upper))
        }

    def plot_prediction_interval(self, y_pred_lower_test, y_pred_upper_test, model_param_string):
        metrics = self.evaluate_model(self.y_test, y_pred_lower_test, y_pred_upper_test)

        idx   = np.arange(len(self.y_test))
        y_lo  = np.asarray(y_pred_lower_test).flatten()
        y_hi  = np.asarray(y_pred_upper_test).flatten()

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.fill_between(idx, y_lo, y_hi, color='gray', alpha=0.2, label='95% Prediction Interval')
        ax.plot(idx, y_lo, 'r--', lw=1.0, label='Lower Bound')
        ax.plot(idx, y_hi, color='orange', linestyle='--', lw=1.0, label='Upper Bound')
        ax.scatter(idx, self.y_test, s=12, color='#1f77b4', alpha=0.8,
                   edgecolors='none', label='Actual Soil Moisture (Test Set)', zorder=4)

        metrics_text = f"PICP: {metrics['PICP']*100:.2f}%\nMPIW: {metrics['MPIW']:.4f}"
        ax.annotate(metrics_text, xy=(0.02, 0.98), xycoords='axes fraction',
                    ha='left', va='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='0.7', alpha=0.9))

        plot_dir = self.results_path / "plots"
        os.makedirs(plot_dir, exist_ok=True)

        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Soil Moisture (%)')
        ax.set_title(f'{self.satellite}: {model_param_string}\n95% Conformal Prediction Interval')
        ax.legend(loc='upper right', ncol=2)
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{self.satellite}_{model_param_string}.png")
        plt.close()



class ConformalizedQuantileExperiment(PredictionIntervalEstimation):
    def __init__(self, X, y, satellite, train_size=0.8, test_size=0.1, val_size=0.1, split_type='train-val-test', print_stats=None):
        # Reuse parent init for data splitting (Train=Fit, Val=Calibration, Test=Evaluate) and scaling
        super().__init__(X, y, satellite, train_size, test_size, val_size, split_type, print_stats)
        self.results_path = OUTPUT_PATH / "conformal_results"
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
        
        # 2. Compute Q (1-alpha quantile)
        # mapie logic usually uses (1-alpha)*(1 + 1/n) for finite sample correction, 
        # but standard np.quantile is acceptable for large n.
        q_hat = np.quantile(scores, 1 - alpha, method='higher')
        
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
        
        # 2. Compute Q
        q_hat = np.quantile(scores, 1 - alpha, method='higher')
        
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
            batch_size=32, 
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
        metrics = self.evaluate_model(self.y_test, y_pred_lower_test, y_pred_upper_test)

        idx  = np.arange(len(self.y_test))
        y_lo = np.asarray(y_pred_lower_test).flatten()
        y_hi = np.asarray(y_pred_upper_test).flatten()

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.fill_between(idx, y_lo, y_hi, color='gray', alpha=0.2, label='95% Confidence')
        ax.plot(idx, y_lo, 'r--', lw=1.0, label='Lower Bound')
        ax.plot(idx, y_hi, color='orange', linestyle='--', lw=1.0, label='Upper Bound')
        ax.scatter(idx, self.y_test, s=12, color='#1f77b4', alpha=0.8,
                   edgecolors='none', label='Actual', zorder=4)

        metrics_text = f"{model_name}\nPICP: {metrics['PICP']*100:.2f}%\nMPIW: {metrics['MPIW']:.4f}"
        ax.annotate(metrics_text, xy=(0.02, 0.98), xycoords='axes fraction',
                    ha='left', va='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='0.7', alpha=0.9))

        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Soil Moisture (%)')
        ax.set_title(f'{self.satellite} - {model_name} Prediction Intervals')
        ax.legend(loc='upper right', ncol=2)
        ax.grid(True)
        plt.tight_layout()
        plot_path = self.results_path / f"{self.satellite}_{model_name}_plot.png"
        plt.savefig(plot_path)
        plt.close()
    

    def plot_tuning_comparison(self, results_df):
        """
        Generates an academic-style dual-axis plot comparing Raw vs CQR
        across different Tau configurations.
        """
        x_labels = list(results_df['Tau_Pair'])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        x     = np.arange(len(results_df))
        width = 0.35

        # --- Plot 1: PICP (Coverage) ---
        ax1.bar(x - width/2, results_df['Raw_PICP'] * 100, width, label='Raw QR',
                color='#4c72b0', alpha=0.85, edgecolor='black', linewidth=0.5)
        ax1.bar(x + width/2, results_df['CQR_PICP'] * 100, width, label='CQR',
                color='#dd8452', alpha=0.85, edgecolor='black', linewidth=0.5)
        ax1.axhline(y=95, color='red', linestyle='--', linewidth=1.5, label='Target (95%)')
        ax1.set_ylabel('Coverage Probability (PICP) [%]')
        ax1.set_title('Coverage Consistency')
        ax1.set_xticks(x)
        ax1.set_xticklabels(x_labels, rotation=45, ha='right')
        ax1.set_ylim(80, 100)
        ax1.legend(loc='lower right')
        ax1.grid(axis='y')

        # --- Plot 2: MPIW (Width) ---
        ax2.bar(x - width/2, results_df['Raw_MPIW'], width, label='Raw QR',
                color='#4c72b0', alpha=0.85, edgecolor='black', linewidth=0.5)
        ax2.bar(x + width/2, results_df['CQR_MPIW'], width, label='CQR',
                color='#dd8452', alpha=0.85, edgecolor='black', linewidth=0.5)
        ax2.set_ylabel('Mean Prediction Interval Width (MPIW)')
        ax2.set_title('Interval Efficiency (Lower is Better)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(x_labels, rotation=45, ha='right')
        ax2.legend(loc='upper right')
        ax2.grid(axis='y')

        fig.suptitle(f"{self.satellite}: Hyperparameter Tuning (Lower/Upper Loss)")
        plt.tight_layout()
        plot_path = self.results_path / f"{self.satellite}_tau_tuning_comparison.png"
        plt.savefig(plot_path)
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


class QuantileSVRExperiment(Experiment):
    def __init__(self, X, y, satellite, train_size=0.8, test_size=0.1, val_size=0.1,
                 split_type='train-val-test', print_stats=None, type='censored'):
        super().__init__(X, y, train_size, test_size, val_size, split_type, print_stats)
        self.satellite = satellite
        self.results_path = OUTPUT_PATH / f"qsvr_pi_estimation_{type}"
        os.makedirs(self.results_path, exist_ok=True)
        self.__scale_data()

    def __scale_data(self):
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()

        self.X_train_scaled = self.x_scaler.fit_transform(self.X_train)
        self.X_val_scaled   = self.x_scaler.transform(self.X_val)
        self.X_test_scaled  = self.x_scaler.transform(self.X_test)

        self.y_train = self.y_train.reshape(-1, 1)
        self.y_val   = self.y_val.reshape(-1, 1)
        self.y_test  = self.y_test.reshape(-1, 1)

        self.y_train_scaled = self.y_scaler.fit_transform(self.y_train)
        self.y_val_scaled   = self.y_scaler.transform(self.y_val)
        self.y_test_scaled  = self.y_scaler.transform(self.y_test)

        self.y_train = self.y_train.ravel()
        self.y_val   = self.y_val.ravel()
        self.y_test  = self.y_test.ravel()
        self.y_train_scaled = self.y_train_scaled.ravel()
        self.y_val_scaled   = self.y_val_scaled.ravel()
        self.y_test_scaled  = self.y_test_scaled.ravel()

    @staticmethod
    def _kernel(X, gamma, Y=None):
        if Y is None:
            Y = X
        return np.exp(-gamma * cdist(X, Y, 'sqeuclidean'))

    @staticmethod
    def _fit(X, Y, gamma, C, tau, eps1=0.0):
        n = X.shape[0]
        H = QuantileSVRExperiment._kernel(X, gamma)
        Hb = np.block([[H, -H], [-H, H]])
        Y_col = Y.reshape(-1, 1)
        c_vec = np.vstack([(1 - tau) * eps1 * np.ones((n, 1)) - Y_col,
                            tau      * eps1 * np.ones((n, 1)) + Y_col]).flatten()
        vub = np.concatenate([tau * C * np.ones(n), (1 - tau) * C * np.ones(n)])
        I   = np.eye(2 * n)
        sol = solvers.qp(matrix(Hb), matrix(c_vec),
                         matrix(np.vstack([-I, I])),
                         matrix(np.hstack([np.zeros(2 * n), vub])))
        alpha = np.array(sol['x']).flatten()
        beta  = alpha[:n] - alpha[n:]
        return beta

    @staticmethod
    def _predict(X_train, X_pred, gamma, beta):
        return QuantileSVRExperiment._kernel(X_pred, gamma, X_train).dot(beta)

    def evaluate_model(self, y_true, y_pred_lower, y_pred_upper):
        y_true  = y_true.flatten()
        y_lower = y_pred_lower.flatten()
        y_upper = y_pred_upper.flatten()
        picp = float(np.mean((y_true >= y_lower) & (y_true <= y_upper)))
        mpiw = float(np.mean(y_upper - y_lower))
        return {'PICP': picp, 'MPIW': mpiw}

    def plot_prediction_interval(self, lo_test, hi_test, lo_val, hi_val, label):
        test_m = self.evaluate_model(self.y_test, lo_test, hi_test)
        val_m  = self.evaluate_model(self.y_val,  lo_val,  hi_val)

        idx  = np.arange(len(self.y_test))
        y_lo = np.asarray(lo_test).flatten()
        y_hi = np.asarray(hi_test).flatten()

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.fill_between(idx, y_lo, y_hi, color='gray', alpha=0.2, label='95% PI')
        ax.plot(idx, y_lo, 'r--', lw=1.0, label='Lower Bound')
        ax.plot(idx, y_hi, color='orange', linestyle='--', lw=1.0, label='Upper Bound')
        ax.scatter(idx, self.y_test, s=12, color='#1f77b4', alpha=0.8,
                   edgecolors='none', label='Actual', zorder=4)

        txt = (f"Test  | PICP: {test_m['PICP']*100:5.2f}% | MPIW: {test_m['MPIW']:.4f}\n"
               f"Valid | PICP: {val_m['PICP']*100:5.2f}% | MPIW: {val_m['MPIW']:.4f}")
        ax.annotate(txt, xy=(0.02, 0.98), xycoords='axes fraction',
                    ha='left', va='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='0.7', alpha=0.9))

        plot_dir = self.results_path / 'plots'
        os.makedirs(plot_dir, exist_ok=True)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Soil Moisture (%)')
        ax.set_title(f'{self.satellite}: {label}\nQ-SVR Prediction Interval')
        ax.legend(loc='upper right', ncol=2)
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{self.satellite}_{label}.png")
        plt.close()

    def run_experiment(self, C, gamma, q_lower=0.025, q_upper=0.975):
        label = f"C={C}_gamma={gamma}"
        print(f"\n--- Q-SVR {self.satellite} | {label} ---")

        t0 = time.time()
        beta_lo = self._fit(self.X_train_scaled, self.y_train, gamma, C, q_lower)
        beta_hi = self._fit(self.X_train_scaled, self.y_train, gamma, C, q_upper)
        print(f"Training done in {time.time()-t0:.1f}s")

        lo_val  = self._predict(self.X_train_scaled, self.X_val_scaled,  gamma, beta_lo)
        hi_val  = self._predict(self.X_train_scaled, self.X_val_scaled,  gamma, beta_hi)
        lo_test = self._predict(self.X_train_scaled, self.X_test_scaled, gamma, beta_lo)
        hi_test = self._predict(self.X_train_scaled, self.X_test_scaled, gamma, beta_hi)

        self.plot_prediction_interval(lo_test, hi_test, lo_val, hi_val, label)

        results = {
            "params": {"C": C, "gamma": gamma, "q_lower": q_lower, "q_upper": q_upper},
            "val":    self.evaluate_model(self.y_val,  lo_val,  hi_val),
            "test":   self.evaluate_model(self.y_test, lo_test, hi_test),
        }
        print(json.dumps({k: v for k, v in results.items() if k != 'params'}, indent=2))
        return results

        return df_results