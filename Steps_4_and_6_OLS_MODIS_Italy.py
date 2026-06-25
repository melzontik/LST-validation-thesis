#!/usr/bin/env python
# coding: utf-8

# In[40]:


import os
from glob import glob
import re
import csv
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, MonthLocator
import scipy
from scipy import stats

import statsmodels.api as sm
from statsmodels.stats.weightstats import DescrStatsW
from statsmodels.stats.stattools import jarque_bera
from patsy import dmatrices
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score





# ============================================================================
# PATH CONFIGURATION
# ============================================================================

relative_path = os.path.join("Downloads", "python_validation", "data", 
                             "VALIDATION_RESULTS_ALG_V1_2023", "Step_4",
                            "per_pixel_data", "modis_v1")

#relative_path = os.path.join("Downloads", "python_validation", "data", 
                            # "Validation_Results_SUMMER_2024_alg_version_1", "Step_4",
                           # "per_pixel_data", "modis_v1")

#relative_path = os.path.join("Downloads", "python_validation", "data", 
                          #   "VALIDATION_RESULTS_ITALY_2025", "STEP_4",
                           #  "modis_v1")

home_directory = os.path.expanduser("~")
path = os.path.join(home_directory, relative_path)

if os.path.exists(path):
    print("Directory exists:", path)
else:
    print("Directory does not exist:", path)

output_path = os.path.join(path, 'corr_clip_Modis_Milan_RLM_alpha_1')
os.makedirs(output_path, exist_ok=True)





# ============================================================================
# CONFIGURATION
# ============================================================================

date_str = "2023-09-01"
date_plot = "September 2023" #January, February, March, April, May, June, July, August, September, October, November, December
name_station = "Milan"
output_prefix = os.path.join(output_path, f"{date_str}_")
alpha = 0.01  # Significance level (0.01 0.05 0.10)
threshold_hub = 0.52 # 0.52=99% per a=0.01,
                     # 0.69=95% per a=0.05, 
                     # 0.81=90% per a=0.10




# ============================================================================
# DATA READING FUNCTION
# ============================================================================

def reading_file(filename):
    df = pd.read_csv(filename, na_values=["null", "undefined"])
    
    # Rinomina le colonne
    df = df.rename(columns={
        "col1": "image_id", 
        "col2": "image_date", 
        "col3": "feature_id", 
        "col4": "feature_zone", 
        "col991": "pix_num", 
        "col992": "mean",     
        "col993": "median", 
        "col994": "mode", 
        "col995": "std", 
        "col996": "max", 
        "col997": "min"
    }, errors="raise")
    
    return df





# ============================================================================
# READ DATA
# ============================================================================

df_super = reading_file(os.path.join(path, "super_resolved", "Milan_clip", 
                                     f'super_resolved_{date_str}.csv')) 
df_orig = reading_file(os.path.join(path, "original", "Milan_clip",        
                                    f'original_{date_str}.csv'))
date = df_super['image_date'].iloc[0]

print(f"\n{'='*60}")
print(f"ANALYSIS FOR: {name_station} - {date}")
print(f"{'='*60}\n")





# ============================================================================
# DATA PREPROCESSING - Keep as numpy arrays throughout
# ============================================================================

x_all = df_orig['mean'].to_numpy()
y_all = df_super['mean'].to_numpy()

valid_mask = ~np.isnan(x_all) & ~np.isnan(y_all)
x_all, y_all = x_all[valid_mask], y_all[valid_mask]

print(f"Total valid observations: {len(x_all)}")
print(f"Original LST range: [{x_all.min():.2f}, {x_all.max():.2f}]")
print(f"Downscaled LST range: [{y_all.min():.2f}, {y_all.max():.2f}]")

# Create design matrix with intercept (using numpy arrays)
X_all = sm.add_constant(x_all)





# ============================================================================
# STATISTICAL METRICS FUNCTION
# ============================================================================

def compute_metrics(y_true, y_pred, model_name="Model"):
    """Compute comprehensive evaluation metrics"""
    
    # Basic metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Bias
    bias = np.mean(y_pred - y_true)
    
    # Std
    std_error = np.sqrt(rmse**2 - bias**2)
    
    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    
    # Spearman correlation (rank-based, robust)
    spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)
    
    # Normalized RMSE
    nrmse = rmse / (y_true.max() - y_true.min())
    
    # Mean Absolute Percentage Error (careful with zeros)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    # Residuals
    residuals = y_true - y_pred
    
    # Residual statistics
    std_residuals = np.std(residuals, ddof=1)  # Standard deviation of residuals
    mad_residuals = np.median(np.abs(residuals - np.median(residuals)))  # MAD
    nmad_residuals = 1.4826 * mad_residuals  # NMAD (normalized MAD)
    
    metrics = {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'Bias': bias,
        'Std_Error': std_error,
        'Pearson_r': pearson_r,
        'Pearson_p': pearson_p,
        'Spearman_r': spearman_r,
        'Spearman_p': spearman_p,
        'NRMSE': nrmse,
        'MAPE': mape,
        'Residuals': residuals,
        'Std_Residuals': std_residuals,
        'MAD_Residuals': mad_residuals,
        'NMAD_Residuals': nmad_residuals
    }
        
    return metrics






# ============================================================================
# NORMALITY TESTS FUNCTION
# ============================================================================

def perform_normality_tests(residuals, alpha=0.05, model_name="Model"):
    """Perform normality tests and visualizations"""
    
    print("=" * 60)
    print(f"NORMALITY TESTS FOR {model_name} RESIDUALS")
    print(f'alpha = {alpha}')
    print("=" * 60)
    
    # 1. Jarque-Bera Test
    jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(residuals)
    print(f"\n1. Jarque-Bera Test:")
    print(f"   Statistic: {jb_stat:.4f}, p-value: {jb_pvalue:.8f}")
    print(f"   Skewness: {skew:.4f}, Kurtosis: {kurtosis:.4f}")
    print(f"   → {'PASS: Normal' if jb_pvalue > alpha else 'FAIL: Not normal'}")
   
    
    # 2. Shapiro-Wilk Test
    shapiro_stat, shapiro_pvalue = stats.shapiro(residuals)
    print(f"\n2. Shapiro-Wilk Test:")
    print(f"   Statistic: {shapiro_stat:.4f}, p-value: {shapiro_pvalue:.8f}")
    print(f"   → {'PASS: Normal' if shapiro_pvalue > alpha else 'FAIL: Not normal'}")
    
    # 3. Kolmogorov-Smirnov Test
    ks_stat, ks_pvalue = stats.kstest(residuals, 'norm', args=(residuals.mean(), residuals.std()))
    print(f"\n3. Kolmogorov-Smirnov Test:")
    print(f"   Statistic: {ks_stat:.4f}, p-value: {ks_pvalue:.8f}")
    print(f"   → {'PASS: Normal' if ks_pvalue > alpha else 'FAIL: Not normal'}")
   
    # ==========================
    # Histogram (export separately)
    # ==========================
    fig_hist, ax_hist = plt.subplots(figsize=(8, 6))

    ax_hist.hist(residuals, bins=30, density=True, alpha=0.7, edgecolor='black')
    xmin, xmax = ax_hist.get_xlim()

    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, residuals.mean(), residuals.std())
    ax_hist.plot(x, p, 'r-', linewidth=2, label='Normal distribution')

    ax_hist.set_xlabel('Residuals', weight='bold')
    ax_hist.set_ylabel('Density', weight='bold')
    ax_hist.set_title(f'Residual Distribution – {model_name}', weight='bold')
    ax_hist.legend(loc='upper left')
    ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_prefix + f"Histogram_{model_name.replace(' ', '_')}_{date_str}.png")
    plt.savefig(output_prefix + f"Histogram_{model_name.replace(' ', '_')}_{date_str}.pdf")
    plt.show()
    plt.close()


    # ==========================
    # Q–Q Plot (export separately)
    # ==========================
    fig_qq, ax_qq = plt.subplots(figsize=(8, 6))

    stats.probplot(residuals, dist="norm", plot=ax_qq)
    ax_qq.set_title(f'{model_name} Q-Q Plot', weight='bold')
    ax_qq.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_prefix + f"QQplot_{model_name.replace(' ', '_')}_{date_str}.png")
    plt.savefig(output_prefix + f"QQplot_{model_name.replace(' ', '_')}_{date_str}.pdf")
    plt.show()
    plt.close()

    
    print("\n" + "=" * 60)
    print("CONCLUSION:")
    if jb_pvalue > alpha and shapiro_pvalue > alpha:
        print("✓ Residuals are normally distributed - Model assumptions valid!")
    else:
        print("✗ Residuals may not be normally distributed")
        print("  Note: With large samples, minor deviations become significant.")
        print("  Check Q-Q plots for practical assessment of normality.")
    print("=" * 60 + "\n")    





# ============================================================================
# 1. ORDINARY LEAST SQUARES (OLS) - FULL DATASET
# ============================================================================

print(f"\n{'='*60}")
print("STEP 1: ORDINARY LEAST SQUARES (OLS) - FULL DATASET")
print(f"{'='*60}\n")

# Fit OLS model using numpy arrays
model_ols = sm.OLS(y_all, X_all)
results_ols = model_ols.fit()

print(results_ols.summary())

print("\nCoefficient significance based on t-tests:")
for i, param in enumerate(['Intercept', 'x']):
    p = results_ols.pvalues[i]
    print(f"{param}: p-value = {p:.4e} → {'significant' if p < alpha else 'not significant'}")

# Predictions
y_pred_ols = results_ols.predict(X_all)

# Calculate residuals explicitly
residuals_ols = y_all - y_pred_ols

# Parameters
intercept_ols = results_ols.params[0]
slope_ols = results_ols.params[1]
equation_ols = f"y = {slope_ols:.3f}x + {intercept_ols:.3f}"

# Metrics 
metrics_ols = compute_metrics(y_all, y_pred_ols, "OLS (Full Dataset)")

print(f"\n--- OLS Performance Metrics (Full Dataset) ---")
print(f"Equation: {equation_ols}")
print(f"RMSE: {metrics_ols['RMSE']:.4f}")
print(f"MAE: {metrics_ols['MAE']:.4f}")
print(f"R²: {metrics_ols['R²']:.4f}")
print(f"Bias: {metrics_ols['Bias']:.4f}")
print(f"Std: {metrics_ols['Std_Error']:.4f}")
print(f"Pearson r: {metrics_ols['Pearson_r']:.4f} (p={metrics_ols['Pearson_p']:.2e})")
print(f'alpha = {alpha}')

# Test the correct residuals
perform_normality_tests(residuals_ols, alpha, "OLS (Full Dataset)")





# ============================================================================
# 2. ROBUST LINEAR MODEL - HUBER (FULL DATASET)
# ============================================================================

print(f"\n{'='*60}")
print("STEP 2: ROBUST LINEAR MODEL - HUBER (OUTLIER DETECTION)")
print(f"{'='*60}\n")

# Fit Huber model
huber_model = sm.RLM(y_all, X_all, M=sm.robust.norms.HuberT())
huber_res = huber_model.fit()

print(huber_res.summary())

# Weights and outlier detection
weights_hub = huber_res.weights

inliers_hub = weights_hub >= threshold_hub
outliers_hub = weights_hub < threshold_hub

print(f"\nOutlier Detection (threshold={threshold_hub}):")
print(f"Significance (α={alpha*100}%):")
print(f"Number of inliers: {inliers_hub.sum()} ({100*inliers_hub.sum()/len(weights_hub):.1f}%)")
print(f"Number of outliers: {outliers_hub.sum()} ({100*outliers_hub.sum()/len(weights_hub):.1f}%)")

# Visualize weights distribution
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(weights_hub, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(threshold_hub, color='r', linestyle='--', linewidth=2, label=f'Threshold = {threshold_hub}')
ax.set_xlabel('Huber Weights', weight='bold')
ax.set_ylabel('Frequency', weight='bold')
ax.set_title('Distribution of Huber Weights', weight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_prefix + f"Huber_Weights_{date_str}.png")
plt.savefig(output_prefix + f"Huber_Weights_{date_str}.pdf")
plt.show()






# ============================================================================
# 3. RE-FIT OLS MODEL ON INLIERS ONLY
# ============================================================================

print(f"\n{'='*60}")
print("STEP 3: OLS MODEL ON HUBER INLIERS ONLY")
print(f"{'='*60}\n")

# Extract inliers from numpy arrays
X_inliers = X_all[inliers_hub]
y_inliers = y_all[inliers_hub]
x_inliers = x_all[inliers_hub]

# Fit OLS on inliers only
model_ols_inliers = sm.OLS(y_inliers, X_inliers)
results_ols_inliers = model_ols_inliers.fit()

print(results_ols_inliers.summary())

# Predictions on inliers
y_pred_inliers = results_ols_inliers.predict(X_inliers)

# Calculate residuals explicitly
residuals_inliers = y_inliers - y_pred_inliers

# Parameters
intercept_inliers = results_ols_inliers.params[0]
slope_inliers = results_ols_inliers.params[1]
equation_inliers = f"y = {slope_inliers:.3f}x + {intercept_inliers:.3f}"

# Metrics on inliers
metrics_inliers = compute_metrics(y_inliers, y_pred_inliers, "OLS (Inliers Only)")

print(f"\n--- OLS Performance Metrics (Inliers Only) ---")
print(f"Equation: {equation_inliers}")
print(f"RMSE: {metrics_inliers['RMSE']:.4f}")
print(f"MAE: {metrics_inliers['MAE']:.4f}")
print(f"R²: {metrics_inliers['R²']:.4f}")
print(f"Bias: {metrics_inliers['Bias']:.4f}")
print(f"Std: {metrics_inliers['Std_Error']:.4f}")
print(f"Pearson r: {metrics_inliers['Pearson_r']:.4f} (p={metrics_inliers['Pearson_p']:.2e})")

# Test the correct residuals
perform_normality_tests(residuals_inliers, alpha, "OLS (Inliers Only)")





# ============================================================================
# 4. COMPARISON SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("FINAL COMPARISON: OLS (Full) vs OLS (Huber Inliers)")
print(f"{'='*80}\n")

# Get standard errors for slope and intercept
intercept_se_full = results_ols.bse[0]
slope_se_full = results_ols.bse[1]
intercept_se_inliers = results_ols_inliers.bse[0]
slope_se_inliers = results_ols_inliers.bse[1]


comparison_df = pd.DataFrame({
    'Metric': ['N observations', 
               'RMSE', 'MAE', 'R²', 'Bias', 'Std Error',
               'Std Residuals', 'MAD Residuals', 'NMAD Residuals',
               'Pearson r', 
               'Slope', 'Slope SE',
               'Intercept', 'Intercept SE'],
    'OLS (Full)': [
        len(x_all), 
        metrics_ols['RMSE'], 
        metrics_ols['MAE'], 
        metrics_ols['R²'], 
        metrics_ols['Bias'], 
        metrics_ols['Std_Error'],
        metrics_ols['Std_Residuals'],
        metrics_ols['MAD_Residuals'],
        metrics_ols['NMAD_Residuals'],
        metrics_ols['Pearson_r'], 
        slope_ols,
        slope_se_full,
        intercept_ols,
        intercept_se_full
    ],
    'OLS (Inliers)': [
        len(x_inliers), 
        metrics_inliers['RMSE'], 
        metrics_inliers['MAE'],
        metrics_inliers['R²'], 
        metrics_inliers['Bias'], 
        metrics_inliers['Std_Error'],
        metrics_inliers['Std_Residuals'],
        metrics_inliers['MAD_Residuals'],
        metrics_inliers['NMAD_Residuals'],
        metrics_inliers['Pearson_r'], 
        slope_inliers,
        slope_se_inliers,
        intercept_inliers,
        intercept_se_inliers
    ],
    'Improvement': [
        f"-{outliers_hub.sum()}", 
        f"{((metrics_ols['RMSE'] - metrics_inliers['RMSE'])/metrics_ols['RMSE']*100):.2f}%",
        f"{((metrics_ols['MAE'] - metrics_inliers['MAE'])/metrics_ols['MAE']*100):.2f}%",
        f"{((metrics_inliers['R²'] - metrics_ols['R²'])/metrics_ols['R²']*100):.2f}%",
        f"{((abs(metrics_ols['Bias']) - abs(metrics_inliers['Bias']))/abs(metrics_ols['Bias'])*100):.2f}%",
        f"{((metrics_ols['Std_Error'] - metrics_inliers['Std_Error'])/metrics_ols['Std_Error']*100):.2f}%",
        f"{((metrics_ols['Std_Residuals'] - metrics_inliers['Std_Residuals'])/metrics_ols['Std_Residuals']*100):.2f}%",
        f"{((metrics_ols['MAD_Residuals'] - metrics_inliers['MAD_Residuals'])/metrics_ols['MAD_Residuals']*100):.2f}%",
        f"{((metrics_ols['NMAD_Residuals'] - metrics_inliers['NMAD_Residuals'])/metrics_ols['NMAD_Residuals']*100):.2f}%",
        f"{((metrics_inliers['Pearson_r'] - metrics_ols['Pearson_r'])/metrics_ols['Pearson_r']*100):.2f}%",
        f"{((slope_inliers - slope_ols)/slope_ols*100):.2f}%",
        f"{((slope_se_full - slope_se_inliers)/slope_se_full*100):.2f}%",
        f"{((intercept_inliers - intercept_ols)/intercept_ols*100):.2f}%",
        f"{((intercept_se_full - intercept_se_inliers)/intercept_se_full*100):.2f}%"
    ]
})

print(comparison_df.to_string(index=False))

# Export to Excel
excel_filename = f"{output_prefix}OLS_comparison_stats.xlsx"
comparison_df.to_excel(excel_filename, index=False, sheet_name='Statistics')

print(f"\n✓ Results exported to: {excel_filename}")




# ============================================================================
# 5. VISUALIZATION - SCATTER PLOTS
# ============================================================================

# Scatter plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

# OLS Full dataset
axes[0].scatter(x_all, y_all, alpha=0.5, s=20, label='Full Dataset')
axes[0].plot(x_all, y_pred_ols, 'r-', linewidth=2, label=f'OLS: {equation_ols}')
axes[0].plot(x_all, x_all, 'k--', linewidth=1, label='1:1 line')
axes[0].set_xlabel('Original LST [°C]', weight='bold')
axes[0].set_ylabel('Downscaled LST [°C]', weight='bold')
axes[0].set_title(f'LST Comparison in {name_station} in {date_plot} (N={len(x_all)})\nR²={metrics_ols["R²"]:.2f}, RMSE={metrics_ols["RMSE"]:.2f}', weight='bold')
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

# OLS Inliers only
axes[1].scatter(x_inliers, y_inliers, alpha=0.5, s=20, c='green', label='Huber Inliers')
axes[1].plot(x_inliers, y_pred_inliers, 'r-', linewidth=2, label=f'OLS: {equation_inliers}')
axes[1].plot(x_inliers, x_inliers, 'k--', linewidth=1, label='1:1 line')
axes[1].set_xlabel('Original LST [°C]', weight='bold')
axes[1].set_ylabel('Downscaled LST [°C]', weight='bold')
axes[1].set_title(f'LST Comparison in {name_station} in {date_plot} (N={len(x_inliers)})\nR²={metrics_inliers["R²"]:.2f}, RMSE={metrics_inliers["RMSE"]:.2f}', weight='bold')
axes[1].legend(loc='upper left')
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(labelbottom=True, labelleft=True)

plt.tight_layout()
plt.savefig(output_prefix + f"2OLS_{date_str}.png")
plt.savefig(output_prefix + f"2OLS_{date_str}.pdf")
plt.show()


# Export only OLS for Huber Inliers (large format)
fig2, ax = plt.subplots(figsize=(15, 15))

ax.scatter(x_inliers, y_inliers, alpha=0.5, s=20, c='green', label='Huber Inliers')
ax.plot(x_inliers, y_pred_inliers, 'r-', linewidth=2, label=f'OLS: {equation_inliers}')
ax.plot(x_inliers, x_inliers, 'k--', linewidth=1, label='1:1 line')

ax.set_xlabel('Original LST [°C]', weight='bold', fontsize=25)
ax.set_ylabel('Downscaled LST [°C]', weight='bold', fontsize=25)
ax.set_title(
    f'LST Comparison in {name_station} in {date_plot} (N={len(x_inliers)})\n'
    f'R²={metrics_inliers["R²"]:.2f}, RMSE={metrics_inliers["RMSE"]:.2f}',
    fontsize=25, weight='bold'
)

ax.legend(loc='upper left', fontsize=25)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='both', labelsize=25)

plt.tight_layout()
plt.savefig(output_prefix + f"OLS_HuberInliers_{date_str}.png")
plt.savefig(output_prefix + f"OLS_HuberInliers_{date_str}.pdf")
plt.show()

print(f"\n{'='*60}")
print("ANALYSIS COMPLETE")
print(f"{'='*60}\n")







