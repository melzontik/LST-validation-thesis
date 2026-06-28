# GEE data extraction script
# https://code.earthengine.google.com/5a94e8495f8383da067ec9ac3e9494c9

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import csv
import json
from sklearn import datasets, linear_model
from sklearn.linear_model import RANSACRegressor
import glob
import pdb
from matplotlib import rc,rcParams
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, MonthLocator
from scipy.stats import pearsonr, median_abs_deviation
plt.close('all')

rc('font', weight='bold')

# Function to read and preprocess data files
def reading_file(filename):
    df = pd.read_csv(filename, na_values=["null", "undefined"])
    json_cols = ['col9', 'col995']
    for col in json_cols:
        df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)
    df = df.rename(columns={"col1": "image_id", "col2": "image_date", "col3": "feature_id", 
                            "col4": "feature_zone", 'col992': 'mean', 'col993': 'median', 
                            'col995': 'std'}, errors="raise")
    return df

path = os.path.dirname(os.path.abspath(__file__))+'//'

output_path = path +'Landsat_day_Florence_ransac'+'//'
isExist = os.path.exists(output_path)
if not isExist:

   # Create a new directory 
   os.makedirs(output_path)
   
#For RANSAC use all values from 5 AOI united in one csv
df_super = reading_file(path+'per_pixel_data//landsat_v1//florence_united//super_resolved//'+"super_resolved_2023-01-01"+".csv")
df_orig = reading_file(path+'per_pixel_data//landsat_v1//florence_united//original//'+"original_2023-01-01"+".csv")

date = df_super['image_date'].iloc[0]
print('date',date)

# Initialize lists to store results
dates, ransac_coefs, ransac_intercepts, r_squared_values = [], [], [], []
r_squared_inliers, pearson_coefficients, pearson_inliers, inliers_percent, outliers_percent = [], [], [], [], []


x_for_RANSAC = df_orig['median'].to_numpy() #Change it if needed
y_for_RANSAC = df_super['median'].to_numpy() #mean or median values

mask = ~np.isnan(x_for_RANSAC) & ~np.isnan(y_for_RANSAC)
x_for_RANSAC, y_for_RANSAC = x_for_RANSAC[mask], y_for_RANSAC[mask]
print(len(x_for_RANSAC))
print(len(y_for_RANSAC))

# Compute NMAD for product version 1
residuals = y_for_RANSAC.copy() - x_for_RANSAC.copy()
nmad = median_abs_deviation(residuals, scale='normal') # scale multiply mad by 1.4826
print('NMAD scipy ',nmad)

#Multiply nmad by chosen manual threshold
threshold = nmad * 1.5

# For product Version 0 use predefined threshold:  ransac = linear_model.RANSACRegressor()
ransac = linear_model.RANSACRegressor(residual_threshold = threshold)
ransac.fit(x_for_RANSAC.reshape((x_for_RANSAC.size,1)), y_for_RANSAC.reshape((y_for_RANSAC.size,1)))

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

print("RANSAC inlier_mask:", inlier_mask.shape, inlier_mask)
print("RANSAC Coefficient:", ransac.estimator_.coef_)
print("RANSAC Intercept:", ransac.estimator_.intercept_)
print("Residuals:", residuals.shape, residuals)
print("Sample Residuals:", residuals[:10])
print("Residual Threshold:", ransac.residual_threshold)
print("x_for_RANSAC Mean, Min, Max:", x_for_RANSAC.mean(), x_for_RANSAC.min(), x_for_RANSAC.max())
print("y_for_RANSAC Mean, Min, Max:", y_for_RANSAC.mean(), y_for_RANSAC.min(), y_for_RANSAC.max())

line_X = np.linspace(x_for_RANSAC.min(), x_for_RANSAC.max(),endpoint=True)[:, np.newaxis]
line_y_ransac = ransac.predict(line_X)

#RANSAC plot
fig=plt.figure(figsize=(15,15))
#ax = fig.gca()
plt.title('Florence: LST comparison original Landsat 8-9 day-time\n VS Landsat downscaled '+ df_super['image_date'].iloc[0],weight='bold', fontsize=25)
plt.plot(x_for_RANSAC[inlier_mask], y_for_RANSAC[inlier_mask], 'bo', label='RANSAC inliers')
plt.plot(x_for_RANSAC[outlier_mask], y_for_RANSAC[outlier_mask], 'bx', label='RANSAC outliers')
plt.ylabel('Downscaled median LST [°C]',weight='bold', fontsize=25)
plt.xlabel('Original LST [°C]', weight='bold', fontsize=25)
plt.plot(line_X,line_y_ransac,'r--', linewidth=2, label="RANSAC regressor: Y = "+ format(ransac.estimator_.coef_[0][0],'.2f')+' X + '+format(ransac.estimator_.intercept_[0],'.2f'))
plt.legend(loc='upper left', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.grid()
plt.savefig(output_path+date+'_'+ 'scatter_plot.png')


# Calculate statistics
r_squared = ransac.score(x_for_RANSAC.reshape((x_for_RANSAC.size, 1)), y_for_RANSAC.reshape((y_for_RANSAC.size, 1)))
r_squared_inliers_only = ransac.score(x_for_RANSAC[inlier_mask].reshape((x_for_RANSAC[inlier_mask].size, 1)), y_for_RANSAC[inlier_mask].reshape((y_for_RANSAC[inlier_mask].size, 1)))
print('R2 score', r_squared)

# Calculate R2 using a custom formula
y_pred_ransac = ransac.predict(x_for_RANSAC.reshape(-1, 1))

# Residuals for custom R²
numerator = ((y_for_RANSAC - y_pred_ransac.flatten()) ** 2).sum()
denominator = ((y_for_RANSAC - y_for_RANSAC.mean()) ** 2).sum()

r2_custom = 1 - (numerator / denominator)
print(f"Custom R² : {r2_custom}")

pearson_coefficient = pearsonr(x_for_RANSAC, y_for_RANSAC)[0]
pearson_inlier = pearsonr(x_for_RANSAC[inlier_mask], y_for_RANSAC[inlier_mask])[0]
num_inliers = np.sum(inlier_mask)
num_outliers = np.sum(outlier_mask)
    
# Check total number of points after filtering and ensure complete accounting
total_num_px = num_inliers + num_outliers
if total_num_px != len(x_for_RANSAC):
    raise ValueError("Mismatch in total number of data points after filtering. Check data processing steps.")
    
inlier_percentage = (num_inliers / total_num_px) * 100
outlier_percentage = (num_outliers / total_num_px) * 100 
   
# Log information for export
dates.append(date)
ransac_coefs.append(ransac.estimator_.coef_[0].item())
ransac_intercepts.append(ransac.estimator_.intercept_[0].item())
r_squared_values.append(r_squared)
r_squared_inliers.append(r_squared_inliers_only)
pearson_coefficients.append(pearson_coefficient)
pearson_inliers.append(pearson_inlier)
inliers_percent.append(inlier_percentage)
outliers_percent.append(outlier_percentage)

#Stats for export
data = {'Date': dates, 'RANSAC Coefficient': ransac_coefs, 'RANSAC Intercept': ransac_intercepts,
        'R-squared': r_squared_values, 'R-squared_inliers': r_squared_inliers, 'Pearson': pearson_coefficients, 
        'Pearson_for_inliers': pearson_inliers, 'inliers_percent': inliers_percent, 'outliers_percent': outliers_percent}
df_export = pd.DataFrame(data)
df_export.to_excel(output_path + 'landsat_florence_ransac_' + date + '.xlsx', index=False)
