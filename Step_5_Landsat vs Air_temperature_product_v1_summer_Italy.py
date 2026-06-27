import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, MonthLocator
import scipy 
from datetime import date, datetime, timezone, timedelta
from sklearn.metrics import r2_score 
plt.close('all')

import pdb
#lo script per i dati di ARPA Lombardia
path = os.path.dirname(os.path.abspath(__file__))+'//'

nome_stazione = 'Orto_Botanico'
output_path = path +'Landsat_Day_vs_Tuscany_v1_tryy'+nome_stazione+'//'
isExist = os.path.exists(output_path)
if not isExist:

  # Create a new directory because it doesnot exist
  os.makedirs(output_path)
print("The new directory is created!")


df_original =pd.read_csv(path+"AIR_temperature_2023//version_1//landsat//original//"+nome_stazione+'_Landsat_science'+'.csv',sep=',')

df_downscaled = pd.read_csv(path+"AIR_temperature_2023//version_1//landsat//downscaled//"+nome_stazione+'_Landsat_downscaled'+'.csv',sep=',')

df_gee = pd.read_csv(path+"AIR_temperature_2023//version_1//landsat//gee_landsat//"+nome_stazione+'_gee_collection'+'.csv',sep=',')

df_arpa =pd.read_csv(path+"AIR_temperature_2023//arpa//"+nome_stazione+'.csv',sep=';') #for ARPA ',' for Tuscany ';'
#df_arpa =pd.read_excel(path+"AIR_temperature_2023//arpa//"+nome_stazione+'.xlsx') #uncomment for Naples

#df_original = df_original.dropna() 
#df_downscaled = df_downscaled.dropna()

#print(df_original.head(10))
#pdb.set_trace()

# Convert 'b_system_time_start' to datetime format
df_gee['b_system_time_start'] = pd.to_datetime(df_gee['b_system_time_start'], format='%Y-%m-%d %H:%M:%S')

# Convert 'b_system_time_start' to string format
df_gee['b_system_time_start_str'] = df_gee['b_system_time_start'].dt.strftime('%Y-%m-%d %H:%M:%S')

#ROUND satellite timestamp to confron with: ARPA 10 min, Tuscany 15 min, Naples 1h
df_gee['b_system_time_start'] = pd.to_datetime(df_gee['b_system_time_start'], format='%Y-%m-%d %H:%M:%S').dt.round('15min')

# Assuming df_arpa is your DataFrame with a column named 'Data-Ora' in ARPA dataset format
#df_arpa['Data-Ora'] = pd.to_datetime(df_arpa['Data-Ora'], format='%Y/%m/%d %H:%M')

# Assuming df_arpa is your DataFrame with a column named 'Data-Ora' in Tuscany dataset format
df_arpa['Data-Ora'] = pd.to_datetime(df_arpa['Data-Ora'], format='%d/%m/%Y %H:%M')

# Assuming df_arpa is your DataFrame with a column named 'Data-Ora' in Campania dataset format
# No need to convert it

# Convert to the new format '%Y-%m-%d %H:%M:%S' for ARPA and Tuscany
df_arpa['Data-Ora'] = df_arpa['Data-Ora'].dt.strftime('%Y-%m-%d %H:%M:%S')#for Campania comment this line
df_arpa['Data-Ora'] = pd.to_datetime(df_arpa['Data-Ora'], format='%Y-%m-%d %H:%M:%S')

# Landsat UTC (convert the weather datasets to UTC)
# Weather data is in Central European Time (CET), without summer/winter changes

fixed_utc1 = timezone(timedelta(hours=1))

# Convert the datetime column to the UTC timezone
df_arpa["Data-Ora"] = df_arpa["Data-Ora"].dt.tz_localize(fixed_utc1)
df_arpa["Data-Ora"] = df_arpa["Data-Ora"].dt.tz_convert("UTC")
df_arpa["Data-Ora"] = df_arpa["Data-Ora"].dt.tz_localize(None)

# Now df_arpa['Data-Ora'] contains datetime values in UTC for Italy

df_arpa['Data-Ora'] = pd.to_datetime(df_arpa['Data-Ora'], format='%Y-%m-%d %H:%M')

df_arpa.rename(columns={'Data-Ora':'b_system_time_start'}, inplace=True)
df_arpa.rename(columns={' Medio':'Medio'}, inplace=True)

# Replace commas with dots and convert to float for Tuscany dataset
df_arpa['Medio'] = df_arpa['Medio'].str.replace(',', '.').astype(float)

# Replace commas with dots and convert to float for Campania dataset
#df_arpa['Medio'] = df_arpa['Medio'].astype(str).str.replace(',', '.').astype(float)

#Filter out anomalies from ARPA dataset
#df_arpa = df_arpa[df_arpa['Medio'] != -999] 

#print('df_arpa', df_arpa)
#pdb.set_trace()

#############merge df_gee with df_arpa###############

# Ensure b_system_time_start is a datetime column
df_gee['b_system_time_start'] = pd.to_datetime(df_gee['b_system_time_start'])

# Convert to the desired format
df_gee['b_system_time_start'] = df_gee['b_system_time_start'].dt.strftime('%Y-%m-%d %H:%M:%S')

df_arpa['b_system_time_start'] = pd.to_datetime(df_arpa['b_system_time_start'])
df_arpa['b_system_time_start'] = df_arpa['b_system_time_start'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Perform the merge
df_gee_arpa = df_gee[['b_system_time_start', 'c_DATE_ACQUIRED', 'g_LSTmedian']].merge(
    df_arpa[['b_system_time_start', 'Medio']],
    left_on='b_system_time_start',
    right_on='b_system_time_start',
    how='inner'  # Use 'inner' join to keep only matching rows
)

# Optional: Drop the duplicate column
#df_gee_arpa = df_modis_gbov.drop(columns=['b_system_time_start'])

#print('df_gee_arpa\n', df_gee_arpa)

#pdb.set_trace()

#for merge with monthly median products i should first count median for all column of interest

# Convert ' a_image_id' to datetime format
df_gee_arpa['c_DATE_ACQUIRED'] = pd.to_datetime(df_gee_arpa['c_DATE_ACQUIRED'], format='%Y-%m-%d')

# Extract year and month for grouping
df_gee_arpa['year_month'] = df_gee_arpa['c_DATE_ACQUIRED'].dt.to_period('M')


# Convert 'b_system_time_start' to datetime

df_gee_arpa['b_system_time_start'] = pd.to_datetime(df_gee_arpa['b_system_time_start'], format='%Y-%m-%d %H:%M:%S')
df_gee_arpa['time'] = df_gee_arpa['b_system_time_start'].dt.time  # Extract only the time
df_gee_arpa['time'] = pd.to_datetime(df_gee_arpa['time'], format='%H:%M:%S')

# Calculate seconds from the time since midnight
df_gee_arpa['i_utc_seconds'] = (
    df_gee_arpa['time'] - df_gee_arpa['time'].dt.normalize()
).dt.total_seconds()

# Calculate monthly medians
monthly_medians = df_gee_arpa.groupby('year_month').agg({
    'Medio' : 'median',
    'i_utc_seconds': 'median'
}).reset_index()

# Convert the 'year_month' to the first day of each month
monthly_medians['month_start_date'] = monthly_medians['year_month'].dt.start_time

# Convert 'i_utc_seconds' back to time format
monthly_medians['i_utc_median'] = pd.to_timedelta(
    monthly_medians['i_utc_seconds'], unit='s'
).apply(lambda x: (x.components.hours, x.components.minutes, x.components.seconds))

# Format time as "HH:MM:SS"
monthly_medians['i_utc_median'] = monthly_medians['i_utc_median'].apply(
    lambda x: f"{x[0]:02}:{x[1]:02}:{x[2]:02}"
)

# Select and rename columns for clarity
monthly_medians = monthly_medians[['month_start_date', 'Medio', 'i_utc_median']]

df_gee_arpa = monthly_medians
# Display the result
#print('df_gee_arpa_monthly_medians\n', df_gee_arpa)
#pdb.set_trace() 


##################################merge with downscaled dataset

# Convert 'b_system_time_start' to datetime format
df_downscaled['b_system_time_start'] = pd.to_datetime(df_downscaled['b_system_time_start'])
df_gee_arpa['month_start_date'] = pd.to_datetime(df_gee_arpa['month_start_date'])

df_comp = df_gee_arpa[['month_start_date', 'Medio']].merge(df_downscaled[['b_system_time_start', 'g_LSTmedian_down']],left_on='month_start_date',right_on='b_system_time_start')

df_comp.sort_values(by='month_start_date', inplace = True) 
 
print('df_comp', df_comp)
#pdb.set_trace()

####################merge with original dataset

# Convert 'b_system_time_start' to datetime format
df_original['b_system_time_start'] = pd.to_datetime(df_original['b_system_time_start'])
df_comp['b_system_time_start'] = pd.to_datetime(df_comp['b_system_time_start'])

df_final = df_comp[['b_system_time_start','month_start_date','g_LSTmedian_down','Medio']].merge(df_original[['b_system_time_start', 'g_LSTmedian']],left_on='b_system_time_start',right_on='b_system_time_start', how = 'outer')

df_final.sort_values(by='b_system_time_start', inplace = True) 
df_final.interpolate(method='linear', inplace=True)
 
print('df_final_add_original', df_final)
pdb.set_trace()

# Convert 'date' column to datetime type if it's not already
df_final['month_start_date'] = pd.to_datetime(df_final['month_start_date'])

# Extract year and month to group by
df_final['year'] = df_final['month_start_date'].dt.year
df_final['month'] = df_final['month_start_date'].dt.month


# Create a 'month_start_date' column for sorting
df_final['month_start_date'] = pd.to_datetime(df_final[['year', 'month']].assign(day=15))#TIGHT VALUES WITH MIDDLE OF THE MONTH

# Sort the DataFrame by 'month_start_date' column
df_final.sort_values(by='month_start_date', inplace=True)

# Drop the 'year' and 'month' columns if they are no longer needed
df_final.drop(columns=['year', 'month'], inplace=True)

#print('df_final_first_assighned_15_day',df_final)
#pdb.set_trace()

#DELTA LST = SATELLITE SENSOR - STAZIONE A TERRA
df_final['Delta_LST_original'] = df_final['g_LSTmedian'] - df_final['Medio']
df_final['Delta_LST_downscaled'] = df_final['g_LSTmedian_down'] - df_final['Medio']
#df_final['Delta_LST'] = df_final['g_LSTmedian'] - df_final['g_LSTmedian_down']
# Display the resulting dataframe
#print('FINAL table for stats', df_final)
#pdb.set_trace()

#STATS
years = pd.DatetimeIndex(df_final['month_start_date']).year.to_numpy()
years = np.unique(years, return_index=False)
#print('years', years)
date_form = DateFormatter("%d-%m-%y")



def compute_stats (data_frame, info_periodo=''):#, start_date, end_date):
  
    mediaOrigin = data_frame['Delta_LST_original'].mean(axis=0)
    stdOrigin = data_frame['Delta_LST_original'].std(axis=0)
    medianaOrigin = data_frame['Delta_LST_original'].median(axis=0)
    nmadOrigin = scipy.stats.median_abs_deviation(data_frame['Delta_LST_original'], scale='normal')
        
    mediaDown = data_frame['Delta_LST_downscaled'].mean(axis=0)
    stdDown = data_frame['Delta_LST_downscaled'].std(axis=0)
    medianaDown = data_frame['Delta_LST_downscaled'].median(axis=0)
    nmadDown = scipy.stats.median_abs_deviation(data_frame['Delta_LST_downscaled'], scale='normal')   
    
    # Calculate R2 using a custom formula
    y_true = data_frame['Medio']
    y_pred = data_frame['g_LSTmedian']
    y_pred_down = data_frame['g_LSTmedian_down']
    y_mean = y_true.mean()
    numerator = ((y_true - y_pred) ** 2).sum()
    denominator = ((y_true - y_mean) ** 2).sum()
    r2_original = 1 - (numerator / denominator)   
    # R2 for Downscaled dataset
    numerator_down = ((y_true - y_pred_down) ** 2).sum()
    denominator_down = ((y_true - y_mean) ** 2).sum()
    r2_down = 1 - (numerator_down / denominator_down)
    # Calculate R² using scikit-learn
    r2_sklearn_orig = r2_score(y_true, y_pred)
    r2_sklearn_down = r2_score(y_true, y_pred_down)
    r2_sklearn_sat = r2_score(y_pred, y_pred_down)
    
    start_date = data_frame['month_start_date'].iloc[0].strftime("%Y-%m-%d")
    end_date = data_frame['month_start_date'].iloc[-1].strftime("%Y-%m-%d")  
    stats = [data_frame['month_start_date'].iloc[0], data_frame['month_start_date'].iloc[-1],len(data_frame.index), mediaOrigin, stdOrigin, medianaOrigin, nmadOrigin, mediaDown, stdDown, medianaDown, nmadDown, r2_original, r2_down, r2_sklearn_orig, r2_sklearn_down, r2_sklearn_sat]  
    
    
    return stats 
    
def get_season(dt):
    #https://stackoverflow.com/questions/63531607/how-to-get-season-and-if-a-date-range-falls-on-a-us-holiday-with-pandas
    if isinstance(dt, datetime):
        dt = dt.date()
    dt = dt.replace(year=Y)
    return next(season for season, (start, end) in seasons
                if start <= dt <= end)

Y = 2000 # dummy leap year to allow input X-02-29 (leap day)
#  stagioni astronomiche
# seasons = [('winter', (date(Y,  1,  1),  date(Y,  3, 20))),
           # ('spring', (date(Y,  3, 21),  date(Y,  6, 20))), 
           # ('summer', (date(Y,  6, 21),  date(Y,  9, 22))),
           # ('autumn', (date(Y,  9, 23),  date(Y, 12, 20))),
           # ('winter', (date(Y, 12, 21),  date(Y, 12, 31)))]

#  stagioni meteorologiche           
seasons = [('winter', (date(Y,  1,  1),  date(Y,  2, 28))),
           ('spring', (date(Y,  3, 1),  date(Y,  5, 31))),
           ('summer', (date(Y,  6, 1),  date(Y,  8, 31))),
           ('autumn', (date(Y,  9, 1),  date(Y, 11, 30))),
           ('winter', (date(Y, 12, 1),  date(Y, 12, 31)))]
df_final['season'] = df_final['month_start_date'].apply(get_season)

# Iterate through unique seasons and create plots
for season in df_final['season'].unique():
    # Filter data for the current season
    season_data = df_final[df_final['season'] == season]
    if len(season_data) > 0:
      # Create and save the plot for the current season
      fig = plt.figure(figsize=(15, 15))
      plt.title('Original Landsat 8-9 LST vs Air Temperature \n@' + nome_stazione + '_' + season + '_' + '2023',weight='bold', fontsize=25)
      ax = fig.gca()
      ax.xaxis.set_major_locator(mdates.MonthLocator())
      date_form = DateFormatter('%Y-%m')
      ax.xaxis.set_major_formatter(date_form)
      plt.plot(season_data['month_start_date'], season_data['g_LSTmedian'], 'bo--', label='Landsat 8-9')#ORIGINAL MODIS
      plt.plot(season_data['month_start_date'], season_data['Medio'], 'ro--', label="Tuscany_Weather_Station")
      
      plt.ylabel('Temperature [°C]',weight='bold', fontsize=25)
      plt.xlabel('Time',weight='bold', fontsize=25)
      # Adjusting tick parameters for larger fontsize
      ax.tick_params(axis='both', which='major', labelsize=20)
      # Setting legend with larger fontsize
      plt.legend(loc='upper right', fontsize=20)   
      plt.grid()      
      #plt.tight_layout()
      plt.savefig(output_path+'V1_Original_Landsat_LST_vs_air_Temperature_' +nome_stazione+'_'+season+'_'+'2023'+'.png')
      plt.close()
       
 
for season in df_final['season'].unique():
    # Filter data for the current season
    season_data = df_final[df_final['season'] == season]
    if len(season_data) > 0:         
      #plot with downscaled values and export the table df_final as excel
      fig = plt.figure(figsize=(15, 15))
      plt.title('Downscaled Landsat 8-9 LST vs Air Temperature \n@' + nome_stazione + '_' + season + '_' + '2023',weight='bold', fontsize=25)
      ax = fig.gca()
      ax.xaxis.set_major_locator(mdates.MonthLocator())
      date_form = DateFormatter('%Y-%m')
      ax.xaxis.set_major_formatter(date_form)
      plt.plot(season_data['month_start_date'], season_data['g_LSTmedian_down'], 'bo--', label='Landsat 8-9')#DOWNSCALED MODIS
      plt.plot(season_data['month_start_date'], season_data['Medio'], 'ro--', label="Tuscany_Weather_Station")
      plt.grid()
      plt.ylabel('Temperature [°C]',weight='bold', fontsize=25)
      plt.xlabel('Time',weight='bold', fontsize=25)
      # Adjusting tick parameters for larger fontsize
      ax.tick_params(axis='both', which='major', labelsize=20)
      # Setting legend with larger fontsize
      plt.legend(loc='upper right', fontsize=20)      
      #plt.tight_layout()
      plt.savefig(output_path+'V1_Downscaled_Landsat_LST_vs_air_Temperature_' +nome_stazione+'_'+season+'_'+'2023'+'.png')
      plt.close()

stats = []

print('overall all years')
df_final_all = df_final.copy()
all_start_date = df_final_all['month_start_date'].iloc[0].strftime("%Y-%m-%d")
all_end_date = df_final_all['month_start_date'].iloc[-1].strftime("%Y-%m-%d")
result1 = compute_stats(df_final_all, 'overall_all_years')
result1.append('overall all years')
stats.append(result1)

df_final_all.to_excel(output_path+'output_overall_all_years_'+all_start_date+'_'+all_end_date+'.xlsx',index=False)
#pdb.set_trace()

#stats_winter = []
print('winter all years')
mask_winter = df_final_all['season'] == 'winter'
df_final_all_winter = df_final_all[mask_winter]
all_winter_start_date = df_final_all_winter['month_start_date'].iloc[0].strftime("%Y-%m-%d")
all_winter_end_date =   df_final_all_winter['month_start_date'].iloc[-1].strftime("%Y-%m-%d")
result2 = compute_stats(df_final_all_winter, 'winter_all_years')
result2.append('winter all years')
stats.append(result2)
df_final_all_winter.to_excel(output_path+'output_winter_all_years_'+all_winter_start_date+'_'+all_winter_end_date+'.xlsx',index=False)

#stats_spring = []
print('spring all years')
mask_spring = df_final_all['season'] == 'spring'
df_final_all_spring = df_final_all[mask_spring]
all_spring_start_date = df_final_all_spring['month_start_date'].iloc[0].strftime("%Y-%m-%d")
all_spring_end_date =   df_final_all_spring['month_start_date'].iloc[-1].strftime("%Y-%m-%d")
result5 = compute_stats(df_final_all_spring, 'spring_all_years')
result5.append('spring all years')
stats.append(result5)
df_final_all_spring.to_excel(output_path+'output_spring_all_years_'+all_spring_start_date+'_'+all_spring_end_date+'.xlsx',index=False)

#stats_summer = []
print('summer all years')
mask_summer = df_final_all['season'] == 'summer'
df_final_all_summer = df_final_all[mask_summer]
all_summer_start_date = df_final_all_summer['month_start_date'].iloc[0].strftime("%Y-%m-%d")
all_summer_end_date =   df_final_all_summer['month_start_date'].iloc[-1].strftime("%Y-%m-%d")
result6 = compute_stats(df_final_all_summer, 'summer_all_years')
result6.append('summer all years')
stats.append(result6)
df_final_all_summer.to_excel(output_path+'output_summer_all_years_'+all_summer_start_date+'_'+all_summer_end_date+'.xlsx',index=False)

#stats_autumn = []
print('autumn all years')
mask_autumn = df_final_all['season'] == 'autumn'
df_final_all_autumn = df_final_all[mask_autumn]
all_autumn_start_date = df_final_all_autumn['month_start_date'].iloc[0].strftime("%Y-%m-%d")
all_autumn_end_date =   df_final_all_autumn['month_start_date'].iloc[-1].strftime("%Y-%m-%d")
result7 = compute_stats(df_final_all_autumn, 'autumn_all_years')
result7.append('autumn all years')
stats.append(result7)
df_final_all_autumn.to_excel(output_path+'output_autumn_all_years_'+all_autumn_start_date+'_'+all_autumn_end_date+'.xlsx',index=False)

for year in years:
            print('Year:', year)
            start_date = str(year)+'-01-01'
            end_date = str(year)+'-12-31'
            mask_time = (df_final_all['month_start_date'] >= start_date) & (df_final_all['month_start_date'] <= end_date)
            df_final = df_final_all.loc[mask_time]

            print('overall year ', year)
            print('df_final\n', df_final.head(5))
            print('rows df_final', len(df_final.index))
            result3 = compute_stats(df_final, str(year)+'_overall')
            result3.append(str(year)+'_overall')
            stats.append(result3)
            df_final.to_excel(output_path+'output_'+str(year)+'_overall_'+start_date+'_'+end_date+'.xlsx',index=False)
            
            print('winter year', year)
            mask_winter_year = df_final['season'] == 'winter'
            df_winter_year = df_final[ mask_winter_year]
            
            if len(df_winter_year.index) != 0:
                result4 = compute_stats(df_winter_year, str(year)+'_winter')
                result4.append(str(year)+'_winter')
                stats.append(result4)
                year_winter_start_date = df_winter_year['month_start_date'].iloc[0].strftime("%Y-%m-%d")
                year_winter_end_date =   df_winter_year['month_start_date'].iloc[-1].strftime("%Y-%m-%d")
                df_winter_year.to_excel(output_path+'output_'+str(year)+'_winter_'+year_winter_start_date+'_'+year_winter_end_date+'.xlsx',index=False)
            
            print('spring year', year)
            mask_spring_year = df_final['season'] == 'spring'
            df_spring_year = df_final[ mask_spring_year]
            
            if len(df_spring_year.index) != 0:
                result8 = compute_stats(df_spring_year, str(year)+'_spring')
                result8.append(str(year)+'_spring')
                stats.append(result8)
                year_spring_start_date = df_spring_year['month_start_date'].iloc[0].strftime("%Y-%m-%d")
                year_spring_end_date =   df_spring_year['month_start_date'].iloc[-1].strftime("%Y-%m-%d")
                df_spring_year.to_excel(output_path+'output_'+str(year)+'_spring_'+year_spring_start_date+'_'+year_spring_end_date+'.xlsx',index=False)
                
            print('summer year', year)
            mask_summer_year = df_final['season'] == 'summer'
            df_summer_year = df_final[ mask_summer_year]
            
            if len(df_summer_year.index) != 0:
                result9 = compute_stats(df_summer_year, str(year)+'_summer')
                result9.append( str(year)+'_summer')
                stats.append(result9)
                year_summer_start_date = df_summer_year['month_start_date'].iloc[0].strftime("%Y-%m-%d")
                year_summer_end_date =   df_summer_year['month_start_date'].iloc[-1].strftime("%Y-%m-%d")
                df_summer_year.to_excel(output_path+'output_'+str(year)+'_summer_'+year_summer_start_date+'_'+year_summer_end_date+'.xlsx',index=False)
                
            print('autumn year', year)
            mask_autumn_year = df_final['season'] == 'autumn'
            df_autumn_year = df_final[ mask_autumn_year]
            
            if len(df_autumn_year.index) != 0:
                result10 = compute_stats(df_autumn_year, str(year)+'_autumn')
                result10.append(str(year)+'_autumn')
                stats.append(result10)
                year_autumn_start_date = df_autumn_year['month_start_date'].iloc[0].strftime("%Y-%m-%d")
                year_autumn_end_date =   df_autumn_year['month_start_date'].iloc[-1].strftime("%Y-%m-%d")
                df_autumn_year.to_excel(output_path+'output_'+str(year)+'_autumn_'+year_autumn_start_date+'_'+year_autumn_end_date+'.xlsx',index=False)
            

# Create and save the plot for the year
fig = plt.figure(figsize=(15, 15))
plt.title('Landsat Day LST vs Air Temperature in Florence \n@' + nome_stazione + '_2023',weight='bold', fontsize=25)
ax = fig.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator())
date_form = DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(date_form)
plt.plot(df_final['month_start_date'], df_final['g_LSTmedian'], 'bo--', label='Landsat_Day_Original')#ORIGINAL Landsat
plt.plot(df_final['month_start_date'], df_final['g_LSTmedian_down'], 'go--', label='Landsat_Day_Downscaled')
plt.plot(df_final['month_start_date'], df_final['Medio'], 'ro--', label="Tuscany_weather_station")
plt.grid()
plt.ylabel('Temperature [°C]',weight='bold', fontsize=25)
plt.xlabel('Time',weight='bold', fontsize=25)
plt.legend(loc='upper left', fontsize=20)
# Adjusting tick parameters for larger fontsize
ax.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout()
plt.savefig(output_path+'Landsat_Day_LST_vs_air_Temperature_' +nome_stazione+'_2023_v1'+'.png')
    
    
fig, ax = plt.subplots(figsize=(15, 15))
plt.title('Temporal Relationship Between Landsat Day LST \nvs Air Temperature @' + nome_stazione + '_2023',weight='bold', fontsize=25) 
# Manually adjust the y-values for better visibility
offset = 5.0  # You can adjust this value to control the vertical separation
plt.plot(df_final['month_start_date'], df_final['g_LSTmedian'] + offset, '-', label='Landsat_Day_Original', color='blue')
plt.plot(df_final['month_start_date'], df_final['g_LSTmedian_down'], '-', label='Landsat_Day_Downscaled', color='green')
plt.plot(df_final['month_start_date'], df_final['Medio']- offset, '-', label="Tuscany_weather_station", color='red')
plt.grid()
plt.xlabel('Time',weight='bold', fontsize=25)
#plt.ylabel('Temperature Trend [°C]')
plt.legend(loc='upper left', fontsize=20)
# Setting date format and major ticks
date_form = DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(MonthLocator())
# Remove y-axis ticks and labels
plt.yticks([])
plt.tight_layout()
plt.tick_params(axis='x', which='major', labelsize=20)
plt.savefig(output_path + 'TREND_Landsat_Day_LST_vs_air_Temperature_' + nome_stazione + '_' + '2023_v1' + '.png')

# Assuming stats is a list of data in the following order: [start_date, end_date, number_of_rows, mean, std, median, NMAD, tcwv_mean, tcwv_median, VA_mean, VA_median, cc_mean, cc_median, season]
stats = np.array(stats, dtype=object)
#stats = [data_frame['month_start_date'].iloc[0], data_frame['month_start_date'].iloc[-1],len(data_frame.index), mediaOrigin, stdOrigin, medianaOrigin, nmadOrigin, mediaDown, stdDown, medianaDown, nmadDown]  
# Define column names including "season"
column_names = ['start_date', 'end_date', 'number_of_rows', 'delta_LST_mean_origin', 'delta_LST_std_origin', 'delta_LST_median_origin', 'delta_LST_NMAD_origin', 'delta_LST_mean_down', 'delta_LST_std_down', 'delta_LST_median_down', 'delta_LST_NMAD_down' ,'r2_origin','r2_down', 'r2_sklearn_orig', 'r2_sklearn_down', 'r2_sklearn_sat', 'season']
# Create a pandas DataFrame
df = pd.DataFrame(stats, columns=column_names)

# Specify the Excel file path
excel_filename = output_path + 'stats_Landsat_vs_Tuscany_' + nome_stazione + '.xlsx'

# Save the DataFrame to an Excel file
df.to_excel(excel_filename, index=False)