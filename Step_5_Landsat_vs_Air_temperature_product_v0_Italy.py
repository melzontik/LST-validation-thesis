
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

path = os.path.dirname(os.path.abspath(__file__))+'//'

nome_stazione = 'Zavattari'
output_path = path +'Landsat_vs_air_temperature_'+nome_stazione+'//'
isExist = os.path.exists(output_path)
if not isExist:

  # Create a new directory because it doesnot exist
  os.makedirs(output_path)
print("The new directory is created!")


df_original =pd.read_csv(path+"AIR_temperature_2023//version_0//landsat//gee_landsat//"+nome_stazione+'_gee_collection'+'.csv',sep=',')

df_science =pd.read_csv(path+"AIR_temperature_2023//version_0//landsat//science_landsat//"+nome_stazione+'_science'+'.csv',sep=',')

df_downscaled = pd.read_csv(path+"AIR_temperature_2023//version_0//landsat//downscaled_landsat//"+nome_stazione+'_downscaled'+'.csv',sep=',')

df_arpa =pd.read_csv(path+"AIR_temperature_2023//arpa//"+nome_stazione+'.csv',sep=',') #sep= for ARPA ',' for Tuscany ';'
#df_arpa =pd.read_excel(path+"AIR_temperature_2023//arpa//"+nome_stazione+'.xlsx') #uncomment for Naples

#df_original = df_original.dropna()
#df_science = df_science.dropna()
#df_downscaled = df_downscaled.dropna()

#ROUND df_original for: ARPA 10 min, Tuscany 15 min, Naples 1h
df_original['b_system_time_start'] = pd.to_datetime(df_original['b_system_time_start'], format='%Y-%m-%d %H:%M:%S').dt.round('10min')

# ARPA original format UTC+1 hour. GEE in UTC

# Assuming df_arpa is your DataFrame with a column named 'Data-Ora' in ARPA dataset format
df_arpa['Data-Ora'] = pd.to_datetime(df_arpa['Data-Ora'], format='%Y/%m/%d %H:%M')

# Assuming df_arpa is your DataFrame with a column named 'Data-Ora' in Tuscany dataset format
#df_arpa['Data-Ora'] = pd.to_datetime(df_arpa['Data-Ora'], format='%d/%m/%Y %H:%M')

# Assuming df_arpa is your DataFrame with a column named 'Data-Ora' in Campania dataset format
# No need to convert it

# Convert to the new format '%Y-%m-%d %H:%M:%S' for ARPA and Tuscany
df_arpa['Data-Ora'] = df_arpa['Data-Ora'].dt.strftime('%Y-%m-%d %H:%M:%S') #comment this line for Naples
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
#df_arpa['Medio'] = df_arpa['Medio'].str.replace(',', '.').astype(float)

# Replace commas with dots and convert to float for Campania dataset
#df_arpa['Medio'] = df_arpa['Medio'].astype(str).str.replace(',', '.').astype(float)

#Filter out anomalies from ARPA dataset
df_arpa = df_arpa[df_arpa['Medio'] != -999]

#rename downscaled column 
df_downscaled.rename(columns={'b_system_time_start':'date'}, inplace=True)
#print('science', df_science.head(5))
#print('t_original\n',df_original.head(5))
#print('t_downscaled\n',df_downscaled.head(5))
#print('t_arpa\n',df_arpa.head(5))
#pdb.set_trace()

df_original['c_DATE_ACQUIRED'] = pd.to_datetime(df_original['c_DATE_ACQUIRED'])
df_original.rename(columns={'c_DATE_ACQUIRED':'date'}, inplace=True)

df_downscaled['date'] = pd.to_datetime(df_downscaled['date'])

#https://stackoverflow.com/questions/59394690/python-pandas-find-matching-values-from-two-dataframes-and-return-third-value
df_co = df_original[['date','b_system_time_start', 'j_SAZ', 'i_cloud_cover']].merge(df_downscaled[['date','f_LSTmean_down','g_LSTmedian_down']], left_on='date',
                  right_on='date')

df_co.sort_values(by='date', inplace = True)    

df_science['b_system_time_start'] = pd.to_datetime(df_science['b_system_time_start'])
df_science.rename(columns={'b_system_time_start':'date'}, inplace=True)

df_comp = df_co[['date','b_system_time_start', 'j_SAZ', 'i_cloud_cover', 'f_LSTmean_down','g_LSTmedian_down']].merge(df_science[['date','a_image_id','f_LSTmean','g_LSTmedian']], left_on='date',
                  right_on='date')

df_comp.sort_values(by='date', inplace = True) 
 
#print('tabella prima combinata', df_comp.head(5))
 
#pdb.set_trace()


#df_arpa['b_system_time_start'] = pd.to_datetime(df_arpa['b_system_time_start'], format='%Y-%m-%d %H:%M').dt.round('10min')
df_comparison = df_comp[['date','b_system_time_start', 'j_SAZ', 'i_cloud_cover', 'f_LSTmean_down','g_LSTmedian_down', 'a_image_id','f_LSTmean','g_LSTmedian']].merge(df_arpa[['b_system_time_start','Medio']], left_on='b_system_time_start',
                  right_on='b_system_time_start')


df_comparison.sort_values(by='date', inplace = True)  

#DELTA LST = SATELLITE SENSOR - STAZIONE A TERRA
df_comparison['Delta_LST_original'] = df_comparison['f_LSTmean'] - df_comparison['Medio']
df_comparison['Delta_LST_downscaled'] = df_comparison['f_LSTmean_down'] - df_comparison['Medio']

print('tabella finale', df_comparison.head(5))

#pdb.set_trace()


#TRUE value - ARPA's data, PREDICTED value - satellite data

years = pd.DatetimeIndex(df_comparison['date']).year.to_numpy()
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
    y_pred = data_frame['f_LSTmean']
    y_pred_down = data_frame['f_LSTmean_down']
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
    
    start_date = data_frame['date'].iloc[0].strftime("%Y-%m-%d")
    end_date = data_frame['date'].iloc[-1].strftime("%Y-%m-%d")  
    stats = [data_frame['date'].iloc[0], data_frame['date'].iloc[-1],len(data_frame.index), mediaOrigin, stdOrigin, medianaOrigin, nmadOrigin, mediaDown, stdDown, medianaDown, nmadDown, r2_original, r2_down, r2_sklearn_orig, r2_sklearn_down, r2_sklearn_sat]  
    
    
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
seasons = [('winter', (date(Y,  1,  1),  date(Y,  3, 20))),
           ('spring', (date(Y,  3, 21),  date(Y,  6, 20))),
           ('summer', (date(Y,  6, 21),  date(Y,  9, 22))),
           ('autumn', (date(Y,  9, 23),  date(Y, 12, 20))),
           ('winter', (date(Y, 12, 21),  date(Y, 12, 31)))]
df_comparison['season'] = df_comparison['date'].apply(get_season)

# Iterate through unique seasons and create plots
for season in df_comparison['season'].unique():
    # Filter data for the current season
    season_data = df_comparison[df_comparison['season'] == season]
    if len(season_data) > 0:
      # Create and save the plot for the current season
      fig = plt.figure(figsize=(15, 15))
      plt.title('Original Landsat 8-9 LST vs air Temperature \n@' + nome_stazione + '_' + season + '_' + '2023',weight='bold', fontsize=27)
      ax = fig.gca()
      ax.xaxis.set_major_locator(mdates.MonthLocator())
      date_form = DateFormatter('%Y-%m')
      ax.xaxis.set_major_formatter(date_form)
      plt.plot(season_data['date'], season_data['f_LSTmean'], 'bo--', label='Landsat 8-9')#ORIGINAL 
      plt.plot(season_data['date'], season_data['Medio'], 'ro--', label="ARPA's_Weather_Station")
      plt.grid()
      plt.ylabel('Temperature [°C]',weight='bold', fontsize=25)
      plt.xlabel('Time',weight='bold', fontsize=25)
      # Adjusting tick parameters for larger fontsize
      ax.tick_params(axis='both', which='major', labelsize=20)
      # Setting legend with larger fontsize
      plt.legend(loc='upper left', fontsize=22)      
      plt.tight_layout()
      plt.savefig(output_path+'Landsat_LST_vs_air_Temperature_' +nome_stazione+'_'+season+'_'+'2023'+'.png')
      plt.close()
       
 
for season in df_comparison['season'].unique():
    # Filter data for the current season
    season_data = df_comparison[df_comparison['season'] == season]
    if len(season_data) > 0:         
      #plot with downscaled values and export the table df_comparison as excel
      fig = plt.figure(figsize=(15, 15))
      plt.title('Downscaled Landsat 8-9 LST vs air Temperature \n@' + nome_stazione + '_' + season + '_' + '2023',weight='bold', fontsize=27)
      ax = fig.gca()
      ax.xaxis.set_major_locator(mdates.MonthLocator())
      date_form = DateFormatter('%Y-%m')
      ax.xaxis.set_major_formatter(date_form)
      plt.plot(season_data['date'], season_data['f_LSTmean_down'], 'bo--', label='Landsat 8-9')#DOWNSCALED 
      plt.plot(season_data['date'], season_data['Medio'], 'ro--', label="ARPA's_Weather_Station")
      plt.grid()
      plt.ylabel('Temperature [°C]',weight='bold', fontsize=25)
      plt.xlabel('Time',weight='bold', fontsize=25)
      # Adjusting tick parameters for larger fontsize
      ax.tick_params(axis='both', which='major', labelsize=20)
      # Setting legend with larger fontsize
      plt.legend(loc='upper left', fontsize=22)      
      plt.tight_layout()
      plt.savefig(output_path+'Downscaled_Landsat_LST_vs_air_Temperature_' +nome_stazione+'_'+season+'_'+'2023'+'.png')
      plt.close()

stats = []

print('overall all years')
df_comparison_all = df_comparison.copy()
all_start_date = df_comparison_all['date'].iloc[0].strftime("%Y-%m-%d")
all_end_date = df_comparison_all['date'].iloc[-1].strftime("%Y-%m-%d")
result1 = compute_stats(df_comparison_all, 'overall_all_years')
result1.append('overall all years')
stats.append(result1)

df_comparison_all.to_excel(output_path+'output_overall_all_years_'+all_start_date+'_'+all_end_date+'.xlsx',index=False)
#pdb.set_trace()

#stats_winter = []
print('winter all years')
mask_winter = df_comparison_all['season'] == 'winter'
df_comparison_all_winter = df_comparison_all[mask_winter]
all_winter_start_date = df_comparison_all_winter['date'].iloc[0].strftime("%Y-%m-%d")
all_winter_end_date =   df_comparison_all_winter['date'].iloc[-1].strftime("%Y-%m-%d")
result2 = compute_stats(df_comparison_all_winter, 'winter_all_years')
result2.append('winter all years')
stats.append(result2)
df_comparison_all_winter.to_excel(output_path+'output_winter_all_years_'+all_winter_start_date+'_'+all_winter_end_date+'.xlsx',index=False)

#stats_spring = []
print('spring all years')
mask_spring = df_comparison_all['season'] == 'spring'
df_comparison_all_spring = df_comparison_all[mask_spring]
all_spring_start_date = df_comparison_all_spring['date'].iloc[0].strftime("%Y-%m-%d")
all_spring_end_date =   df_comparison_all_spring['date'].iloc[-1].strftime("%Y-%m-%d")
result5 = compute_stats(df_comparison_all_spring, 'spring_all_years')
result5.append('spring all years')
stats.append(result5)
df_comparison_all_spring.to_excel(output_path+'output_spring_all_years_'+all_spring_start_date+'_'+all_spring_end_date+'.xlsx',index=False)

#stats_summer = []
print('summer all years')
mask_summer = df_comparison_all['season'] == 'summer'
df_comparison_all_summer = df_comparison_all[mask_summer]
all_summer_start_date = df_comparison_all_summer['date'].iloc[0].strftime("%Y-%m-%d")
all_summer_end_date =   df_comparison_all_summer['date'].iloc[-1].strftime("%Y-%m-%d")
result6 = compute_stats(df_comparison_all_summer, 'summer_all_years')
result6.append('summer all years')
stats.append(result6)
df_comparison_all_summer.to_excel(output_path+'output_summer_all_years_'+all_summer_start_date+'_'+all_summer_end_date+'.xlsx',index=False)

#stats_autumn = []
print('autumn all years')
mask_autumn = df_comparison_all['season'] == 'autumn'
df_comparison_all_autumn = df_comparison_all[mask_autumn]
all_autumn_start_date = df_comparison_all_autumn['date'].iloc[0].strftime("%Y-%m-%d")
all_autumn_end_date =   df_comparison_all_autumn['date'].iloc[-1].strftime("%Y-%m-%d")
result7 = compute_stats(df_comparison_all_autumn, 'autumn_all_years')
result7.append('autumn all years')
stats.append(result7)
df_comparison_all_autumn.to_excel(output_path+'output_autumn_all_years_'+all_autumn_start_date+'_'+all_autumn_end_date+'.xlsx',index=False)

for year in years:
            print('Year:', year)
            start_date = str(year)+'-01-01'
            end_date = str(year)+'-12-31'
            mask_time = (df_comparison_all['date'] >= start_date) & (df_comparison_all['date'] <= end_date)
            df_comparison = df_comparison_all.loc[mask_time]

            print('overall year ', year)
            print('df_comparison\n', df_comparison.head(5))
            print('rows df_comparison', len(df_comparison.index))
            result3 = compute_stats(df_comparison, str(year)+'_overall')
            result3.append(str(year)+'_overall')
            stats.append(result3)
            df_comparison.to_excel(output_path+'output_'+str(year)+'_overall_'+start_date+'_'+end_date+'.xlsx',index=False)
            
            print('winter year', year)
            mask_winter_year = df_comparison['season'] == 'winter'
            df_winter_year = df_comparison[ mask_winter_year]
            
            if len(df_winter_year.index) != 0:
                result4 = compute_stats(df_winter_year, str(year)+'_winter')
                result4.append(str(year)+'_winter')
                stats.append(result4)
                year_winter_start_date = df_winter_year['date'].iloc[0].strftime("%Y-%m-%d")
                year_winter_end_date =   df_winter_year['date'].iloc[-1].strftime("%Y-%m-%d")
                df_winter_year.to_excel(output_path+'output_'+str(year)+'_winter_'+year_winter_start_date+'_'+year_winter_end_date+'.xlsx',index=False)
            
            print('spring year', year)
            mask_spring_year = df_comparison['season'] == 'spring'
            df_spring_year = df_comparison[ mask_spring_year]
            
            if len(df_spring_year.index) != 0:
                result8 = compute_stats(df_spring_year, str(year)+'_spring')
                result8.append(str(year)+'_spring')
                stats.append(result8)
                year_spring_start_date = df_spring_year['date'].iloc[0].strftime("%Y-%m-%d")
                year_spring_end_date =   df_spring_year['date'].iloc[-1].strftime("%Y-%m-%d")
                df_spring_year.to_excel(output_path+'output_'+str(year)+'_spring_'+year_spring_start_date+'_'+year_spring_end_date+'.xlsx',index=False)
                
            print('summer year', year)
            mask_summer_year = df_comparison['season'] == 'summer'
            df_summer_year = df_comparison[ mask_summer_year]
            
            if len(df_summer_year.index) != 0:
                result9 = compute_stats(df_summer_year, str(year)+'_summer')
                result9.append( str(year)+'_summer')
                stats.append(result9)
                year_summer_start_date = df_summer_year['date'].iloc[0].strftime("%Y-%m-%d")
                year_summer_end_date =   df_summer_year['date'].iloc[-1].strftime("%Y-%m-%d")
                df_summer_year.to_excel(output_path+'output_'+str(year)+'_summer_'+year_summer_start_date+'_'+year_summer_end_date+'.xlsx',index=False)
                
            print('autumn year', year)
            mask_autumn_year = df_comparison['season'] == 'autumn'
            df_autumn_year = df_comparison[ mask_autumn_year]
            
            if len(df_autumn_year.index) != 0:
                result10 = compute_stats(df_autumn_year, str(year)+'_autumn')
                result10.append(str(year)+'_autumn')
                stats.append(result10)
                year_autumn_start_date = df_autumn_year['date'].iloc[0].strftime("%Y-%m-%d")
                year_autumn_end_date =   df_autumn_year['date'].iloc[-1].strftime("%Y-%m-%d")
                df_autumn_year.to_excel(output_path+'output_'+str(year)+'_autumn_'+year_autumn_start_date+'_'+year_autumn_end_date+'.xlsx',index=False)
            
 
# Create and save the plot for the year
fig = plt.figure(figsize=(15, 15))
plt.title('Landsat 8-9 LST vs Air Temperature in Milan \n@' + nome_stazione + '_2023',weight='bold', fontsize=27)
ax = fig.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator())
date_form = DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(date_form)
plt.plot(df_comparison['date'], df_comparison['f_LSTmean'], 'bo--', label='Landsat 8-9 Original')#ORIGINAL 
plt.plot(df_comparison['date'], df_comparison['f_LSTmean_down'], 'go--', label='Landsat 8-9 Downscaled')
plt.plot(df_comparison['date'], df_comparison['Medio'], 'ro--', label="ARPA's weather station") #Campania/ARPA's/Tuscany
plt.grid()
plt.ylabel('Temperature [°C]',weight='bold', fontsize=25)
plt.xlabel('Time',weight='bold', fontsize=25)
# Adjusting tick parameters for larger fontsize
ax.tick_params(axis='both', which='major', labelsize=20)
# Setting legend with larger fontsize
plt.legend(loc='upper left', fontsize=22)
plt.tight_layout()
plt.savefig(output_path+'V0_Landsat_LST_vs_air_Temperature_' +nome_stazione+'_2023'+'.png')
plt.savefig(output_path+'V0_Landsat_LST_vs_air_Temperature_' +nome_stazione+'_2023'+'.pdf')
    

# Assuming stats is a list of data in the following order: [start_date, end_date, number_of_rows, mean, std, median, NMAD, tcwv_mean, tcwv_median, VA_mean, VA_median, cc_mean, cc_median, season]
stats = np.array(stats, dtype=object)
#stats = [data_frame['date'].iloc[0], data_frame['date'].iloc[-1],len(data_frame.index), mediaOrigin, stdOrigin, medianaOrigin, nmadOrigin, mediaDown, stdDown, medianaDown, nmadDown]  
# Define column names including "season"
column_names = ['start_date', 'end_date', 'number_of_rows', 'delta_LST_mean_origin', 'delta_LST_std_origin', 'delta_LST_median_origin', 'delta_LST_NMAD_origin', 'delta_LST_mean_down', 'delta_LST_std_down', 'delta_LST_median_down', 'delta_LST_NMAD_down' ,'r2_origin','r2_down', 'r2_sklearn_orig', 'r2_sklearn_down', 'r2_sklearn_sat', 'season']

# Create a pandas DataFrame
df = pd.DataFrame(stats, columns=column_names)

# Specify the Excel file path
excel_filename = output_path + 'v0_stats_Landsat_vs_Milan_' + nome_stazione + '.xlsx'

# Save the DataFrame to an Excel file
df.to_excel(excel_filename, index=False)
#plt.show()