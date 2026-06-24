
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, MonthLocator
import scipy 
from datetime import date, datetime
plt.close('all')

import pdb

path = os.path.dirname(os.path.abspath(__file__))+'//'

nome_stazione = 'USA_GCMK'
output_path = path +'CELSIUS_Cross_satellite_buffer_l8_modis_1100_vs_01_px_'+nome_stazione+'//'
isExist = os.path.exists(output_path)
if not isExist:

  # Create a new directory
  os.makedirs(output_path)
print("The new directory is created!")


df_landsat =pd.read_csv(path+"GEE//"+nome_stazione+'_L8_buffer_1100'+'.csv',sep=',')

df_modis =pd.read_csv(path+"GEE//"+nome_stazione+'_Modis_Day_TZonly_buffer_01'+'.csv',sep=',')

df_stazione = pd.read_csv(path+"GBOV//"+'GBOV_RM09_GoodwinCreek_w15_20130101T000100Z_20221231T235900Z_049_ACR_V2.0'+'.csv', sep=';')

df_tcwv =pd.read_csv(path+"VAPOUR//"+nome_stazione+'_VAPOUR'+'.csv',sep=',')

df_landsat = df_landsat.dropna()
df_modis = df_modis.dropna()
df_stazione = df_stazione.dropna()

#https://stackoverflow.com/questions/62917882/convert-datetime64ns-utc-pandas-column-to-datetime
df_stazione['TIME_IS'] =  pd.to_datetime(df_stazione['TIME_IS']).dt.tz_localize(None)#, format='%d%b%Y:%H:%M:%S.%f')

# https://stephenallwright.com/pandas-round-datetime-to-minute/
df_landsat['b_system_time_start']= pd.to_datetime(df_landsat['b_system_time_start']).dt.round("min")

df_stazione.rename(columns={'TIME_IS':'time'}, inplace=True)
df_landsat.rename(columns={'b_system_time_start':'time'}, inplace=True)

df_modis['b_system_time_start']= pd.to_datetime(df_modis['b_system_time_start']).dt.round("min")
df_modis.rename(columns={'b_system_time_start':'time'}, inplace=True)

#Filter out anomalies from GBOV dataset
df_stazione = df_stazione[df_stazione['LST_STD'] <= 1.5] 

#CONVERTIRE TUTTO IN GRADI CENTIGRADI
df_stazione['LST'] -= 273.15
df_modis['f_LSTmean'] -= 273.15
df_landsat['f_LSTmean'] -= 273.15

# Convert the 'time' column to datetime
df_tcwv['valid_time'] = pd.to_datetime(df_tcwv['valid_time'])


desired_time = "16:00:00"  # Desired time as a string
latitude = 34.3
longitude = -89.9

# Use loc to filter the DataFrame based on conditions
filtered_df_tcwv = df_tcwv.loc[
    (df_tcwv['valid_time'].dt.strftime('%H:%M:%S') == desired_time)
    &
    (df_tcwv['latitude'].astype(float) == latitude) &
    (df_tcwv['longitude'].astype(float) == longitude)
]

filtered_df_tcwv.loc[:, 'valid_time'] = pd.to_datetime(filtered_df_tcwv['valid_time']).dt.date


#print('t_tcwv\n',filtered_df_tcwv['valid_time'].head(5))
# print('t_station\n',df_stazione.head(5))
# print('t_landsat\n',df_landsat.head(5))
# print('t_modis\n',df_modis.head(5))
# pdb.set_trace()
#https://stackoverflow.com/questions/59394690/python-pandas-find-matching-values-from-two-dataframes-and-return-third-value
df_comp = df_landsat[['time', 'a_image_id', 'e_number_of_pixels', 'f_LSTmean', 'g_LSTmedian','h_LSTstd', 'i_cloud_cover','j_roll','k_SZA_calc', 'l_SZA_mean', 'm_SZA_median', 'n_SZA_std', 'o_SAA_mean', 'p_SAA_median', 'q_SAA_std', 'r_VZA_mean','s_VZA_median','t_VZA_std', 'u_VAA_mean', 'v_VAA_median', 'w_VAA_std','x_SZA_number_of_pixels','y_SAA_number_of_pixels','z_VZA_number_of_pixels','zz_VAA_number_of_pixels']].merge(df_stazione[['time','LST']], left_on='time',
                  right_on='time')

df_comp.sort_values(by='time', inplace = True)             
df_comp['Delta_LST'] = df_comp['f_LSTmean'] - df_comp['LST']

#Convert 'time' columns to datetime objects  and Create a new column with year-month-day 

df_comp['time'] = pd.to_datetime(df_comp['time'])
df_comp['valid_time'] = df_comp['time'].dt.strftime('%Y-%m-%d')
df_comp['valid_time'] = pd.to_datetime(df_comp['valid_time'], format='%Y-%m-%d')


# Merge DataFrames based on the 'valid_time' column

df_comparison = df_comp[['valid_time', 'time', 'a_image_id', 'f_LSTmean', 'LST', 'Delta_LST', 'g_LSTmedian', 'h_LSTstd', 'i_cloud_cover', 'j_roll', 'k_SZA_calc', 'l_SZA_mean', 'm_SZA_median', 'n_SZA_std', 'o_SAA_mean', 'p_SAA_median', 'q_SAA_std', 'r_VZA_mean','s_VZA_median','t_VZA_std', 'u_VAA_mean', 'v_VAA_median', 'w_VAA_std','x_SZA_number_of_pixels','y_SAA_number_of_pixels','z_VZA_number_of_pixels','zz_VAA_number_of_pixels','e_number_of_pixels']].merge(filtered_df_tcwv[['valid_time','latitude', 'longitude', 'number', 'step', 'surface', 'tcwv']], left_on='valid_time',
                  right_on='valid_time')

# Drop the column if not needed in the final result
df_comparison = df_comparison.drop(columns=['latitude', 'longitude', 'number', 'step', 'surface'])

# Create a dictionary to map the old column names to new names
column_rename_dict = {
    'a_image_id': 'Landsat8_image_id',
    'f_LSTmean': 'Landsat8_f_LSTmean',
    'LST': 'Landsat8_GBOV_LST',
    'Delta_LST': 'Landsat8_Delta_LST',
    'g_LSTmedian': 'Landsat8_g_LSTmedian',
    'h_LSTstd': 'Landsat8_h_LSTstd',
    'i_cloud_cover': 'Landsat8_cloud_cover',
    'j_roll': 'Landsat8_j_roll',
    'k_SZA_calc': 'Landsat8_k_SZA_calc',
    'l_SZA_mean': 'Landsat8_l_SZA_mean',
    'm_SZA_median': 'Landsat8_m_SZA_median',
    'n_SZA_std': 'Landsat8_n_SZA_std',
    'o_SAA_mean': 'Landsat8_o_SAA_mean',
    'p_SAA_median' : 'Landsat8_p_SAA_median',
    'q_SAA_std': 'Landsat8_q_SAA_std',
    'r_VZA_mean': 'Landsat8_r_VZA_mean',
    's_VZA_median': 'Landsat8_s_VZA_median',
    't_VZA_std': 'Landsat8_t_VZA_std',
    'u_VAA_mean': 'Landsat8_u_VAA_mean',
    'v_VAA_median': 'Landsat8_v_VAA_median',
    'w_VAA_std': 'Landsat8_w_VAA_std',
    'x_SZA_number_of_pixels': 'Landsat8_x_SZA_number_of_pixels',
    'y_SAA_number_of_pixels': 'Landsat8_y_SAA_number_of_pixels',
    'z_VZA_number_of_pixels': 'Landsat8_z_VZA_number_of_pixels',
    'zz_VAA_number_of_pixels': 'Landsat8_zz_VAA_number_of_pixels',
    'e_number_of_pixels': 'Landsat8_e_number_of_pixels',
    'tcwv': 'Landsat8_tcwv',
    'valid_time': 'overpass',
    'time': 'Landsat8_time'
}

# Rename the columns
df_comparison = df_comparison.rename(columns=column_rename_dict)


# Filtra il DataFrame per i_cloud_cover <= x
df_comparison = df_comparison[df_comparison['Landsat8_cloud_cover'] <= 50]

df_comparison.sort_values(by='Landsat8_time', inplace = True)  

#print('Landsat tabella finale', df_comparison.head(5))

df_comp1 = df_modis[['time', 'a_image_id', 'e_number_of_pixels', 'f_LSTmean', 'g_LSTmedian', 'h_LSTstd', 'i_local_solar_time', 'j_VA']].merge(df_stazione[['time','LST']], left_on='time',
                  right_on='time')

df_comp1.sort_values(by='time', inplace = True)             
df_comp1['Delta_LST'] = df_comp1['f_LSTmean'] - df_comp1['LST']

#Convert 'time' columns to datetime objects  and Create a new column with year-month-day 

df_comp1['time'] = pd.to_datetime(df_comp1['time'])
df_comp1['valid_time'] = df_comp1['time'].dt.strftime('%Y-%m-%d')
df_comp1['valid_time'] = pd.to_datetime(df_comp1['valid_time'], format='%Y-%m-%d')


# Merge DataFrames based on the 'valid_time' column

df_comparison1 = df_comp1[['valid_time', 'time', 'a_image_id', 'e_number_of_pixels', 'f_LSTmean',  'LST', 'Delta_LST','g_LSTmedian', 'h_LSTstd', 'i_local_solar_time', 'j_VA']].merge(filtered_df_tcwv[['valid_time','latitude', 'longitude', 'number', 'step', 'surface', 'tcwv']], left_on='valid_time',
                  right_on='valid_time')

# Drop the column if not needed in the final result
df_comparison1 = df_comparison1.drop(columns=['latitude', 'longitude', 'number', 'step', 'surface'])

# Create a dictionary to map the old column names to new names
column_rename_dict1 = {
    'a_image_id': 'Modis_image_id',
    'f_LSTmean': 'Modis_f_LSTmean',
    'LST': 'Modis_GBOV_LST',
    'Delta_LST': 'Modis_Delta_LST',
    'e_number_of_pixels': 'Modis_e_number_of_pixels',
    'g_LSTmedian': 'Modis_g_LSTmedian',
    'h_LSTstd': 'Modis_h_LSTstd',
    'i_local_solar_time': 'Modis_i_local_solar_time',
    'j_VA': 'Modis_j_VA',
    'tcwv': 'Modis_tcwv',
    'valid_time': 'overpass',
    'time': 'Modis_time'    
}

# Rename the columns
df_comparison1 = df_comparison1.rename(columns=column_rename_dict1)

df_comparison1.sort_values(by='Modis_time', inplace = True)  

#print('Modis tabella finale', df_comparison1.head(5))

#pdb.set_trace()
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!pausa
#FINAL MERGE IN UNIQUE TABLE ALL COLUMNS FROM 2 SENSORS



df_comparison2 = df_comparison[['overpass', 'Landsat8_time', 'Landsat8_image_id', 'Landsat8_f_LSTmean', 'Landsat8_GBOV_LST', 'Landsat8_Delta_LST', 'Landsat8_g_LSTmedian',
'Landsat8_h_LSTstd', 'Landsat8_cloud_cover', 'Landsat8_j_roll', 'Landsat8_k_SZA_calc', 'Landsat8_tcwv', 'Landsat8_l_SZA_mean', 'Landsat8_m_SZA_median', 'Landsat8_n_SZA_std',
'Landsat8_o_SAA_mean', 'Landsat8_p_SAA_median', 'Landsat8_q_SAA_std', 'Landsat8_r_VZA_mean','Landsat8_s_VZA_median','Landsat8_t_VZA_std', 'Landsat8_u_VAA_mean',
'Landsat8_v_VAA_median', 'Landsat8_w_VAA_std','Landsat8_x_SZA_number_of_pixels','Landsat8_y_SAA_number_of_pixels','Landsat8_z_VZA_number_of_pixels',
'Landsat8_zz_VAA_number_of_pixels','Landsat8_e_number_of_pixels']].merge(df_comparison1[['overpass', 'Modis_time', 'Modis_image_id', 'Modis_e_number_of_pixels', 'Modis_f_LSTmean', 'Modis_GBOV_LST', 'Modis_Delta_LST','Modis_g_LSTmedian',
'Modis_h_LSTstd', 'Modis_i_local_solar_time', 'Modis_j_VA', 'Modis_tcwv']], left_on='overpass', right_on='overpass')

df_comparison2.sort_values(by='overpass', inplace = True)
#print('cross satellite tabella finale', df_comparison2.head(5))
#pdb.set_trace()


years = pd.DatetimeIndex(df_comparison2['overpass']).year.to_numpy()
years = np.unique(years, return_index=False)
#print('years', years)
date_form = DateFormatter("%d-%m-%y")

# years1 = pd.DatetimeIndex(df_comparison2['Modis_time']).year.to_numpy()
# years1 = np.unique(years, return_index=False)
# #print('years', years)
# date_form1 = DateFormatter("%d-%m-%y")



def compute_stats (data_frame, info_periodo=''):#, start_date, end_date):
            media = data_frame['Landsat8_Delta_LST'].mean(axis=0)
            std =   data_frame['Landsat8_Delta_LST'].std(axis=0)
            mediana = data_frame['Landsat8_Delta_LST'].median(axis=0)
            nmad = scipy.stats.median_abs_deviation(data_frame['Landsat8_Delta_LST'], scale='normal')
            delta_lst_buffer = round(data_frame['Landsat8_e_number_of_pixels'].mean(axis=0))
            tcwv_mean = data_frame['Landsat8_tcwv'].mean(axis=0)
            tcwv_std = data_frame['Landsat8_tcwv'].std(axis=0)
            tcwv_median = data_frame['Landsat8_tcwv'].median(axis=0)
            tcwv_nmad = scipy.stats.median_abs_deviation(data_frame['Landsat8_tcwv'], scale='normal')
            sza_mean = data_frame['Landsat8_l_SZA_mean'].mean(axis=0)
            sza_std = data_frame['Landsat8_l_SZA_mean'].std(axis=0)
            sza_median = data_frame['Landsat8_l_SZA_mean'].median(axis=0)
            sza_nmad = scipy.stats.median_abs_deviation(data_frame['Landsat8_l_SZA_mean'], scale='normal')
            sza_buffer = round(data_frame['Landsat8_x_SZA_number_of_pixels'].mean(axis=0))
            saa_mean = data_frame['Landsat8_o_SAA_mean'].mean(axis=0)
            saa_std = data_frame['Landsat8_o_SAA_mean'].std(axis=0)
            saa_median = data_frame['Landsat8_o_SAA_mean'].median(axis=0)
            saa_nmad = scipy.stats.median_abs_deviation(data_frame['Landsat8_o_SAA_mean'], scale='normal')
            saa_buffer = round(data_frame['Landsat8_y_SAA_number_of_pixels'].mean(axis=0))
            vza_mean = data_frame['Landsat8_r_VZA_mean'].mean(axis=0)
            vza_std = data_frame['Landsat8_r_VZA_mean'].std(axis=0)
            vza_median = data_frame['Landsat8_r_VZA_mean'].median(axis=0)
            vza_nmad = scipy.stats.median_abs_deviation(data_frame['Landsat8_r_VZA_mean'], scale='normal')
            vza_buffer = round(data_frame['Landsat8_z_VZA_number_of_pixels'].mean(axis=0))
            vaa_mean = data_frame['Landsat8_u_VAA_mean'].mean(axis=0)
            vaa_std = data_frame['Landsat8_u_VAA_mean'].std(axis=0)
            vaa_median = data_frame['Landsat8_u_VAA_mean'].median(axis=0)
            vaa_nmad = scipy.stats.median_abs_deviation(data_frame['Landsat8_u_VAA_mean'], scale='normal')
            vaa_buffer = round(data_frame['Landsat8_zz_VAA_number_of_pixels'].mean(axis=0))
            cc_mean = data_frame['Landsat8_cloud_cover'].mean(axis=0)
            cc_std = data_frame['Landsat8_cloud_cover'].std(axis=0)
            cc_median = data_frame['Landsat8_cloud_cover'].median(axis=0)
            cc_nmad = scipy.stats.median_abs_deviation(data_frame['Landsat8_cloud_cover'], scale='normal')
            #
            media1 = data_frame['Modis_Delta_LST'].mean(axis=0)
            std1 =   data_frame['Modis_Delta_LST'].std(axis=0)
            mediana1 = data_frame['Modis_Delta_LST'].median(axis=0)
            nmad1 = scipy.stats.median_abs_deviation(data_frame['Modis_Delta_LST'], scale='normal')
            delta_lst_buffer1 = round(data_frame['Modis_e_number_of_pixels'].mean(axis=0))
            tcwv_mean1 = data_frame['Modis_tcwv'].mean(axis=0)
            tcwv_std1 = data_frame['Modis_tcwv'].std(axis=0)
            tcwv_median1 = data_frame['Modis_tcwv'].median(axis=0)
            tcwv_nmad1 = scipy.stats.median_abs_deviation(data_frame['Modis_tcwv'], scale='normal')
            vza_mean1 = data_frame['Modis_j_VA'].mean(axis=0)
            vza_std1= data_frame['Modis_j_VA'].std(axis=0)
            vza_median1 = data_frame['Modis_j_VA'].median(axis=0)
            vza_nmad1 = scipy.stats.median_abs_deviation(data_frame['Modis_j_VA'], scale='normal')
            start_date = data_frame['overpass'].iloc[0].strftime("%Y-%m-%d")
            end_date = data_frame['overpass'].iloc[-1].strftime("%Y-%m-%d")
            
            print('------- Landsat8_period analyzed -------')
            print(start_date, '--',end_date )
            print('rows compared:', len(data_frame.index))
            print('media delta LST [°]:', format(media,'.2f'))
            print('std delta LST [°]:', format(std,'.2f'))
            print('median delta LST [°]:',format(mediana,'.2f'))
            print('nmad delta LST [°]:', format(nmad,'.2f'))
            print('buffer delta LST:', delta_lst_buffer)
            print('tcwv mean [kg m-2]:', format(tcwv_mean,'.2f'))
            print('tcwv std [kg m-2]:', format(tcwv_std,'.2f'))
            print('tcwv median [kg m-2]:', format(tcwv_median,'.2f'))
            print('tcwv nmad [kg m-2]:', format(tcwv_nmad,'.2f'))
            print('SZA mean[°]:', format(sza_mean,'.2f'))
            print('SZA std [°]:', format(sza_std,'.2f'))
            print('SZA median[°]:', format(sza_median,'.2f'))
            print('SZA nmad [°]:', format(sza_nmad,'.2f'))
            print('buffer SZA:', sza_buffer)
            print('SAA mean[°]:', format(saa_mean,'.2f'))
            print('SAA std [°]:', format(saa_std,'.2f'))
            print('SAA median[°]:', format(saa_median,'.2f'))
            print('SAA nmad [°]:', format(saa_nmad,'.2f'))
            print('buffer SAA:', saa_buffer)
            print('VZA mean[°]:', format(vza_mean,'.2f'))
            print('VZA std [°]:', format(vza_std,'.2f'))
            print('VZA median[°]:', format(vza_median,'.2f'))
            print('VZA nmad [°]:', format(vza_nmad,'.2f'))
            print('buffer VZA:', vza_buffer)
            print('VAA mean[°]:', format(vaa_mean,'.2f'))
            print('VAA std [°]:', format(vaa_std,'.2f'))
            print('VAA median[°]:', format(vaa_median,'.2f'))
            print('VAA nmad [°]:', format(vaa_nmad,'.2f'))
            print('buffer VAA:', vaa_buffer)
            print('cc mean [%]:', format(cc_mean,'.2f'))
            print('cc std [%]:', format(cc_std,'.2f'))
            print('cc median [%]:', format(cc_median,'.2f'))
            print('cc nmad [%]:', format(cc_nmad,'.2f'))
           
            print('-------------------------------')
            
            print('-------Modis period analyzed -------')
            print(start_date, '--',end_date )
            print('rows compared:', len(data_frame.index))
            print('media delta LST [°]:', format(media1,'.2f'))
            print('std delta LST [°]:', format(std1,'.2f'))
            print('median delta LST [°]:',format(mediana1,'.2f'))
            print('nmad delta LST [°]:', format(nmad1,'.2f'))
            print('buffer delta LST:', delta_lst_buffer1)
            print('tcwv mean [kg m-2]:', format(tcwv_mean1,'.2f'))
            print('tcwv std [kg m-2]:', format(tcwv_std1,'.2f'))
            print('tcwv median [kg m-2]:', format(tcwv_median1,'.2f'))
            print('tcwv nmad [kg m-2]:', format(tcwv_nmad1,'.2f'))
            print('VZA mean[°]:', format(vza_mean1,'.2f'))
            print('VZA std [°]:', format(vza_std1,'.2f'))
            print('VZA median[°]:', format(vza_median1,'.2f'))
            print('VZA nmad [°]:', format(vza_nmad1,'.2f'))
                       
            print('-------------------------------')
            
            stats = [data_frame['overpass'].iloc[0], data_frame['overpass'].iloc[-1],len(data_frame.index), data_frame['Landsat8_time'].iloc[0], data_frame['Landsat8_time'].iloc[-1], 
            media, std, mediana, nmad, delta_lst_buffer, tcwv_mean, tcwv_std, tcwv_median, tcwv_nmad, sza_mean, sza_std, sza_median, sza_nmad, sza_buffer, 
            saa_mean, saa_std, saa_median, saa_nmad, saa_buffer, vza_mean, vza_std, vza_median, vza_nmad, vza_buffer, vaa_mean, vaa_std, vaa_median, vaa_nmad, vaa_buffer, 
            cc_mean, cc_std, cc_median, cc_nmad, data_frame['Modis_time'].iloc[0], data_frame['Modis_time'].iloc[-1], media1, std1, mediana1, nmad1, delta_lst_buffer1, 
            tcwv_mean1, tcwv_std1, tcwv_median1, tcwv_nmad1, vza_mean1, vza_std1, vza_median1, vza_nmad1]


            fig=plt.figure(figsize=(15,15))
            plt.title('Landsat8 and Modis LST comparison @' +nome_stazione+ '- '+ start_date + ' - '+ end_date +' '+ str(delta_lst_buffer) + ' px Landsat8 Buffer vs '+ str(delta_lst_buffer1) + ' px Modis Buffer')            
            ax = fig.gca()
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            date_form = DateFormatter('%Y-%m')
            plt.plot(data_frame['overpass'],data_frame['Landsat8_f_LSTmean'], 'bo--', label='Landsat_8')
            plt.plot(data_frame['overpass'],data_frame['Modis_f_LSTmean'], 'ro--', label='Modis_Day')
            plt.grid()
            plt.ylabel('LST [°]')
            plt.xlabel('Time')
            plt.legend(loc='upper right')
            ax.xaxis.set_major_formatter(date_form) 
            plt.tight_layout()
            plt.savefig(output_path+'Landsat8_Modis_LST_comparison_at_' +nome_stazione+'_'+info_periodo+'_'+ data_frame['Landsat8_time'].iloc[0].strftime("%Y-%m-%d") + '_'+ data_frame['Landsat8_time'].iloc[-1].strftime("%Y-%m-%d")+'.pdf')                      
                        
            fig=plt.figure(figsize=(15,15))
            plt.title('Landsat8 and Modis_Day Delta_LST comparison @' +nome_stazione+ '- '+ start_date + ' - '+ end_date +' '+ str(delta_lst_buffer) + ' px Landsat8 Buffer vs '+ str(delta_lst_buffer1) + ' px Modis Buffer')            
            ax = fig.gca()
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            date_form = DateFormatter('%Y-%m')
            plt.plot(data_frame['overpass'],data_frame['Landsat8_Delta_LST'], 'bo--', label='Landsat_8')
            plt.plot(data_frame['overpass'],data_frame['Modis_Delta_LST'], 'ro--', label='Modis_Day')
            plt.grid()
            plt.ylabel('Delta_LST [°]')
            plt.xlabel('Time')
            plt.legend(loc='upper right')
            ax.xaxis.set_major_formatter(date_form) 
            plt.tight_layout()
            plt.savefig(output_path+'Cross_satellite_Delta_LST_comparison_at_' +nome_stazione+'_'+info_periodo+'_'+ data_frame['overpass'].iloc[0].strftime("%Y-%m-%d") + '_'+ data_frame['overpass'].iloc[-1].strftime("%Y-%m-%d")+'.pdf')
            
            fig=plt.figure(figsize=(15,15))
            plt.title('Landsat8 and Modis_Day View Zenith Angle comparison @' +nome_stazione+ '- '+ start_date + ' - '+ end_date +' '+ str(vza_buffer) + ' px Landsat8 VZA Buffer vs 1 px Modis VZA Buffer')            
            ax = fig.gca()
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            date_form = DateFormatter('%Y-%m')
            plt.plot(data_frame['overpass'],data_frame['Landsat8_r_VZA_mean'], 'bo--', label='Landsat_8')
            plt.plot(data_frame['overpass'],data_frame['Modis_j_VA'], 'ro--', label='Modis_Day')
            plt.grid()
            plt.ylabel('View Zenith Angle [°]')
            plt.xlabel('Time')
            plt.legend(loc='upper right')
            ax.xaxis.set_major_formatter(date_form) 
            plt.tight_layout()
            plt.savefig(output_path+'Cross_satellite_VZA_comparison_at_' +nome_stazione+'_'+info_periodo+'_'+ data_frame['overpass'].iloc[0].strftime("%Y-%m-%d") + '_'+ data_frame['overpass'].iloc[-1].strftime("%Y-%m-%d")+'.pdf')
            
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
df_comparison2['season'] = df_comparison2['overpass'].apply(get_season)

stats = []

print('overall all years')
df_comparison_all = df_comparison2.copy()
all_start_date = df_comparison_all['overpass'].iloc[0].strftime("%Y-%m-%d")
all_end_date = df_comparison_all['overpass'].iloc[-1].strftime("%Y-%m-%d")
result1 = compute_stats(df_comparison_all, 'overall_all_years')
result1.append('overall all years')
stats.append(result1)

df_comparison_all.to_excel(output_path+'cross_satellite_overall_all_years_'+all_start_date+'_'+all_end_date+'.xlsx',index=False)
#pdb.set_trace()

#stats_winter = []
print('winter all years')
mask_winter = df_comparison_all['season'] == 'winter'
df_comparison_all_winter = df_comparison_all[mask_winter]
all_winter_start_date = df_comparison_all_winter['overpass'].iloc[0].strftime("%Y-%m-%d")
all_winter_end_date =   df_comparison_all_winter['overpass'].iloc[-1].strftime("%Y-%m-%d")
result2 = compute_stats(df_comparison_all_winter, 'winter_all_years')
result2.append('winter all years')
stats.append(result2)
df_comparison_all_winter.to_excel(output_path+'cross_satellite_winter_all_years_'+all_winter_start_date+'_'+all_winter_end_date+'.xlsx',index=False)

#stats_spring = []
print('spring all years')
mask_spring = df_comparison_all['season'] == 'spring'
df_comparison_all_spring = df_comparison_all[mask_spring]
all_spring_start_date = df_comparison_all_spring['overpass'].iloc[0].strftime("%Y-%m-%d")
all_spring_end_date =   df_comparison_all_spring['overpass'].iloc[-1].strftime("%Y-%m-%d")
result5 = compute_stats(df_comparison_all_spring, 'spring_all_years')
result5.append('spring all years')
stats.append(result5)
df_comparison_all_spring.to_excel(output_path+'cross_satellite_spring_all_years_'+all_spring_start_date+'_'+all_spring_end_date+'.xlsx',index=False)

#stats_summer = []
print('summer all years')
mask_summer = df_comparison_all['season'] == 'summer'
df_comparison_all_summer = df_comparison_all[mask_summer]
all_summer_start_date = df_comparison_all_summer['overpass'].iloc[0].strftime("%Y-%m-%d")
all_summer_end_date =   df_comparison_all_summer['overpass'].iloc[-1].strftime("%Y-%m-%d")
result6 = compute_stats(df_comparison_all_summer, 'summer_all_years')
result6.append('summer all years')
stats.append(result6)
df_comparison_all_summer.to_excel(output_path+'cross_satellite_summer_all_years_'+all_summer_start_date+'_'+all_summer_end_date+'.xlsx',index=False)

#stats_autumn = []
print('autumn all years')
mask_autumn = df_comparison_all['season'] == 'autumn'
df_comparison_all_autumn = df_comparison_all[mask_autumn]
all_autumn_start_date = df_comparison_all_autumn['overpass'].iloc[0].strftime("%Y-%m-%d")
all_autumn_end_date =   df_comparison_all_autumn['overpass'].iloc[-1].strftime("%Y-%m-%d")
result7 = compute_stats(df_comparison_all_autumn, 'autumn_all_years')
result7.append('autumn all years')
stats.append(result7)
df_comparison_all_autumn.to_excel(output_path+'cross_satellite_autumn_all_years_'+all_autumn_start_date+'_'+all_autumn_end_date+'.xlsx',index=False)

for year in years:
            print('Year:', year)
            start_date = str(year)+'-01-01'
            end_date = str(year)+'-12-31'
            mask_time = (df_comparison_all['overpass'] >= start_date) & (df_comparison_all['overpass'] <= end_date)
            df_comparison2 = df_comparison_all.loc[mask_time]

            print('overall year ', year)
            print('df_comparison\n', df_comparison2.head(5))
            print ('rows gbov', len(df_stazione.index))
            print ('rows landsat', len(df_landsat.index))
            print ('rows modis', len(df_modis.index))
            print('rows df_comparison', len(df_comparison2.index))
            result3 = compute_stats(df_comparison2, str(year)+'_overall')
            result3.append(str(year)+'_overall')
            stats.append(result3)
            df_comparison2.to_excel(output_path+'cross_satellite_'+str(year)+'_overall_'+start_date+'_'+end_date+'.xlsx',index=False)
            
            print('winter year', year)
            mask_winter_year = df_comparison2['season'] == 'winter'
            df_winter_year = df_comparison2[ mask_winter_year]
            
            if len(df_winter_year.index) != 0:
                result4 = compute_stats(df_winter_year, str(year)+'_winter')
                result4.append(str(year)+'_winter')
                stats.append(result4)
                year_winter_start_date = df_winter_year['overpass'].iloc[0].strftime("%Y-%m-%d")
                year_winter_end_date =   df_winter_year['overpass'].iloc[-1].strftime("%Y-%m-%d")
                df_winter_year.to_excel(output_path+'cross_satellite_'+str(year)+'_winter_'+year_winter_start_date+'_'+year_winter_end_date+'.xlsx',index=False)
            
            print('spring year', year)
            mask_spring_year = df_comparison2['season'] == 'spring'
            df_spring_year = df_comparison2[ mask_spring_year]
            
            if len(df_spring_year.index) != 0:
                result8 = compute_stats(df_spring_year, str(year)+'_spring')
                result8.append(str(year)+'_spring')
                stats.append(result8)
                year_spring_start_date = df_spring_year['overpass'].iloc[0].strftime("%Y-%m-%d")
                year_spring_end_date =   df_spring_year['overpass'].iloc[-1].strftime("%Y-%m-%d")
                df_spring_year.to_excel(output_path+'cross_satellite_'+str(year)+'_spring_'+year_spring_start_date+'_'+year_spring_end_date+'.xlsx',index=False)
                
            print('summer year', year)
            mask_summer_year = df_comparison2['season'] == 'summer'
            df_summer_year = df_comparison2[ mask_summer_year]
            
            if len(df_summer_year.index) != 0:
                result9 = compute_stats(df_summer_year, str(year)+'_summer')
                result9.append( str(year)+'_summer')
                stats.append(result9)
                year_summer_start_date = df_summer_year['overpass'].iloc[0].strftime("%Y-%m-%d")
                year_summer_end_date =   df_summer_year['overpass'].iloc[-1].strftime("%Y-%m-%d")
                df_summer_year.to_excel(output_path+'cross_satellite_'+str(year)+'_summer_'+year_summer_start_date+'_'+year_summer_end_date+'.xlsx',index=False)
                
            print('autumn year', year)
            mask_autumn_year = df_comparison2['season'] == 'autumn'
            df_autumn_year = df_comparison2[ mask_autumn_year]
            
            if len(df_autumn_year.index) != 0:
                result10 = compute_stats(df_autumn_year, str(year)+'_autumn')
                result10.append(str(year)+'_autumn')
                stats.append(result10)
                year_autumn_start_date = df_autumn_year['overpass'].iloc[0].strftime("%Y-%m-%d")
                year_autumn_end_date =   df_autumn_year['overpass'].iloc[-1].strftime("%Y-%m-%d")
                df_autumn_year.to_excel(output_path+'cross_satellite_'+str(year)+'_autumn_'+year_autumn_start_date+'_'+year_autumn_end_date+'.xlsx',index=False)
            
            
# Assuming stats is a list of data in the following order: [start_date, end_date, number_of_rows, mean, std, median, NMAD, tcwv_mean, tcwv_median, SAZ_mean, SAZ_median, cc_mean, cc_median, season]
stats = np.array(stats, dtype=object)

# Define column names including "season"
column_names = ['start_date', 'end_date', 'number_of_rows', 'Landsat8_start_date', 'Landsat8_end_date', 'Landsat8_delta_LST_mean', 'Landsat8_delta_LST_std',
'Landsat8_delta_LST_median', 'Landsat8_delta_LST_NMAD','Landsat8_LST_buffer', 'Landsat8_tcwv_mean', 'Landsat8_tcwv_std', 'Landsat8_tcwv_median', 'Landsat8_tcwv_NMAD',
'Landsat8_SZA_mean', 'Landsat8_SZA_std', 'Landsat8_SZA_median', 'Landsat8_SZA_NMAD', 'Landsat8_SZA_buffer', 
'Landsat8_SAA_mean', 'Landsat8_SAA_std', 'Landsat8_SAA_median', 'Landsat8_SAA_NMAD', 'Landsat8_SAA_buffer',
'Landsat8_VZA_mean', 'Landsat8_VZA_std', 'Landsat8_VZA_median', 'Landsat8_VZA_NMAD', 'Landsat8_VZA_buffer',
'Landsat8_VAA_mean', 'Landsat8_VAA_std', 'Landsat8_VAA_median', 'Landsat8_VAA_NMAD', 'Landsat8_VAA_buffer',
'Landsat8_CC_mean', 'Landsat8_CC_std', 'Landsat8_CC_median', 'Landsat8_CC_NMAD', 
'Modis_start_date', 'Modis_end_date', 'Modis_delta_LST_mean', 'Modis_delta_LST_std', 'Modis_delta_LST_median', 'Modis_delta_LST_NMAD', 
'Modis_LST_buffer', 'Modis_tcwv_mean', 'Modis_tcwv_std', 'Modis_tcwv_median', 'Modis_tcwv_NMAD', 
'Modis_VZA_mean', 'Modis_VZA_std', 'Modis_VZA_median', 'Modis_VZA_NMAD', 'season']

# Create a pandas DataFrame
df = pd.DataFrame(stats, columns=column_names)

# Specify the Excel file path
excel_filename = output_path + 'SEASON_COMBINED_STATISTICS_buffer_1100_vs_01_' + nome_stazione + '.xlsx'

# Save the DataFrame to an Excel file
df.to_excel(excel_filename, index=False)
#plt.show()
