# GEE data extraction script
# https://code.earthengine.google.com/df72bb16282e94a844acba2ba8d7f5a4

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, MonthLocator
import scipy 
from datetime import date, datetime
import pdb
plt.close('all')

path = os.path.dirname(os.path.abspath(__file__))+'//'

nome_stazione = 'USA_BOND'
output_path = path +'landsat8_buffer_1100_pixels_'+nome_stazione+'//'
isExist = os.path.exists(output_path)
if not isExist:

  # Create a new directory 
  os.makedirs(output_path)
print("The new directory is created!")



df_landsat =pd.read_csv(path+"GEE//"+nome_stazione+'_L8_buffer_1100'+'.csv',sep=',')

df_stazione = pd.read_csv(path+"GBOV//"+'GBOV_RM09_Bondville_w12_20130101T000100Z_20221231T235900Z_046_ACR_V2.0'+'.csv', sep=';')

df_tcwv =pd.read_csv(path+"VAPOUR//"+nome_stazione+'_VAPOUR'+'.csv',sep=',')

df_landsat = df_landsat.dropna()
df_stazione = df_stazione.dropna()

#https://stackoverflow.com/questions/62917882/convert-datetime64ns-utc-pandas-column-to-datetime
df_stazione['TIME_IS'] =  pd.to_datetime(df_stazione['TIME_IS']).dt.tz_localize(None)#, format='%d%b%Y:%H:%M:%S.%f')

# https://stephenallwright.com/pandas-round-datetime-to-minute/
df_landsat['b_system_time_start']= pd.to_datetime(df_landsat['b_system_time_start']).dt.round("min")

df_stazione.rename(columns={'TIME_IS':'time'}, inplace=True)
df_landsat.rename(columns={'b_system_time_start':'time'}, inplace=True)

#Filter out anomalies from GBOV dataset
df_stazione = df_stazione[df_stazione['LST_STD'] <= 1.5] 

# Convert the 'time' column to datetime
df_tcwv['valid_time'] = pd.to_datetime(df_tcwv['valid_time'])


desired_time = "16:00:00"  # Desired time as a string
latitude = 40.1            
longitude = -88.4

# Use loc to filter the DataFrame based on conditions
filtered_df_tcwv = df_tcwv.loc[
    (df_tcwv['valid_time'].dt.strftime('%H:%M:%S') == desired_time)
    &
    (df_tcwv['latitude'].astype(float) == latitude) &
    (df_tcwv['longitude'].astype(float) == longitude)
]

filtered_df_tcwv.loc[:, 'valid_time'] = pd.to_datetime(filtered_df_tcwv['valid_time']).dt.date


#print('t_tcwv\n',filtered_df_tcwv['valid_time'].head(5))
#print('t_station\n',df_stazione['time'].head(5))
#print('t_landsat\n',df_landsat['time'].head(5))
#pdb.set_trace()
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
df_comparison = df_comparison.drop(columns=['valid_time', 'latitude', 'longitude', 'number', 'step', 'surface'])

# Filtra il DataFrame per i_cloud_cover <= x
df_comparison = df_comparison[df_comparison['i_cloud_cover'] <= 50]

df_comparison.sort_values(by='time', inplace = True)  

#print('tabella finale', df_comparison.head(5))


#pdb.set_trace()
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!pausa

years = pd.DatetimeIndex(df_comparison['time']).year.to_numpy()
years = np.unique(years, return_index=False)
#print('years', years)
date_form = DateFormatter("%d-%m-%y")



def compute_stats (data_frame, info_periodo=''):#, start_date, end_date):
            media = data_frame['Delta_LST'].mean(axis=0)
            std =   data_frame['Delta_LST'].std(axis=0)
            mediana = data_frame['Delta_LST'].median(axis=0)
            nmad = scipy.stats.median_abs_deviation(data_frame['Delta_LST'], scale='normal')
            delta_lst_buffer = round(data_frame['e_number_of_pixels'].mean(axis=0))
            tcwv_mean = data_frame['tcwv'].mean(axis=0)
            tcwv_std = data_frame['tcwv'].std(axis=0)
            tcwv_median = data_frame['tcwv'].median(axis=0)
            tcwv_nmad = scipy.stats.median_abs_deviation(data_frame['tcwv'], scale='normal')
            sza_mean = data_frame['l_SZA_mean'].mean(axis=0)
            sza_std = data_frame['l_SZA_mean'].std(axis=0)
            sza_median = data_frame['l_SZA_mean'].median(axis=0)
            sza_nmad = scipy.stats.median_abs_deviation(data_frame['l_SZA_mean'], scale='normal')
            sza_buffer = round(data_frame['x_SZA_number_of_pixels'].mean(axis=0))
            saa_mean = data_frame['o_SAA_mean'].mean(axis=0)
            saa_std = data_frame['o_SAA_mean'].std(axis=0)
            saa_median = data_frame['o_SAA_mean'].median(axis=0)
            saa_nmad = scipy.stats.median_abs_deviation(data_frame['o_SAA_mean'], scale='normal')
            saa_buffer = round(data_frame['y_SAA_number_of_pixels'].mean(axis=0))
            vza_mean = data_frame['r_VZA_mean'].mean(axis=0)
            vza_std = data_frame['r_VZA_mean'].std(axis=0)
            vza_median = data_frame['r_VZA_mean'].median(axis=0)
            vza_nmad = scipy.stats.median_abs_deviation(data_frame['r_VZA_mean'], scale='normal')
            vza_buffer = round(data_frame['z_VZA_number_of_pixels'].mean(axis=0))
            vaa_mean = data_frame['u_VAA_mean'].mean(axis=0)
            vaa_std = data_frame['u_VAA_mean'].std(axis=0)
            vaa_median = data_frame['u_VAA_mean'].median(axis=0)
            vaa_nmad = scipy.stats.median_abs_deviation(data_frame['u_VAA_mean'], scale='normal')
            vaa_buffer = round(data_frame['zz_VAA_number_of_pixels'].mean(axis=0))
            cc_mean = data_frame['i_cloud_cover'].mean(axis=0)
            cc_std = data_frame['i_cloud_cover'].std(axis=0)
            cc_median = data_frame['i_cloud_cover'].median(axis=0)
            cc_nmad = scipy.stats.median_abs_deviation(data_frame['i_cloud_cover'], scale='normal')
            start_date = data_frame['time'].iloc[0].strftime("%Y-%m-%d %H:%M:%S")
            end_date = data_frame['time'].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
            
            print('------- period analyzed -------')
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
            stats = [data_frame['time'].iloc[0], data_frame['time'].iloc[-1],len(data_frame.index), media, std, mediana, nmad, delta_lst_buffer, tcwv_mean, tcwv_std, tcwv_median, tcwv_nmad, sza_mean, sza_std, sza_median, sza_nmad, sza_buffer, saa_mean, saa_std, saa_median, saa_nmad, saa_buffer, vza_mean, vza_std, vza_median, vza_nmad, vza_buffer, vaa_mean, vaa_std, vaa_median, vaa_nmad, vaa_buffer, cc_mean, cc_std, cc_median, cc_nmad]


            fig=plt.figure(figsize=(15,15))
            plt.title('LST comparison @' +nome_stazione+ '- '+ start_date + ' - '+ end_date +' '+ str(delta_lst_buffer) + ' px Buffer')            
            ax = fig.gca()
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            date_form = DateFormatter('%Y-%m')
            plt.plot(data_frame['time'],data_frame['f_LSTmean'], 'bo--', label='Landsat_8')
            plt.plot(data_frame['time'],data_frame['LST'], 'ro--', label='Stazione')
            plt.grid()
            plt.ylabel('LST [°]')
            plt.xlabel('Time')
            plt.legend(loc='upper right')
            ax.xaxis.set_major_formatter(date_form) 
            plt.tight_layout()
            plt.savefig(output_path+'LST_comparison_at_' +nome_stazione+'_'+info_periodo+'_'+ data_frame['time'].iloc[0].strftime("%Y-%m-%d") + '_'+ data_frame['time'].iloc[-1].strftime("%Y-%m-%d")+'.pdf')
                            
            fig=plt.figure(figsize=(15,15))
            plt.title('Delta LST (LST Landsat 8 - LST stazione) @'+ nome_stazione+ '- '+ start_date + ' - '+ end_date+' '+ str(delta_lst_buffer) + ' px Buffer')
            ax = fig.gca()
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.plot(data_frame['time'],data_frame['Delta_LST'], 'ko--', label='Delta LST')
            plt.grid()
            plt.ylabel('Delta LST [°]')
            plt.xlabel('Time')
            plt.legend(loc='upper right')
            ax.xaxis.set_major_formatter(date_form)
            plt.tight_layout()
            # qui salvare 
            plt.savefig(output_path+'delta_LST_at_'+nome_stazione+'_'+info_periodo+'_'+ data_frame['time'].iloc[0].strftime("%Y-%m-%d") + '_'+ data_frame['time'].iloc[-1].strftime("%Y-%m-%d")+'.pdf')
            
            fig=plt.figure(figsize=(15,15))
            plt.title('Total Column Water Vapour @'+ nome_stazione+ '- '+ start_date + ' - '+ end_date+' ≈1100px Buffer')
            ax = fig.gca()
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.plot(data_frame['time'],data_frame['tcwv'], 'ko--', label='TCWV')
            plt.grid()
            plt.ylabel('TCWV [kg m-2]')
            plt.xlabel('Time')
            plt.legend(loc='upper right')
            ax.xaxis.set_major_formatter(date_form)
            plt.tight_layout()
            plt.savefig(output_path+'TCWV_at_'+nome_stazione+'_'+info_periodo+'_'+ data_frame['time'].iloc[0].strftime("%Y-%m-%d") + '_'+ data_frame['time'].iloc[-1].strftime("%Y-%m-%d")+'.pdf')
            
            fig=plt.figure(figsize=(15,15))
            plt.title('Cloud Cover @'+ nome_stazione+ '- '+ start_date + ' - '+ end_date+' ≈1100 px Buffer')
            ax = fig.gca()
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.plot(data_frame['time'],data_frame['i_cloud_cover'], 'ko--', label='Cloud_Cover')
            plt.grid()
            plt.ylabel('Cloud_Cover [%]')
            plt.xlabel('Time')
            plt.legend(loc='upper right')
            ax.xaxis.set_major_formatter(date_form)
            plt.tight_layout()
            plt.savefig(output_path+'CC_at_'+nome_stazione+'_'+info_periodo+'_'+ data_frame['time'].iloc[0].strftime("%Y-%m-%d") + '_'+ data_frame['time'].iloc[-1].strftime("%Y-%m-%d")+'.pdf')
            
            fig=plt.figure(figsize=(15,15))
            plt.title('Solar Zenith Angle @'+ nome_stazione+ '- '+ start_date + ' - '+ end_date +' '+ str(sza_buffer) + ' px Buffer')
            ax = fig.gca()
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.plot(data_frame['time'],data_frame['l_SZA_mean'], 'ko--', label='SZA')
            plt.grid()
            plt.ylabel('SZA [°]')
            plt.xlabel('Time')
            plt.legend(loc='upper right')
            ax.xaxis.set_major_formatter(date_form)
            plt.tight_layout()
            plt.savefig(output_path+'SZA_at_'+nome_stazione+'_'+info_periodo+'_'+ data_frame['time'].iloc[0].strftime("%Y-%m-%d") + '_'+ data_frame['time'].iloc[-1].strftime("%Y-%m-%d")+'.pdf')
            
            fig=plt.figure(figsize=(15,15))
            plt.title('Solar Azimuth Angle @'+ nome_stazione+ '- '+ start_date + ' - '+ end_date +' '+ str(saa_buffer) + ' px Buffer')
            ax = fig.gca()
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.plot(data_frame['time'],data_frame['o_SAA_mean'], 'ko--', label='SAA')
            plt.grid()
            plt.ylabel('SAA [°]')
            plt.xlabel('Time')
            plt.legend(loc='upper right')
            ax.xaxis.set_major_formatter(date_form)
            plt.tight_layout()
            plt.savefig(output_path+'SAA_at_'+nome_stazione+'_'+info_periodo+'_'+ data_frame['time'].iloc[0].strftime("%Y-%m-%d") + '_'+ data_frame['time'].iloc[-1].strftime("%Y-%m-%d")+'.pdf')
            
            fig=plt.figure(figsize=(15,15))
            plt.title('View Zenith Angle @'+ nome_stazione+ '- '+ start_date + ' - '+ end_date +' '+ str(vza_buffer) +' px Buffer')
            ax = fig.gca()
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.plot(data_frame['time'],data_frame['r_VZA_mean'], 'ko--', label='VZA')
            plt.grid()
            plt.ylabel('VZA [°]')
            plt.xlabel('Time')
            plt.legend(loc='upper right')
            ax.xaxis.set_major_formatter(date_form)
            plt.tight_layout()
            plt.savefig(output_path+'VZA_at_'+nome_stazione+'_'+info_periodo+'_'+ data_frame['time'].iloc[0].strftime("%Y-%m-%d") + '_'+ data_frame['time'].iloc[-1].strftime("%Y-%m-%d")+'.pdf')
            
            fig=plt.figure(figsize=(15,15))
            plt.title('View Azimuth Angle @'+ nome_stazione+ '- '+ start_date + ' - '+ end_date+' ' + str(vaa_buffer) + ' px Buffer')
            ax = fig.gca()
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.plot(data_frame['time'],data_frame['u_VAA_mean'], 'ko--', label='VAA')
            plt.grid()
            plt.ylabel('VAA [°]')
            plt.xlabel('Time')
            plt.legend(loc='upper right')
            ax.xaxis.set_major_formatter(date_form)
            plt.tight_layout()
            plt.savefig(output_path+'VAA_at_'+nome_stazione+'_'+info_periodo+'_'+ data_frame['time'].iloc[0].strftime("%Y-%m-%d") + '_'+ data_frame['time'].iloc[-1].strftime("%Y-%m-%d")+'.pdf')
            
            plt.figure(figsize=(10, 6))
            plt.title('Delta LST (LST landsat 8 - LST stazione) histograms@' +nome_stazione+ '- '+ start_date + ' - '+ end_date+' ' + str(delta_lst_buffer) + ' px Buffer')
            n, bins, patches = plt.hist(x=data_frame['Delta_LST'], bins='auto', color='#0504aa', alpha=0.7)
            plt.grid(axis='y', alpha=0.75)
            plt.xlabel('Value')
            plt.ylabel('Occurences')    
            plt.tight_layout()
            plt.savefig(output_path+'histo_delta_LST_at_'+nome_stazione+'_'+info_periodo+'_'+ data_frame['time'].iloc[0].strftime("%Y-%m-%d") + '_'+ data_frame['time'].iloc[-1].strftime("%Y-%m-%d")+'.pdf')
            
            plt.figure(figsize=(10, 6))
            plt.title('Total Column Water Vapour histograms@' +nome_stazione+ '- '+ start_date + ' - '+ end_date+' ≈1100 px Buffer')
            n, bins, patches = plt.hist(x=data_frame['tcwv'], bins='auto', color='navy', alpha=0.7)
            plt.grid(axis='y', alpha=0.75)
            plt.xlabel('Value')
            plt.ylabel('Occurences')
            plt.tight_layout()
            plt.savefig(output_path+'histo_TCWV_at_'+nome_stazione+'_'+info_periodo+'_'+ data_frame['time'].iloc[0].strftime("%Y-%m-%d") + '_'+ data_frame['time'].iloc[-1].strftime("%Y-%m-%d")+'.pdf')
            
            plt.figure(figsize=(10, 6))
            plt.title('Cloud Cover histograms@' +nome_stazione+ '- '+ start_date + ' - '+ end_date+' ≈1100 px Buffer')
            n, bins, patches = plt.hist(x=data_frame['i_cloud_cover'], bins='auto', color='navy', alpha=0.7)
            plt.grid(axis='y', alpha=0.75)
            plt.xlabel('Value')
            plt.ylabel('Occurences')
            plt.tight_layout()
            plt.savefig(output_path+'histo_CC_at_'+nome_stazione+'_'+info_periodo+'_'+ data_frame['time'].iloc[0].strftime("%Y-%m-%d") + '_'+ data_frame['time'].iloc[-1].strftime("%Y-%m-%d")+'.pdf')
            
            plt.figure(figsize=(10, 6))
            plt.title('Solar Zenith Angle histograms@' +nome_stazione+ '- '+ start_date + ' - '+ end_date+' ' + str(sza_buffer) +' px Buffer')
            n, bins, patches = plt.hist(x=data_frame['l_SZA_mean'], bins='auto', color='green', alpha=0.7)
            plt.grid(axis='y', alpha=0.75)
            plt.xlabel('Value')
            plt.ylabel('Occurences')
            plt.tight_layout()
            plt.savefig(output_path+'histo_SZA_at_'+nome_stazione+'_'+info_periodo+'_'+ data_frame['time'].iloc[0].strftime("%Y-%m-%d") + '_'+ data_frame['time'].iloc[-1].strftime("%Y-%m-%d")+'.pdf')
            
            plt.figure(figsize=(10, 6))
            plt.title('Solar Azimuth Angle histograms@' +nome_stazione+ '- '+ start_date + ' - '+ end_date +' '+ str(saa_buffer) +' px Buffer')
            n, bins, patches = plt.hist(x=data_frame['o_SAA_mean'], bins='auto', color='green', alpha=0.7)
            plt.grid(axis='y', alpha=0.75)
            plt.xlabel('Value')
            plt.ylabel('Occurences')
            plt.tight_layout()
            plt.savefig(output_path+'histo_SAA_at_'+nome_stazione+'_'+info_periodo+'_'+ data_frame['time'].iloc[0].strftime("%Y-%m-%d") + '_'+ data_frame['time'].iloc[-1].strftime("%Y-%m-%d")+'.pdf')
            
            plt.figure(figsize=(10, 6))
            plt.title('View Zenith Angle histograms@' +nome_stazione+ '- '+ start_date + ' - ' + end_date + ' ' + str(vza_buffer) +' px Buffer')
            n, bins, patches = plt.hist(x=data_frame['r_VZA_mean'], bins='auto', color='green', alpha=0.7)
            plt.grid(axis='y', alpha=0.75)
            plt.xlabel('Value')
            plt.ylabel('Occurences')
            plt.tight_layout()
            plt.savefig(output_path+'histo_VZA_at_'+nome_stazione+'_'+info_periodo+'_'+ data_frame['time'].iloc[0].strftime("%Y-%m-%d") + '_'+ data_frame['time'].iloc[-1].strftime("%Y-%m-%d")+'.pdf')
            
            plt.figure(figsize=(10, 6))
            plt.title('View Azimuth Angle histograms@' +nome_stazione+ '- '+ start_date + ' - '+ end_date +' ' + str(vaa_buffer) +' px Buffer')
            n, bins, patches = plt.hist(x=data_frame['u_VAA_mean'], bins='auto', color='green', alpha=0.7)
            plt.grid(axis='y', alpha=0.75)
            plt.xlabel('Value')
            plt.ylabel('Occurences')
            plt.tight_layout()
            plt.savefig(output_path+'histo_VAA_at_'+nome_stazione+'_'+info_periodo+'_'+ data_frame['time'].iloc[0].strftime("%Y-%m-%d") + '_'+ data_frame['time'].iloc[-1].strftime("%Y-%m-%d")+'.pdf')
            
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
df_comparison['season'] = df_comparison['time'].apply(get_season)

stats = []

print('overall all years')
df_comparison_all = df_comparison.copy()
all_start_date = df_comparison_all['time'].iloc[0].strftime("%Y-%m-%d")
all_end_date = df_comparison_all['time'].iloc[-1].strftime("%Y-%m-%d")
result1 = compute_stats(df_comparison_all, 'overall_all_years')
result1.append('overall all years')
stats.append(result1)

df_comparison_all.to_excel(output_path+'output_overall_all_years_'+all_start_date+'_'+all_end_date+'.xlsx',index=False)
#pdb.set_trace()

#stats_winter = []
print('winter all years')
mask_winter = df_comparison_all['season'] == 'winter'
df_comparison_all_winter = df_comparison_all[mask_winter]
all_winter_start_date = df_comparison_all_winter['time'].iloc[0].strftime("%Y-%m-%d")
all_winter_end_date =   df_comparison_all_winter['time'].iloc[-1].strftime("%Y-%m-%d")
result2 = compute_stats(df_comparison_all_winter, 'winter_all_years')
result2.append('winter all years')
stats.append(result2)
df_comparison_all_winter.to_excel(output_path+'output_winter_all_years_'+all_winter_start_date+'_'+all_winter_end_date+'.xlsx',index=False)

#stats_spring = []
print('spring all years')
mask_spring = df_comparison_all['season'] == 'spring'
df_comparison_all_spring = df_comparison_all[mask_spring]
all_spring_start_date = df_comparison_all_spring['time'].iloc[0].strftime("%Y-%m-%d")
all_spring_end_date =   df_comparison_all_spring['time'].iloc[-1].strftime("%Y-%m-%d")
result5 = compute_stats(df_comparison_all_spring, 'spring_all_years')
result5.append('spring all years')
stats.append(result5)
df_comparison_all_spring.to_excel(output_path+'output_spring_all_years_'+all_spring_start_date+'_'+all_spring_end_date+'.xlsx',index=False)

#stats_summer = []
print('summer all years')
mask_summer = df_comparison_all['season'] == 'summer'
df_comparison_all_summer = df_comparison_all[mask_summer]
all_summer_start_date = df_comparison_all_summer['time'].iloc[0].strftime("%Y-%m-%d")
all_summer_end_date =   df_comparison_all_summer['time'].iloc[-1].strftime("%Y-%m-%d")
result6 = compute_stats(df_comparison_all_summer, 'summer_all_years')
result6.append('summer all years')
stats.append(result6)
df_comparison_all_summer.to_excel(output_path+'output_summer_all_years_'+all_summer_start_date+'_'+all_summer_end_date+'.xlsx',index=False)

#stats_autumn = []
print('autumn all years')
mask_autumn = df_comparison_all['season'] == 'autumn'
df_comparison_all_autumn = df_comparison_all[mask_autumn]
all_autumn_start_date = df_comparison_all_autumn['time'].iloc[0].strftime("%Y-%m-%d")
all_autumn_end_date =   df_comparison_all_autumn['time'].iloc[-1].strftime("%Y-%m-%d")
result7 = compute_stats(df_comparison_all_autumn, 'autumn_all_years')
result7.append('autumn all years')
stats.append(result7)
df_comparison_all_autumn.to_excel(output_path+'output_autumn_all_years_'+all_autumn_start_date+'_'+all_autumn_end_date+'.xlsx',index=False)

for year in years:
            print('Year:', year)
            start_date = str(year)+'-01-01'
            end_date = str(year)+'-12-31'
            mask_time = (df_comparison_all['time'] >= start_date) & (df_comparison_all['time'] <= end_date)
            df_comparison = df_comparison_all.loc[mask_time]

            print('overall year ', year)
            print('df_comparison\n', df_comparison.head(5))
            print ('rows gbov', len(df_stazione.index))
            print ('rows landsat', len(df_landsat.index))
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
                year_winter_start_date = df_winter_year['time'].iloc[0].strftime("%Y-%m-%d")
                year_winter_end_date =   df_winter_year['time'].iloc[-1].strftime("%Y-%m-%d")
                df_winter_year.to_excel(output_path+'output_'+str(year)+'_winter_'+year_winter_start_date+'_'+year_winter_end_date+'.xlsx',index=False)
            
            print('spring year', year)
            mask_spring_year = df_comparison['season'] == 'spring'
            df_spring_year = df_comparison[ mask_spring_year]
            
            if len(df_spring_year.index) != 0:
                result8 = compute_stats(df_spring_year, str(year)+'_spring')
                result8.append(str(year)+'_spring')
                stats.append(result8)
                year_spring_start_date = df_spring_year['time'].iloc[0].strftime("%Y-%m-%d")
                year_spring_end_date =   df_spring_year['time'].iloc[-1].strftime("%Y-%m-%d")
                df_spring_year.to_excel(output_path+'output_'+str(year)+'_spring_'+year_spring_start_date+'_'+year_spring_end_date+'.xlsx',index=False)
                
            print('summer year', year)
            mask_summer_year = df_comparison['season'] == 'summer'
            df_summer_year = df_comparison[ mask_summer_year]
            
            if len(df_summer_year.index) != 0:
                result9 = compute_stats(df_summer_year, str(year)+'_summer')
                result9.append( str(year)+'_summer')
                stats.append(result9)
                year_summer_start_date = df_summer_year['time'].iloc[0].strftime("%Y-%m-%d")
                year_summer_end_date =   df_summer_year['time'].iloc[-1].strftime("%Y-%m-%d")
                df_summer_year.to_excel(output_path+'output_'+str(year)+'_summer_'+year_summer_start_date+'_'+year_summer_end_date+'.xlsx',index=False)
                
            print('autumn year', year)
            mask_autumn_year = df_comparison['season'] == 'autumn'
            df_autumn_year = df_comparison[ mask_autumn_year]
            
            if len(df_autumn_year.index) != 0:
                result10 = compute_stats(df_autumn_year, str(year)+'_autumn')
                result10.append(str(year)+'_autumn')
                stats.append(result10)
                year_autumn_start_date = df_autumn_year['time'].iloc[0].strftime("%Y-%m-%d")
                year_autumn_end_date =   df_autumn_year['time'].iloc[-1].strftime("%Y-%m-%d")
                df_autumn_year.to_excel(output_path+'output_'+str(year)+'_autumn_'+year_autumn_start_date+'_'+year_autumn_end_date+'.xlsx',index=False)
            
            
# Assuming stats is a list of data in the following order: [start_date, end_date, number_of_rows, mean, std, median, NMAD, tcwv_mean, tcwv_median, SAZ_mean, SAZ_median, cc_mean, cc_median, season]
stats = np.array(stats, dtype=object)

# Define column names including "season"
column_names = ['start_date', 'end_date', 'number_of_rows', 'delta_LST_mean', 'delta_LST_std', 'delta_LST_median', 'delta_LST_NMAD', 'delta_lst_buffer', 'tcwv_mean', 'tcwv_std', 'tcwv_median', 'tcwv_NMAD', 'SZA_mean', 'SZA_std', 'SZA_median', 'SZA_NMAD', 'SZA_buffer', 'SAA_mean', 'SAA_std', 'SAA_median', 'SAA_NMAD', 'SAA_buffer', 'VZA_mean', 'VZA_std', 'VZA_median', 'VZA_NMAD', 'VZA_buffer', 'VAA_mean', 'VAA_std', 'VAA_median', 'VAA_NMAD', 'VAA_buffer', 'CC_mean', 'CC_std', 'CC_median', 'CC_NMAD', 'season']

# Create a pandas DataFrame
df = pd.DataFrame(stats, columns=column_names)

# Specify the Excel file path
excel_filename = output_path + 'stats_l8_buffer_1100_' + nome_stazione + '.xlsx'

# Save the DataFrame to an Excel file
df.to_excel(excel_filename, index=False)
#plt.show()
