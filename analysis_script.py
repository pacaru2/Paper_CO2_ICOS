import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Cargar y ordenar datos
df = pd.read_csv("resultados/excel_resultados_metrics_paper_csv.csv", sep=";", decimal=',')
df = df.sort_values(by=['Station', 'Model'])

# Cálculo de estadísticas por modelo y categoría
median_koppen = df.groupby(['Model', 'Koppen'])[['MAPE', 'MAE', 'MSE', 'RMSE', 'RMSLE']].median().reset_index()
median_modelo = df.groupby(['Model'])[['MAPE', 'MAE', 'MSE', 'RMSE', 'RMSLE']].median().reset_index()
median_modelo_round = median_modelo.round(2)
mean_modelo = df.groupby(['Model'])[['MAPE', 'MAE', 'MSE', 'RMSE', 'RMSLE']].mean().reset_index()
mean_modelo_round = mean_modelo.round(2)

# Percentiles
percentiles_25_modelo = df.groupby(['Model'])[['MAPE', 'MAE', 'MSE', 'RMSE', 'RMSLE']].quantile(0.25).reset_index()
percentiles_75_modelo = df.groupby(['Model'])[['MAPE', 'MAE', 'MSE', 'RMSE', 'RMSLE']].quantile(0.75).reset_index()
percentiles_25_modelo_round = percentiles_25_modelo.round(2)
percentiles_75_modelo_round = percentiles_75_modelo.round(2)

# Errores mínimos por estación y sitio
min_error_station = df.groupby('Station')[['MAPE', 'MAE', 'MSE', 'RMSE', 'RMSLE']].min().reset_index()
min_error_site = df.groupby('Site')[['MAPE', 'MAE', 'MSE', 'RMSE', 'RMSLE']].min().reset_index()

# Medianas por ESA y altura
median_esa = df.groupby(['Model', 'ESA'])[['MAPE', 'MAE', 'MSE', 'RMSE', 'RMSLE']].median().reset_index()
df['Height_dis'] = pd.cut(df['Height'], bins=5, labels=False)
median_height = df.groupby(['Model', 'Height_dis'])[['MAPE', 'MAE', 'MSE', 'RMSE', 'RMSLE']].median().reset_index()

# Redondeo de medianas
median_koppen_round = median_koppen.round(2)
median_esa_round = median_esa.round(2)

# Guardar resultados
median_koppen_round.to_csv('resultados/median_koppen_round.txt', index=False)
median_esa_round.to_csv('resultados/median_esa_round.txt', index=False)
mean_modelo_round.to_csv('resultados/mean_modelo_round.txt', index=False)
median_modelo_round.to_csv('resultados/median_modelo_round.txt', index=False)

# Agrupación por nivel
bins = [0, 200, 500, 1000, 2000, float('inf')]
labels = ['0-200', '200-500', '500-1000', '1000-2000', '>2000']
df['level_group'] = pd.cut(df['Level'], bins=bins, labels=labels, include_lowest=True)
median_level_group = df.groupby(['Model', 'level_group'])['MAPE'].median().reset_index()
median_level_group_round = median_level_group.round(2)
median_level_group_round.to_csv('resultados/median_level_group_round.txt', index=False)

# Agrupación por latitud
bins = [34, 43, 50, 57, 64, float('inf')]
labels = ['34ºN-43ºN', '43ºN-50ºN', '50ºN-57ºN', '57ºN-64ºN', '>64ºN']
df['latitude_group'] = pd.cut(df['Latitude'], bins=bins, labels=labels, include_lowest=True)
median_latitude_group = df.groupby(['Model', 'latitude_group'])['MAPE'].median().reset_index()
median_latitude_group_round = median_latitude_group.round(2)
median_latitude_group_round.to_csv('resultados/median_latitude_group_round.txt', index=False)

# Modelos con menor MAPE por estación
min_mape_station = df.groupby(['Station', 'Model'])['MAPE'].min().reset_index()
min_mape_model_station = min_mape_station.loc[min_mape_station.groupby('Station')['MAPE'].idxmin()].reset_index(drop=True)
min_mape_station_sorted = min_mape_station.sort_values(['Station', 'MAPE'], ascending=[True, True])
top3_models_per_station = min_mape_station_sorted.groupby('Station').head(3).reset_index(drop=True)

# Ranking de modelos por estación
min_mape_station_sorted['Rank'] = min_mape_station_sorted.groupby('Station')['MAPE'].rank(method='first')
top3_models_per_station = min_mape_station_sorted[min_mape_station_sorted['Rank'] <= 3]
top3_models_per_station['MAPE'] = top3_models_per_station['MAPE'].round(2)

# Conteo de ranking de modelos
top3_models_per_station['Rank'] = top3_models_per_station['Rank'].astype(int)
model_rank_count = top3_models_per_station.groupby(['Model', 'Rank']).size().unstack(fill_value=0)
model_rank_count.columns = ['First Place', 'Second Place', 'Third Place']
