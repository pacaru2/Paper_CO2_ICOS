# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 18:24:07 2024

@author: Pablo Catret Ruber
"""
print("Empieza")
import os
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import ExponentialSmoothing, Prophet,  RandomForest, LightGBMModel, BlockRNNModel, NHiTSModel, TCNModel, TiDEModel, NaiveEnsembleModel
from darts.metrics import mae, rmse, mape, mse, smape, rmsle
from darts.dataprocessing.transformers import Scaler
import matplotlib.pyplot as plt
import logging
import joblib

logging.basicConfig(level=logging.INFO)

# Métricas adicionales
def maape(y_true, y_pred):
    return np.mean(np.arctan(np.abs((y_true - y_pred) / y_true)))

def mase(y_true, y_pred, naive_pred):
    return np.mean(np.abs(y_true - y_pred)) / np.mean(np.abs(y_true - naive_pred))

def gmrae(y_true, y_pred, naive_pred):
    return np.exp(np.mean(np.log(np.abs((y_true - y_pred) / (y_true - naive_pred)))))

def lmae(y_true, y_pred):
    return np.mean(np.log(1 + np.abs(y_true - y_pred)))

def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    return np.mean(np.where(np.abs(error) <= delta, 0.5 * error**2, delta * (np.abs(error) - 0.5 * delta)))


def process_station(args):
    nombres_i, nuevos_i, antiguos_i = args
    logging.info(f"Processing station {nombres_i}")
    try:
        with open("data/"+antiguos_i, 'r') as archivo:
            lineas = archivo.readlines()
            numero_de_linea_del_encabezado = None
            for j, linea in enumerate(lineas):
                if linea.startswith('#'):
                    numero_de_linea_del_encabezado = j

        df = pd.read_csv("data/"+antiguos_i, sep=';', comment='#')
        df.columns = ['Site', 'SamplingHeight', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'DecimalDate', 'co2', 'Stdev', 'NbPoints', 'Flag', 'InstrumentId', 'QualityId', 'LTR', 'CMR', 'STTB', 'QcBias', 'QcBiasUncertainty', 'co2-WithoutSpikes', 'Stdev-WithoutSpikes', 'NbPoints-WithoutSpikes'][:len(df.columns)]

        with open("data/"+nuevos_i, 'r') as archivo:
            lineas = archivo.readlines()
            numero_de_linea_del_encabezado = None
            for j, linea in enumerate(lineas):
                if linea.startswith('#'):
                    numero_de_linea_del_encabezado = j

        df2 = pd.read_csv("data/"+nuevos_i, sep=';', comment='#')
        df2.columns = ['Site', 'SamplingHeight', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'DecimalDate', 'co2', 'Stdev', 'NbPoints', 'Flag', 'InstrumentId', 'QualityId', 'LTR', 'CMR', 'STTB', 'QcBias', 'QcBiasUncertainty', 'co2-WithoutSpikes', 'Stdev-WithoutSpikes', 'NbPoints-WithoutSpikes'][:len(df2.columns)]

        df = pd.concat([df, df2])
        df = df.replace(-999.99, np.nan)
        df = df.replace(-9.99, np.nan)
        df['TIMESTAMP'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
        df = df.drop(['Year', 'Month', 'Day', 'Hour', 'Minute'], axis=1)

        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
        df2 = df.set_index('TIMESTAMP')

        df_daily = df2[['co2']].resample('D').mean()

        def find_last_long_nan_interval(series, max_missing_days=30):
            na_run_length = 0
            last_long_nan_end_index = -1

            for i in range(len(series)):
                if pd.isna(series.iloc[i]):
                    na_run_length += 1
                else:
                    if na_run_length > max_missing_days:
                        last_long_nan_end_index = i - 1
                    na_run_length = 0

            if na_run_length > max_missing_days:
                last_long_nan_end_index = len(series) - 1

            return last_long_nan_end_index

        last_long_nan_end_index = find_last_long_nan_interval(df_daily['co2'])

        if last_long_nan_end_index == -1:
            df_daily['co2'] = df_daily['co2']
        else:
            df_daily['co2'] = df_daily['co2'].iloc[last_long_nan_end_index + 1:]

        Q1 = df_daily['co2'].quantile(0.25)
        Q3 = df_daily['co2'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_daily['co2'] = df_daily['co2'][(df_daily['co2'] >= lower_bound) & (df_daily['co2'] <= upper_bound)]

        df_daily['co2_interpolated'] = df_daily['co2'].interpolate(method='pchip')

        df = df_daily.reset_index()
        df = df[['TIMESTAMP', 'co2_interpolated']]
        df.columns = ['TIMESTAMP', 'valor']

        df = df.dropna()
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
        df = df.sort_values('TIMESTAMP')
        print(df[["TIMESTAMP"]].tail(1))
        series = TimeSeries.from_dataframe(df, 'TIMESTAMP', 'valor')

        scaler = Scaler()
        series_scaled = scaler.fit_transform(series)

        train, val = series_scaled.split_before(len(series) - 365)

        input_chunk_length = 366
        lags = 366
        output_chunk_length = 365

        models = {
            'ETS': ExponentialSmoothing(seasonal_periods=365),
            'Prophet': Prophet(),
            'LightGBM': LightGBMModel(lags=lags, output_chunk_length=output_chunk_length),
            'N-HiTS': NHiTSModel(input_chunk_length, output_chunk_length),
            'TCN': TCNModel(input_chunk_length, output_chunk_length),
            'TiDE': TiDEModel(input_chunk_length, output_chunk_length),
            'RF': RandomForest(lags=lags, output_chunk_length=output_chunk_length),
            "ProHITS": NaiveEnsembleModel(forecasting_models=[Prophet(), NHiTSModel(input_chunk_length, output_chunk_length)]),
            "ProphetLGB": NaiveEnsembleModel(forecasting_models=[Prophet(), LightGBMModel(lags=lags, output_chunk_length=output_chunk_length)]),
            "ProphetTCN": NaiveEnsembleModel(forecasting_models=[Prophet(), TCNModel(input_chunk_length, output_chunk_length)]),
            "GRU": BlockRNNModel(input_chunk_length, output_chunk_length, model="GRU", hidden_dim=25, n_rnn_layers=1, dropout=0.10),
            "LSTM": BlockRNNModel(input_chunk_length, output_chunk_length, model="LSTM", hidden_dim=30, n_rnn_layers=2, dropout=0.15),
        }

        evaluation_metrics = []
        csv_file_path = "resultados/model_evaluation_metrics_paper.csv"
        for name, model in models.items():
            print(name)
            model.fit(train)
            forecast = model.predict(len(val))

            forecast = scaler.inverse_transform(forecast)
            val_not_scaled = scaler.inverse_transform(val)

            model_mae = mae(val_not_scaled, forecast)
            model_mse = mse(val_not_scaled, forecast)
            model_rmse = rmse(val_not_scaled, forecast)
            model_mape = mape(val_not_scaled, forecast)
            model_smape = smape(val_not_scaled, forecast)
            model_rmsle = rmsle(val_not_scaled, forecast)

            

            evaluation_metrics.append({
                'Station': nombres_i,
                'Model': name,
                'MAE': model_mae,
                'MSE': model_mse,
                'RMSE': model_rmse,
                'MAPE': model_mape,
                'sMAPE': model_smape,
                'RMSLE': model_rmsle
            })

            plt.figure(figsize=(10, 6))
            val_not_scaled.plot(label='Actual')
            forecast.plot(label='Forecast')
            plt.title(f"Forecast vs Actuals for {name}")
            plt.legend()
            plt.savefig(f"graficas/{nombres_i}_{name}_forecast_vs_actuals.png")
            plt.close()

            joblib.dump(model, f"modelos/{nombres_i}_{name}_paper_model.pkl")

        df_metrics = pd.DataFrame(evaluation_metrics)
        if not os.path.isfile(csv_file_path):
            df_metrics.to_csv(csv_file_path, index=False)
        else:
            df_metrics.to_csv(csv_file_path, mode='a', header=False, index=False)

        print(f"Metrics for station {nombres_i} saved to {csv_file_path}")

    except:
        print(f'Falla: {nombres_i}')
        


def main():
    with open("data/estaciones_alturas.txt", 'r') as archivo1:
        nombres = [line.strip() for line in archivo1]
    with open("data/paths_datos_nuevos_2024.txt", 'r') as archivo2:
        nuevos = [line.strip() for line in archivo2]
    with open("data/paths_datos_antiguos_2024.txt", 'r') as archivo3:
        antiguos = [line.strip() for line in archivo3]
    tasks = zip(nombres, nuevos, antiguos)

    for nombres_i, antiguos_i, nuevos_i in tasks:
        process_station((nombres_i, antiguos_i, nuevos_i))

def main():
    with open("data/estaciones_alturas.txt", 'r') as archivo1:
        nombres = [line.strip() for line in archivo1]
    with open("data/paths_datos_nuevos_2024.txt", 'r') as archivo2:
        nuevos = [line.strip() for line in archivo2]
    with open("data/paths_datos_antiguos_2024.txt", 'r') as archivo3:
        antiguos = [line.strip() for line in archivo3]
    tasks = zip(nombres, nuevos, antiguos)

    allowed_names = [
        "BIR 10.0", "BIR 50.0", "BIR 75.0", "GAT 132.0", "GAT 216.0", "GAT 30.0", "GAT 341.0", "GAT 60.0",
        "HPB 131.0", "HPB 50.0", "HPB 93.0", "HTM 150.0", "HTM 70.0", "IPR 100.0", 
        "IPR 40.0", "IPR 60.0",
        "JFJ 13.9", "JUE 120.0", "JUE 50.0", "JUE 80.0", "KIT 100.0", "KIT 200.0", "KIT 60.0", "KRE 10.0",
        "KRE 125.0", "KRE 250.0", "KRE 50.0", "LMP 8.0", "LUT 60.0", "NOR 100.0", "NOR 32.0", "NOR 58.0",
        "OPE 10.0", "OPE 120.0", "OPE 50.0", "OXK 163.0", "OXK 23.0", "OXK 90.0", "PAL 12.0", "PUI 47.0",
        "PUI 84.0", "PUY 10.0", "RGL 45.0", "RGL 90.0", "SAC 100.0", "SAC 15.0", "SAC 60.0", "SMR 125.0",
        "SMR 16.8", "SMR 67.2", "SNO 20.0", "SNO 50.0", "SNO 85.0", "SSL 12.0", "SSL 35.0", "STE 127.0",
        "STE 187.0", "STE 252.0", "STE 32.0", "TOH 10.0", "TOH 110.0", "TOH 147.0", "TOH 76.0", "TRN 100.0",
        "TRN 180.0", "TRN 5.0", "TRN 50.0", "UTO 57.0", "WES 14.0", "ZSF 3.0"
    ]
    for nombres_i, antiguos_i, nuevos_i in tasks:
        if nombres_i in allowed_names:
            process_station((nombres_i, antiguos_i, nuevos_i))


        
if __name__ == "__main__":
    main()
