# CO2 Data Analysis and Forecasting

This repository contains Python scripts for processing, cleaning, and analyzing CO2 concentration data collected from various monitoring stations. Additionally, the scripts implement multiple time series models to predict future values and perform statistical analysis of model performance.

## Features
- Reading and processing CO2 data files.
- Data cleaning and preprocessing, including interpolation and outlier detection.
- Transformation of data into time series format.
- Application of various forecasting models, including Prophet, LightGBM, N-HiTS, TCN, TiDE, and recurrent neural network models (GRU and LSTM).
- Model evaluation using metrics such as MAE, MSE, RMSE, MAPE, and RMSLE.
- Generation of comparative plots of predictions vs. actual data.
- Statistical analysis of model performance, including median, mean, percentiles, and ranking models by error metrics.
- Saving trained models and evaluation results.

## Requirements
This project requires the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- darts
- joblib
- logging

You can install the required dependencies by running:
```sh
pip install pandas numpy matplotlib seaborn darts joblib
```

## Usage
1. Place the data files in the `data` folder.
2. Ensure that the files `estaciones_alturas.txt`, `paths_datos_nuevos_2024.txt`, and `paths_datos_antiguos_2024.txt` contain the correct paths to the data.
3. Run the main script:
```sh
python train_script.py
```
4. To perform model performance analysis, ensure that the input file contains the necessary columns: `Station`, `Model`, `MAPE`, `MAE`, `MSE`, `RMSE`, `RMSLE`, `Koppen`, `ESA`, `Height`, `Level`, `Latitude`, and `Site`. Then, execute the statistical analysis script:
```sh
python analysis_script.py
```

## Project Structure
```
/
├── data/                  # Input files with CO2 data
├── modelos/               # Saved trained models
├── graficas/              # Prediction plots
├── resultados/            # Evaluation results
├── train_script.py        # Main script
├── analysis_script.py     # Statistical analysis script
└── README.md              # This file
```

## Author
Pablo Catret Ruber

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

