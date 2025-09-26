#main.py
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime 
import logging
import json
import lightgbm as lgb


from src.config import *
from src.loader import cargar_datos
from src.constr_lista_cols import contruccion_cols
from src.feature_engineering import feature_engineering_delta, feature_engineering_lag , feature_engineering_ratio,feature_engineering_linreg,feature_engineering_max_min
from src.preprocesamiento import split_train_binario
from src.optimizacion_lgbm import optim_hiperp_binaria 
# from src.lgbm_train import  entrenamiento_rf,distanceMatrix
## ---------------------------------------------------------Configuraciones Iniciales -------------------------------
## PATH

db_path = PATH_OUTPUT_OPTIMIZACION + 'db/'
bestparams_path = PATH_OUTPUT_OPTIMIZACION+'best_params/'


## Carga de variables
n_trials=N_TRIALS

## config basic logging
os.makedirs("logs",exist_ok=True)
os.makedirs(PATH_OUTPUT_OPTIMIZACION,exist_ok=True)
os.makedirs(db_path,exist_ok=True)
os.makedirs(bestparams_path,exist_ok=True)

fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
nombre_log = f"log_{fecha}.log"

logging.basicConfig(
    level=logging.INFO, #Puede ser INFO o ERROR
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/{nombre_log}", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

## --------------------------------------------------------Funcion main ------------------------------------------

def main():
    logger.info("Inicio de ejecucion.")

    ## 0. load datos
    df=cargar_datos(PATH_INPUT_DATA)
    print(df.head())

    ## 1. Contruccion de las columnas
    columnas=contruccion_cols(df)
    cols_lag_delta_max_min_regl=columnas[0]
    cols_ratios=columnas[1]


    ## 2. Feature Engineering
    df=feature_engineering_lag(df,cols_lag_delta_max_min_regl,2)
    df=feature_engineering_delta(df,cols_lag_delta_max_min_regl,2)
    # df=feature_engineering_max_min(df,cols_lag_delta_max_min_regl)
    df=feature_engineering_ratio(df,cols_ratios)
    # df=feature_engineering_linreg(df,cols_lag_delta_max_min_regl)


# # ----------------------------------------------------------------------------------------------------------
    ## 2. Preprocesamiento para entrenamiento

    # split X_train, y_train
    X_train, y_train_binaria, w_train, X_test, y_test_binaria, y_test_class, w_test = split_train_binario(df,MES_TRAIN,MES_TEST)
                # Guardo df
    # try:
    #     X_train.to_csv(path_output_data + "X_train_sample_imp.csv") 
    #     y_train.to_csv(path_output_data + "y_train_sample.csv") 
    #     logger.info(f"X shape {X_train.shape}, y shape{y_train} guardado en csv")
    # except Exception as e:
    #     logger.error(f"Error al guardar el df : {e}")
    #     raise


    ## 3. Optimizacion Hiperparametros
    name_lgbm=f"_lgbm_{fecha}"
    study = optim_hiperp_binaria(X_train , y_train_binaria,w_train ,n_trials , name=name_lgbm)
    best_params=study

    # model_rf_sample=entrenamiento_rf(X_train_sample_imp , y_train_sample ,best_params_sample,name=name_rf_sample)
    # class_index = np.where(model_rf_sample.classes_ == 1)[0][0]
    # proba_baja_sample=model_rf_sample.predict_proba(X_train_sample_imp)[:,class_index]
    # distancia_sample = distanceMatrix(model_rf_sample,X_train_sample_imp)
    
#    


    
    logger.info(f">>> Ejecucion finalizada. Revisar logs para mas detalles. {nombre_log}")


if __name__ =="__main__":
    main()