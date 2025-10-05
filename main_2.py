#main.py
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime 
import logging
import json
import lightgbm as lgb
import optuna


from src.config import *
from src.loader import cargar_datos
from src.constr_lista_cols import contruccion_cols
from src.feature_engineering import feature_engineering_delta, feature_engineering_lag , feature_engineering_ratio,feature_engineering_linreg,feature_engineering_max_min
from src.preprocesamiento import split_train_binario
from src.lgbm_optimizacion import optim_hiperp_binaria 
from src.lgbm_train_test import  entrenamiento_lgbm , evaluacion_lgbm
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
os.makedirs(PATH_OUTPUT_LGBM,exist_ok=True)

fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
fecha_name = '2025-09-26_17-37-58'
nombre_log = f"log_entrenamiento_directo_sin_optuna_{fecha}.log"

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


    # ## 2. Feature Engineering
    df=feature_engineering_lag(df,cols_lag_delta_max_min_regl,2)
    df=feature_engineering_delta(df,cols_lag_delta_max_min_regl,2)
    df=feature_engineering_ratio(df,cols_ratios)


# # ----------------------------------------------------------------------------------------------------------
    ## 3. Preprocesamiento para entrenamiento

    # split X_train, y_train
    MES_TRAIN.append(MES_TEST)
    X_train, y_train_binaria, w_train, X_apred, y_apred, y_apred_class, w_apred = split_train_binario(df,MES_TRAIN,MES_A_PREDECIR)

    ## 4. Carga de mejores Hiperparametros
    name_lgbm=f"_{fecha}"
    name_best_params_file=f"best_paramsbinaria_{fecha_name}.json"
    storage_name = "sqlite:///" + db_path + "optimization_lgbm.db"
    study = optuna.load_study(study_name='study_lgbm_binaria_'+fecha_name,storage=storage_name)
    
    ## 5. Entrenamiento lgbm con la mejor iteracion y los mejores hiperparametros
    best_iter = study.best_trial.user_attrs["best_iter"]
    
    with open(bestparams_path+name_best_params_file, "r") as f:
        best_params = json.load(f)
    logger.info(f"Esta bien los hiperparametros que cargué ? : {study.best_trial.params ==best_params }")
    
    model_lgbm = entrenamiento_lgbm(X_train , y_train_binaria,w_train ,best_iter,best_params , name=name_lgbm)

    # Preparacion para predecir
    
    y_apred=X_apred[["numero_de_cliente"]]
    y_pred=model_lgbm.predict(X_apred)
    y_apred["prediction"] = y_pred
    y_apred["prediction"]=y_apred["prediction"].apply(lambda x : 1 if x >=0.025 else 0)
    logger.info(f"cantidad de bajas predichas : {(y_apred==1).sum()}")
    y_apred=y_apred.set_index("numero_de_cliente")
    y_apred.to_csv(f"outputs/lgbm_model/final_prediction/prediccion_main2_{fecha}.csv")




    
    logger.info(f">>> Ejecucion finalizada. Revisar logs para mas detalles. {nombre_log}")


if __name__ =="__main__":
    main()