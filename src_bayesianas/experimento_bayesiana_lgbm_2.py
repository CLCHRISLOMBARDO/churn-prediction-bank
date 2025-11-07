#main.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime 
import logging
import json
import lightgbm as lgb
from src.config import *
from src.loader import cargar_datos
from src.constr_lista_cols import contruccion_cols , contrs_cols_dropear_feat_imp
from src.feature_engineering import feature_engineering_delta, feature_engineering_lag , feature_engineering_ratio,feature_engineering_linreg,feature_engineering_max_min ,feature_engineering_normalizacion,feature_engineering_drop_cols
from src.preprocesamiento import split_train_test_apred , conversion_binario
from src.lgbm_optimizacion import optim_hiperp_binaria , graficos_bayesiana
from src.lgbm_train_test import  entrenamiento_lgbm , prediccion_test_lgbm,grafico_feature_importance ,evaluacion_public_private
## ---------------------------------------------------------Configuraciones Iniciales -------------------------------


## Carga de variables
n_trials=N_TRIALS



logger = logging.getLogger(__name__)

## --------------------------------------------------------Funcion main ------------------------------------------

def lanzar_bayesiana_lgbm(fecha:str , semillas:list,n_experimento:str|int ,proceso_ppal:str):
    #"""---------------------- CAMBIAR INPUTS --------------------------------------------------------"""
    numero=n_experimento
    #"""----------------------------------------------------------------------------------------------"""
    # name=f"bayesiana_{numero}_lgbm_{fecha}"
    name=f"{proceso_ppal}_{numero}_lgbm_{len(semillas)}_SEMILLAS_{N_TRIALS}_TRIALS_{N_BOOSTS}_BOOSTS"
    nombre_log=f"log_{name}_{fecha}"
    logger.info(f"Inicio de ejecucion del flujo : {name}")

    ## 0. load datos
    df=cargar_datos(FILE_INPUT_DATA)
    print(df.head())
                                ## A - AGREGADO DE FEATURES

    ## 1. Contruccion de las columnas
    columnas=contruccion_cols(df)
    cols_lag_delta_max_min_regl=columnas[0]
    cols_ratios=columnas[1]
    # ## 2. Feature Engineering
    df=feature_engineering_lag(df,cols_lag_delta_max_min_regl,2)
    df=feature_engineering_delta(df,cols_lag_delta_max_min_regl,2)
    df=feature_engineering_ratio(df,cols_ratios)
    df=feature_engineering_linreg(df,cols_lag_delta_max_min_regl)


# # ----------------------------------------------------------------------------------------------------------
    ## 3. Preprocesamiento para entrenamiento
    # split X_train, y_train
    df=conversion_binario(df)
    X_train, y_train_binaria,y_train_class, w_train, X_test, y_test_binaria, y_test_class, w_test,X_apred, y_apred = split_train_test_apred(df,MES_TRAIN,MES_TEST,MES_A_PREDECIR,SEMILLA,0.4)
 

    ## 4.a. Optimizacion Hiperparametros
   
    study = optim_hiperp_binaria(X_train , y_train_binaria,w_train ,n_trials , name,fecha,semillas)
    graficos_bayesiana(study ,fecha, name)
    logger.info("=== ANÁLISIS DE RESULTADOS ===")
    trials_df = study.trials_dataframe()
    if len(trials_df) > 0:
        top_5 = trials_df.nlargest(5, 'value')
        logger.info("Top 5 mejores trials:")
        for idx, trial in top_5.iterrows():
            logger.info(f"  Trial {trial['number']}: {trial['value']:,.0f}")
  
    logger.info("=== OPTIMIZACIÓN COMPLETADA ===")


    
    logger.info(f">>> Ejecucion finalizada. Revisar logs para mas detalles. {nombre_log}")

