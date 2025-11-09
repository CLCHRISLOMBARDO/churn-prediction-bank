import numpy as np
import pandas as pd
import logging
import json
from src.config import *
from src.loader import cargar_datos
from src.configuracion_inicial import creacion_df_small
from src.preprocesamiento import conversion_binario,split_train_test_apred
from src.constr_lista_cols import contruccion_cols
from src.feature_engineering import feature_engineering_lag,feature_engineering_delta,feature_engineering_max_min,feature_engineering_ratio,feature_engineering_linreg,feature_engineering_normalizacion,feature_engineering_drop_cols,feature_engineering_rank
from src.lgbm_train_test import entrenamiento_lgbm,grafico_feature_importance,prediccion_test_lgbm ,calc_estadisticas_ganancia,grafico_curvas_ganancia, grafico_hist_ganancia ,preparacion_ypred_kaggle
## ---------------------------------------------------------Configuraciones Iniciales -------------------------------
## Carga de variables

logger=logging.getLogger(__name__)

def lanzar_feat_eng(fecha:str ,n_fe:int , proceso_ppal:str):
    numero=n_fe
    #"""----------------------------------------------------------------------------------------------"""
    name=f"FEAT_ENG_{numero}_{proceso_ppal}_VENTANA_{VENTANA}"
    logger.info(f"PROCESO PRINCIPAL ---> {proceso_ppal}")
    logger.info(f"Comienzo del experimento : {name}")
    df_chiquito=creacion_df_small()

    # SELECCION DE COLUMNAS
    columnas=contruccion_cols(df_chiquito)
    cols_lag_delta_max_min_regl=columnas[0]
    cols_ratios=columnas[1]

    # FEATURE ENGINEERING
    feature_engineering_lag(df_chiquito,cols_lag_delta_max_min_regl,VENTANA)
    feature_engineering_delta(df_chiquito,cols_lag_delta_max_min_regl,VENTANA)
    feature_engineering_ratio(df_chiquito,cols_ratios)
    feature_engineering_linreg(df_chiquito , cols_lag_delta_max_min_regl,VENTANA)
    feature_engineering_max_min(df_chiquito,cols_lag_delta_max_min_regl ,VENTANA)
    

    #DROPEO DE COLULNAS
    # cols_a_dropear=["mcuentas_saldo"]
    feature_engineering_drop_cols(df_chiquito,columnas=None)

    logger.info("================ FIN DEL PROCESO DE FEAT ENG =============================")





    

    
