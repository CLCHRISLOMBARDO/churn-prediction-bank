import numpy as np
import pandas as pd
import logging
import json
from src.config import *
from src.configuracion_inicial import creacion_df_small
from src.constr_lista_cols import contruccion_cols,cols_a_dropear_variable_entera
from src.feature_engineering import feature_engineering_percentil,feature_engineering_lag,feature_engineering_delta,feature_engineering_max_min,feature_engineering_ratio,feature_engineering_linreg,feature_engineering_drop_cols,feature_engineering_rank
## ---------------------------------------------------------Configuraciones Iniciales -------------------------------
## Carga de variables

logger=logging.getLogger(__name__)

def lanzar_feat_eng(fecha:str ,n_fe:int , proceso_ppal:str):
    numero=n_fe
    #"""----------------------------------------------------------------------------------------------"""
    name=f"FEAT_ENG_{numero}_{proceso_ppal}_VENTANA_{VENTANA}"
    logger.info(f"PROCESO PRINCIPAL ---> {proceso_ppal}")
    logger.info(f"Comienzo del experimento : {name}")
    df_completo_chiquito=creacion_df_small()

    # SELECCION DE COLUMNAS
    cols_percentil,cols_lag_delta_max_min_regl,cols_ratios=contruccion_cols(df_completo_chiquito)
 
    # # FEATURE ENGINEERING
    # Agregar conteo --> productos , conteos productos total, visa y master --> le aplica lags y deltas
    # Agregar conteo servicio --> conteo de servicios. 
    # Delta proporcion --> 
    # Agregar percentile --> 
    feature_engineering_percentil(df_completo_chiquito ,cols_percentil,bins=20)

    feature_engineering_lag(df_completo_chiquito,cols_lag_delta_max_min_regl,VENTANA)
    feature_engineering_delta(df_completo_chiquito,cols_lag_delta_max_min_regl,VENTANA)
    feature_engineering_ratio(df_completo_chiquito,cols_ratios)
    feature_engineering_linreg(df_completo_chiquito , cols_lag_delta_max_min_regl,VENTANA)
    feature_engineering_max_min(df_completo_chiquito,cols_lag_delta_max_min_regl ,VENTANA)
    

    #DROPEO DE COLULNAS
    # cols_a_dropear=["mcuentas_saldo"]
    # cols_a_dropear = cols_a_dropear_variable_entera(df_chiquito , ["mcuentas_saldo"])
    feature_engineering_drop_cols(df_chiquito,columnas=None , tabla="df_completo")

    logger.info("================ FIN DEL PROCESO DE FEAT ENG =============================")





    

    
