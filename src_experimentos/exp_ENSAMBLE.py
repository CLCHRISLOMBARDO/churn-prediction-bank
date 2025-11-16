#experimento_i.py
# EXPERIMENTO : Ensamble con lgb. Es el 8 pero pongo el subsampleo. Eliminaos cprestamos_personales y mprestamos_personales
import numpy as np
import pandas as pd
import logging
import json

from src.config import *
from src.configuracion_inicial import creacion_df_small
from src.constr_lista_cols import cols_a_dropear_variable_entera,contrs_cols_dropear_por_features_sufijos ,cols_a_dropear_variable_originales_o_percentiles
from src.preprocesamiento import split_train_test_apred
from src.lgbm_train_test import preparacion_nclientesbajas_zulip,entrenamiento_zlgbm,entrenamiento_lgbm,entrenamiento_lgbm_zs,grafico_feature_importance,prediccion_test_lgbm ,calc_estadisticas_ganancia,grafico_curvas_ganancia, grafico_hist_ganancia ,preparacion_ypred_kaggle,preparacion_ytest_proba_kaggle
## ---------------------------------------------------------Configuraciones Iniciales -------------------------------
## Carga de variables

logger=logging.getLogger(__name__)

def lanzar_experimento_ensamble(fecha:str ,semillas:list[int],n_experimento:int,proceso_ppal:str ="experimento"): 
    #"""---------------------- CAMBIAR INPUTS --------------------------------------------------------"""
    numero=n_experimento
    #"""----------------------------------------------------------------------------------------------"""
    n_semillas = len(semillas)
    name=f"{proceso_ppal}_{numero}_ENSAMBLE_{len(semillas)}_SEMILLAS"
    logger.info(f"PROCESO PRINCIPAL ---> {proceso_ppal}")
    logger.info(f"Comienzo del experimento : {name} con {n_semillas} semillas")

    # ---------------------- CONSTRUCCION COLUMNAS A ELIMINAR------------------------

    

logger.info(f">>> Ejecucion finalizada. Revisar logs para mas detalles.")

