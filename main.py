#main.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime 
import logging
import json
import lightgbm as lgb
import optuna

from src.config import *
from src.configuracion_inicial import creacion_directorios,creacion_logg_local , creacion_logg_global
from src.generadora_semillas import create_semilla
from src.creacion_target import lanzar_creacion_clase_ternaria_binaria_peso
from src_feateng.feat_eng_1 import lanzar_feat_eng
from src_bayesianas.bayesiana_lgbm_2 import lanzar_bayesiana_lgbm

from src_experimentos.experimento_eda import lanzar_eda
from src_experimentos.experimento_1_lgbm import lanzar_experimento



## ---------------------------------------------------------Configuraciones Iniciales -------------------------------


## Carga de variables
n_trials=N_TRIALS
creacion_directorios()
fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
test= "TEST_TEST_TEST_TEST"
# ---------------------------------------------------------------------------------------------
competencia = COMPETENCIA
proceso_ppal = PROCESO_PPAL
# Conf fe
n_fe = N_FE
# Conf exp
n_experimento = N_EXPERIMENTO
n_semillas_exp = N_SEMILLAS_EXP
# Conf bay
n_bayesiana = N_BAYESIANA
n_semillas_bay=N_SEMILLAS_BAY
# ---------------------------------------------------------------------------------------------------------------------------
if proceso_ppal =="feat_eng":
    numero_proceso = n_fe
elif proceso_ppal == "experimento" or proceso_ppal =="test_exp":
    numero_proceso = n_experimento
    n_semillas = n_semillas_exp
elif proceso_ppal =="bayesiana" or proceso_ppal =="test_baye":
    numero_proceso = n_bayesiana
    n_semillas = n_semillas_bay
try:
    study_name = f"_COMP_{competencia}_{proceso_ppal}_{numero_proceso}"
except Exception as e:
    study_name = f"_COMP_{competencia}_{proceso_ppal}"
nombre_log=fecha+study_name
# CONFIGURACION LOG LOCAL
creacion_logg_local(nombre_log=nombre_log)
logger = logging.getLogger(__name__)

## --------------------------------------------------------Funcion main ------------------------------------------

def main():
    
    logger.info(f"Inicio de ejecucion del flujo : {nombre_log}")
    try:
        semillas = create_semilla(n_semillas=n_semillas)
        logger.info(f"se crearon {len(semillas)} semillas")
    except Exception as e:
        logger.error(f"No es un error, pero no  se crearon semillas porque estamos en el proceso {proceso_ppal}. : {e}")

    # CONFIGURACION LOG GLOBAL
    try:
        creacion_logg_global(fecha=fecha , competencia=competencia ,proceso_ppal=proceso_ppal,n_experimento=n_experimento,n_semillas=n_semillas)
    except:
        creacion_logg_global(fecha=fecha , competencia=competencia ,proceso_ppal=proceso_ppal,n_experimento=n_experimento,n_semillas=[])

    if proceso_ppal =="creacion_target_clase_ternaria":
        lanzar_creacion_clase_ternaria_binaria_peso()
    elif proceso_ppal=="analisis_exploratorio":
        lanzar_eda(competencia=competencia)
    elif proceso_ppal =="feat_eng":
        lanzar_feat_eng(fecha,n_fe ,proceso_ppal)
    elif proceso_ppal =="bayesiana":
        lanzar_bayesiana_lgbm(fecha,semillas,n_bayesiana,proceso_ppal)
        # lanzar_bayesiana_xgb(fecha,semillas,proceso_ppal)
    elif proceso_ppal =="test_baye":
        lanzar_bayesiana_lgbm(test,semillas,n_bayesiana,proceso_ppal)
        # lanzar_bayesiana_xgb(test,semillas,proceso_ppal)
    elif proceso_ppal =="test_exp":
        lanzar_experimento(test,semillas , n_experimento , proceso_ppal)
    elif proceso_ppal =="test_prediccion_final":
        lanzar_experimento(test,semillas , n_experimento , proceso_ppal)
    elif (proceso_ppal =="experimento") | (proceso_ppal=="prediccion_final"):
        lanzar_experimento(fecha,semillas , n_experimento , proceso_ppal)
    return

if __name__ =="__main__":
    main()