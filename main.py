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
from src.creacion_target import lanzar_creacion_clase_ternaria

from src_bayesianas.experimento_bayesiana_lgbm_2 import lanzar_bayesiana_lgbm
from src_bayesianas.experimento_bayesiana_xgb_2 import lanzar_bayesiana_xgb

from src_experimentos.experimento_10 import lanzar_experimento
from src_experimentos.experimento_eda import lanzar_eda



## ---------------------------------------------------------Configuraciones Iniciales -------------------------------


## Carga de variables
n_trials=N_TRIALS
creacion_directorios()
fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
test= "TEST_TEST_TEST_TEST"
# ---------------------------------------------------------------------------------------------
competencia = COMPETENCIA
proceso_ppal = PROCESO_PPAL
n_experimento = N_EXPERIMENTO
study_name = f"_COMP_{competencia}_{proceso_ppal}_{n_experimento}"
n_semillas = N_SEMILLAS
# ---------------------------------------------------------------------------------------------------------------------------
nombre_log=fecha+study_name
# CONFIGURACION LOG LOCAL
creacion_logg_local(nombre_log=nombre_log)
logger = logging.getLogger(__name__)

## --------------------------------------------------------Funcion main ------------------------------------------

def main():
    
    logger.info(f"Inicio de ejecucion del flujo : {nombre_log}")
    semillas = create_semilla(n_semillas=n_semillas)
    logger.info(f"se crearon {len(semillas)} semillas")
    # CONFIGURACION LOG GLOBAL
    creacion_logg_global(fecha=fecha , competencia=competencia ,proceso_ppal=proceso_ppal,n_experimento=n_experimento,n_semillas=n_semillas)

    if proceso_ppal =="creacion_target_clase_ternaria":
        lanzar_creacion_clase_ternaria()
    elif proceso_ppal=="analisis_exploratorio":
        lanzar_eda(competencia=competencia)
    elif proceso_ppal =="bayesiana":
        lanzar_bayesiana_lgbm(fecha,SEMILLA)
        lanzar_bayesiana_xgb(fecha,SEMILLA)
    elif proceso_ppal =="test":
        lanzar_experimento(test,semillas , n_experimento , proceso_ppal)
    elif (proceso_ppal =="experimento") | (proceso_ppal=="prediccion_final"):
        lanzar_experimento(fecha,semillas , n_experimento , proceso_ppal)
    return

if __name__ =="__main__":
    main()