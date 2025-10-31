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
from src.generadora_semillas import create_semilla

from src_bayesianas.experimento_bayesiana_lgbm_2 import lanzar_bayesiana_lgbm
from src_bayesianas.experimento_bayesiana_xgb_2 import lanzar_bayesiana_xgb

from src_experimentos.experimento_7 import lanzar_experimento_7
from src_experimentos.experimento_8 import lanzar_experimento_8
from src_experimentos.experimento_9 import lanzar_experimento_9


## ---------------------------------------------------------Configuraciones Iniciales -------------------------------


## Carga de variables
n_trials=N_TRIALS

## Creacion de las carpetas
        #LOGS PATHS
os.makedirs(PATH_LOGS,exist_ok=True)
        #OUTPUT PATHS
os.makedirs(PATH_OUTPUT_BAYESIAN,exist_ok=True)
os.makedirs(PATH_OUTPUT_FINALES,exist_ok=True)
os.makedirs(PATH_OUTPUT_EXPERIMENTOS,exist_ok=True)
        #BAYESIANA
os.makedirs(path_output_bayesian_db,exist_ok=True)
os.makedirs(path_output_bayesian_bestparams,exist_ok=True)
os.makedirs(path_output_bayesian_best_iter,exist_ok=True)
os.makedirs(path_output_bayesian_graf,exist_ok=True)
        #FINALES
os.makedirs(path_output_finales_model,exist_ok=True)
os.makedirs(path_output_finales_feat_imp,exist_ok=True)
os.makedirs(path_output_prediccion_final,exist_ok=True)
        #EXPERIMENTOS
os.makedirs(path_output_exp_model,exist_ok=True)
os.makedirs(path_output_exp_feat_imp,exist_ok=True)
os.makedirs(path_output_exp_graf_gan_hist_grilla,exist_ok=True)
os.makedirs(path_output_exp_graf_gan_hist_total,exist_ok=True)
os.makedirs(path_output_exp_graf_gan_hist_semillas,exist_ok=True)
os.makedirs(path_output_exp_graf_curva_ganancia,exist_ok=True)
os.makedirs(path_output_exp_umbral,exist_ok=True)



fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# ----------------------------Cambiar numero y proceso ppal -----------------------------------------------------------------
n_experimento=9
proceso_ppal="prediccion_final"
# ---------------------------------------------------------------------------------------------------------------------------
test= "TEST_TEST_TEST_TEST"
# comentario=input(f"Ingrese un comentario")
nombre_log=fecha+f"_{proceso_ppal}_{n_experimento}"
# CONFIGURACION LOG
logging.basicConfig(
    level=logging.INFO, #Puede ser INFO o ERROR
    format='%(asctime)s - %(levelname)s - %(name)s  - %(funcName)s -  %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(f"{PATH_LOGS}/{nombre_log}", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

## --------------------------------------------------------Funcion main ------------------------------------------

def main():
    
    logger.info(f"Inicio de ejecucion del flujo : {nombre_log}")
    semillas = create_semilla(15)
    logger.info(f"se crearon {len(semillas)} semillas")
    lanzar_experimento_9(fecha , semillas,n_experimento ,proceso_ppal )

    # lanzar_bayesiana_lgbm(fecha , SEMILLA)
    # lanzar_experimento_9(test,semillas[:3],n_experimento,proceso_ppal)
    # lanzar_experimento_9(fecha,semillas,n_experimento,proceso_ppal)
    # lanzar_experimento_9(test,semillas[:3],n_experimento,'prediccion_final')
    # lanzar_bayesiana_lgbm(fecha,SEMILLA)
    # lanzar_bayesiana_xgb(fecha,SEMILLA)
    # lanzar_experimento_8(fecha,semillas,'prediccion_final')
    # lanzar_experimento_8(fecha,semillas,'experimento')
    # lanzar_experimento_8(test,semillas[:3] ,'prediccion_final')
    # lanzar_experimento_8(test,semillas[:3] ,'experimento')
    # lanzar_experimento_7(fecha,semillas[:5] ,'prediccion_final')
    # lanzar_experimento_7(fecha,semillas[:5] ,'experimento')
    # lanzar_bayesiana_lgbm(test,SEMILLA)
    # lanzar_bayesiana_xgb(test,SEMILLA)


    # lanzar_experimento_i(fecha ,semillas , "experimento" )

    # lanzar_experimento_7(fecha ,semillas , "prediccion_final" )
    # lanzar_experimento_10(fecha ,semillas , "prediccion_final" )
    # lanzar_experimento_9(fecha ,semillas , "prediccion_final" )


    # 
   
    # lanzar_bayesiana_xgb(fecha , SEMILLA)
    # lanzar_bayesiana_lgbm(fecha , SEMILLA)
    # lanzar_experimento_8(fecha ,semillas , "prediccion_final" )
    # lanzar_experimento_7(fecha ,semillas , "prediccion_final" )
    # lanzar_experimento_6(fecha ,semillas , "prediccion_final" )
    # lanzar_experimento_6(fecha ,semillas , "experimento" )
    # lanzar_experimento_5(fecha ,semillas , "prediccion_final" )
    # lanzar_experimento_5(fecha ,semillas , "experimento" )
    # lanzar_experimento_5(test ,semillas[:3] , "prediccion_final" )
#     lanzar_experimento_5(test ,semillas[:2] , "experimento" )
    #lanzar_bayesiana(fecha , SEMILLA)
    
#     lanzar_experimento_4(fecha , [semillas[0]],"prediccion_final")
#     lanzar_experimento_4(fecha , semillas,"experimento")
#     lanzar_experimento_ensamble(fecha , semillas,"experimento")
#     lanzar_experimento_2(fecha , semillas,"experimento")
        



#     lanzar_experimento_1(fecha , [SEMILLA],"experimento")

    return

if __name__ =="__main__":
    main()