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
from src_experimentos.experimento2 import lanzar_experimento_2
## ---------------------------------------------------------Configuraciones Iniciales -------------------------------


## Carga de variables
n_trials=N_TRIALS

## Creacion de las carpetas
        #LOGS PATHS
os.makedirs(PATH_LOGS,exist_ok=True)
        #OUTPUT PATHS
os.makedirs(PATH_OUTPUT_FINALES,exist_ok=True)
os.makedirs(PATH_OUTPUT_EXPERIMENTOS,exist_ok=True)
        #BAYESIANA
os.makedirs(PATH_OUTPUT_OPTIMIZACION,exist_ok=True)
os.makedirs(db_path,exist_ok=True)
os.makedirs(bestparams_path,exist_ok=True)
os.makedirs(best_iter_path,exist_ok=True)
os.makedirs(graf_bayesiana_path,exist_ok=True)
        #MODELS
os.makedirs(PATH_OUTPUT_LGBM,exist_ok=True)
os.makedirs(model_path,exist_ok=True)
os.makedirs(prediccion_final_path,exist_ok=True)
os.makedirs(graf_curva_ganancia_path,exist_ok=True)
os.makedirs(graf_hist_ganancia_grilla_path,exist_ok=True)
os.makedirs(graf_hist_ganancia_total_path,exist_ok=True)
os.makedirs(graf_hist_ganancia_semillas_path,exist_ok=True)
os.makedirs(umbrales_path,exist_ok=True)
os.makedirs(feat_imp_path,exist_ok=True)
        #EXPERIMENTOS
os.makedirs(path_output_exp_model,exist_ok=True)
os.makedirs(path_output_exp_feat_imp,exist_ok=True)
os.makedirs(path_output_exp_graf_gan_hist_grilla,exist_ok=True)
os.makedirs(path_output_exp_graf_gan_hist_total,exist_ok=True)
os.makedirs(path_output_exp_graf_gan_hist_semillas,exist_ok=True)
os.makedirs(path_output_exp_graf_curva_ganancia,exist_ok=True)
os.makedirs(path_output_exp_umbral,exist_ok=True)



fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
comentario=input(f"Ingrese un comentario")
nombre_log=fecha+"_experimento"
# CONFIGURACION LOG
logging.basicConfig(
    level=logging.INFO, #Puede ser INFO o ERROR
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(f"{PATH_LOGS}/{nombre_log}", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

## --------------------------------------------------------Funcion main ------------------------------------------

def main():
    
    logger.info(f"Inicio de ejecucion del flujo : {nombre_log}")
    lanzar_experimento_2(fecha , SEMILLAS[:3],"experimento")



#     lanzar_experimento_1(fecha , [SEMILLA],"experimento")

    return

    ## 5. Primer Entrenamiento lgbm con la mejor iteracion y los mejores hiperparametros en [01,02,03] y evaluamos en 04    
    name_1rst_train="1rst_train"
    model_lgbm = entrenamiento_lgbm(X_train , y_train_binaria,w_train ,best_iter,best_params , fecha,name_1rst_train)
    grafico_feature_importance(model_lgbm,X_train,name_1rst_train,fecha)
    y_pred_lgbm=prediccion_test_lgbm(X_test , y_test_binaria ,model_lgbm) 

    #Evaluacion umbral fijo
    name_umbral_fijo = name_1rst_train + "_umbral_fijo"
    umbral_fijo=UMBRAL
    ganancia = ganancia_prob_umbral_fijo(y_pred_lgbm , y_test_binaria)
    evaluacion_public_private(X_test , y_test_binaria,y_pred_lgbm ,umbral_fijo, name_umbral_fijo,fecha)

    #Evaluacion umbral movil
    name_umbral_movil=name_1rst_train + "_umbral_movil"
    umbrales= prediccion_lgbm_umbral_movil(X_test , y_test_binaria , y_test_class,model_lgbm,name_umbral_movil,fecha)
    umbral_optimo= umbrales["umbral_optimo"]
    ganancia = ganancia_prob_umbral_fijo(y_pred_lgbm , y_test_binaria,1,umbral_optimo)
    evaluacion_public_private(X_test , y_test_binaria,y_pred_lgbm ,umbral_optimo, name_umbral_movil,fecha)

    ## 6. FINAL TRAIN con mejores hiperp, mejor iter y mejor umbral
    name_final_train="final_train"
    # X_train_final= pd.concat([X_train , X_test],axis=0)
    # logger.info(f"meses en train {X_train_final['foto_mes'].unique()}")
    # logger.info(f"train shape {X_train_final.shape}")
    # y_train_binaria_final = pd.concat([y_train_binaria , y_test_binaria],axis=0)
    # w_train_final=pd.concat([w_train,w_test],axis=0)

    MES_TRAIN.append(MES_TEST)
    X_train_final, y_train_binaria_final,y_train_class_final, w_train_final, X_test, y_test_binaria, y_test_class, w_test,X_apred, y_apred = split_train_binario(df,MES_TRAIN,MES_TEST,MES_A_PREDECIR)
    # umbral_optimo=0.03083123681364618 
    umbral_optimo =0.025
    model_lgbm_final = entrenamiento_lgbm(X_train_final , y_train_binaria_final,w_train_final ,best_iter,best_params , fecha,name_final_train)
    grafico_feature_importance(model_lgbm_final,X_train_final,name_final_train,fecha)
    y_apred=X_apred[["numero_de_cliente"]]
    
    y_apred_final=prediccion_apred(X_apred ,y_apred,model_lgbm_final,umbral_optimo,fecha,comentario)
    
    logger.info(f">>> Ejecucion finalizada. Revisar logs para mas detalles. {nombre_log}")


if __name__ =="__main__":
    main()