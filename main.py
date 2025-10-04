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
import sys # Eliminar despues

from src.config import *
from src.loader import cargar_datos
from src.constr_lista_cols import contruccion_cols
from src.feature_engineering import feature_engineering_delta, feature_engineering_lag , feature_engineering_ratio,feature_engineering_linreg,feature_engineering_max_min
from src.preprocesamiento import split_train_binario
from src.lgbm_optimizacion import optim_hiperp_binaria , graficos_bayesiana
from src.lgbm_train_test import  entrenamiento_lgbm , prediccion_test_lgbm,ganancia_prob_umbral_fijo,grafico_feature_importance ,evaluacion_public_private,prediccion_lgbm_umbral_movil,prediccion_apred
## ---------------------------------------------------------Configuraciones Iniciales -------------------------------
## PATH

db_path = PATH_OUTPUT_OPTIMIZACION + 'db/'
bestparams_path = PATH_OUTPUT_OPTIMIZACION+'best_params/'
best_iter_path = PATH_OUTPUT_OPTIMIZACION+'best_iters/'
graf_bayesiana_path = PATH_OUTPUT_OPTIMIZACION+'grafico_bayesiana/'

model_path=PATH_OUTPUT_LGBM + 'model/'
prediccion_final_path = PATH_OUTPUT_LGBM + 'final_prediction/'
graf_train_path=PATH_OUTPUT_LGBM +'grafico_train/'
umbrales_path=PATH_OUTPUT_LGBM +'umbrales/'

## Carga de variables
n_trials=N_TRIALS

## config basic logging
os.makedirs("logs",exist_ok=True)
os.makedirs(PATH_OUTPUT_OPTIMIZACION,exist_ok=True)
os.makedirs(db_path,exist_ok=True)
os.makedirs(bestparams_path,exist_ok=True)
os.makedirs(best_iter_path,exist_ok=True)
os.makedirs(graf_bayesiana_path,exist_ok=True)

os.makedirs(PATH_OUTPUT_LGBM,exist_ok=True)
os.makedirs(model_path,exist_ok=True)
os.makedirs(prediccion_final_path,exist_ok=True)
os.makedirs(graf_train_path,exist_ok=True)
os.makedirs(umbrales_path,exist_ok=True)



# os.makedirs(PATH_NOTAS,exist_ok=True ) No me gusto, ya suficiente con los logs

fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# DEFINIR EL NOMBRE DE LOG DEL FLUJO BAYESIANA O ENTRENAMIENTO DIRECTO
respuesta_correcta=0
while respuesta_correcta ==0:
    pedido = input(f"Ingrese si quiere:\na) Optimizacion Bayesiana\nb) Entrenamiento directo\n")
    pedido=pedido.lower()
    if pedido=="a":
        nombre_log = f"log_{fecha}_opt_hp.log"
        respuesta_correcta=1
    elif pedido =="b":
        nombre_log = f"log_{fecha}_entrenamiento_directo.log"
        respuesta_correcta=1
    else:
        print("Ingrese una opcion valida 'a' o 'b'\n")

# CONFIGURACION LOG
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
    logger.info(f"Inicio de ejecucion del flujo : {nombre_log}")

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
    # # df=feature_engineering_max_min(df,cols_lag_delta_max_min_regl)
    df=feature_engineering_ratio(df,cols_ratios)
    # # df=feature_engineering_linreg(df,cols_lag_delta_max_min_regl)




# # ----------------------------------------------------------------------------------------------------------
    ## 3. Preprocesamiento para entrenamiento
    # split X_train, y_train
    X_train, y_train_binaria,y_train_class, w_train, X_test, y_test_binaria, y_test_class, w_test,X_apred, y_apred = split_train_binario(df,MES_TRAIN,MES_TEST,MES_A_PREDECIR)
                # Guardo df
    # try:
    #     X_train.to_csv(path_output_data + "X_train_sample_imp.csv") 
    #     y_train.to_csv(path_output_data + "y_train_sample.csv") 
    #     logger.info(f"X shape {X_train.shape}, y shape{y_train} guardado en csv")
    # except Exception as e:
    #     logger.error(f"Error al guardar el df : {e}")
    #     raise
    # name_lgbm=f"{fecha}"

    
    
    ## 4. Carga de mejores Hiperparametros

    ## 4.a. Optimizacion Hiperparametros
    if pedido =="a":
        study = optim_hiperp_binaria(X_train , y_train_binaria,w_train ,n_trials , name=fecha)
        graficos_bayesiana(study , fecha)
        best_iter = study.best_trial.user_attrs["best_iter"]
        best_params = study.best_trial.params
        logger.info("Best params y best iter cargados")


    ## 4.b. Ingreso de hiperparametros
    elif pedido =="b":
        logger.info("Ingreso de hiperparametros de una Bayesiana ya realizada")
         
        respuesta_correcta=0
        while respuesta_correcta==0:
            modo_incrustacion_hiperparametros=input("""Ingrese forma de obtener los hiperparametros:\n
                a) A partir de una optimizacion Bayesiana\n
                b) Manualmente\n """)
            try:
                modo_incrustacion_hiperparametros=modo_incrustacion_hiperparametros.lower()
            except Exception as e:
                logger.error(f"Error porque se incrusto un modo_incrustacion_hiperparametros que no es string :{e}")
            if modo_incrustacion_hiperparametros == "a":
                
                bayesiana_fecha=input("Ingrese fecha de la bayesiana yyyy-mm-dd: ")
                bayesiana_hora=input("Ingrese fecha de la bayesiana hh-mm-ss: ")
                bayesiana_fecha_hora= bayesiana_fecha +'_'+bayesiana_hora

                name_best_params_file=f"best_paramsbinaria_{bayesiana_fecha_hora}.json"
                name_best_iter_file=f"best_iter_binaria_{bayesiana_fecha_hora}.json"

                try:
                    with open(bestparams_path+name_best_params_file, "r") as f:
                        best_params = json.load(f)
                        logger.info(f"Correcta carga de los best params : {best_params}")

                    with open(best_iter_path+name_best_iter_file, "r") as f:
                        best_iter = json.load(f)
                        logger.info(f"Correcta carga de la best iter : {best_iter}")
                    respuesta_correcta=1
                except Exception as e:
                    logger.error(f"No se pudo encontrar los best params ni best iter por el error {e}")
                    raise
            elif modo_incrustacion_hiperparametros == "b":
                logger.info("Aun no se realizo la construccion manual de los hiperparametros. PRONTO")
                return

    ## 5. Primer Entrenamiento lgbm con la mejor iteracion y los mejores hiperparametros en [01,02,03] y evaluamos en 04    
    name_1rst_train="1rst_train"
    model_lgbm = entrenamiento_lgbm(X_train , y_train_binaria,w_train ,best_iter,best_params , fecha,name_1rst_train)
    grafico_feature_importance(model_lgbm,name_1rst_train,fecha)
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
    X_train_final= pd.concat([X_train , X_test],axis=0)
    y_train_binaria_final = pd.concat([y_train_binaria , y_test_binaria],axis=0)
    w_train_final=pd.concat([w_train,w_test],axis=0)
    model_lgbm_final = entrenamiento_lgbm(X_train_final , y_train_binaria_final,w_train_final ,best_iter,best_params , fecha,name_final_train)
    grafico_feature_importance(model_lgbm_final,name_final_train,fecha)
    y_apred_final=prediccion_apred(X_apred ,y_apred,model_lgbm_final,umbral_optimo,fecha)

    logger.info(f">>> Ejecucion finalizada. Revisar logs para mas detalles. {nombre_log}")


if __name__ =="__main__":
    main()