#main.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import json
import lightgbm as lgb

from src.config import *
from src.loader import cargar_datos
from src.preprocesamiento import conversion_binario,split_train_binario
from src.constr_lista_cols import contruccion_cols
from src.feature_engineering import feature_engineering_lag,feature_engineering_delta,feature_engineering_max_min,feature_engineering_ratio,feature_engineering_linreg,feature_engineering_normalizacion,feature_engineering_drop_cols

## ---------------------------------------------------------Configuraciones Iniciales -------------------------------
## Carga de variables

logger=logging.getLogger(__name__)

def lanzar_experimento_1(fecha:str ,semilla:int):
    name=f"{fecha}_EXPERIMENTO_1"
    logger.info(f"Comienzo del experimento 1 : {name}")

    ## 0. load datos
    df=cargar_datos(PATH_INPUT_DATA)
    print(df.head())

    ## 1. PREPROCESAMIENTO
        #a. Binarizacion
    df = conversion_binario(df)

        #b. Separacion 

    df_01 = df[~df["foto_mes"].isin( [MES_02 , MES_03])]
    df_02 = df[~df["foto_mes"].isin([MES_01 , MES_03])]
    df_03 = df[~df["foto_mes"].isin([MES_01 , MES_02])]
                            ## A - AGREGADO DE FEATURES

    ## 1. Contruccion de las columnas
    columnas=contruccion_cols(df)
    cols_lag_delta_max_min_regl=columnas[0]
    cols_ratios=columnas[1]

    # ## 2. Feature Engineering
                #01
    df_01=feature_engineering_ratio(df_01,cols_ratios)
                #02
    df_02=feature_engineering_lag(df_02,cols_lag_delta_max_min_regl,1)
    df_02=feature_engineering_delta(df_02,cols_lag_delta_max_min_regl,1)
    df_02=feature_engineering_ratio(df_02,cols_ratios)
                #03
    df_03=feature_engineering_lag(df_03,cols_lag_delta_max_min_regl,2)
    df_03=feature_engineering_delta(df_03,cols_lag_delta_max_min_regl,2)
    df_03=feature_engineering_ratio(df_03,cols_ratios)
    df_03=feature_engineering_linreg(df_03,cols_lag_delta_max_min_regl)



    #                             ## B - ELIMINACION DE FEATURES
    # ## 1. Contruccion de las columnas
    # feat_imp_file_name='2025-10-05_11-38-34_final_train_lgbm_data_frame_feat_imp.xlsx'
    # feat_imp_file=feat_imp_path+feat_imp_file_name
    # cols_dropear=contrs_cols_dropear_feat_imp(df,feat_imp_file,0.02)
    # ## 2. Feat engin
    # df=feature_engineering_drop_cols(df,cols_dropear)



    #3. spliteo train - test - apred
    X_train_01, y_train_binaria_01,y_train_class_01, w_train_01, X_test_para_01, y_test_binaria_para_01, y_test_class_para_01, w_test_para_01,X_apred_para_01, y_apred_para_01 = split_train_binario(df_01,[MES_01],MES_TEST,MES_A_PREDECIR)
    X_train_02, y_train_binaria_02,y_train_class_02, w_train_02,X_test_para_02, y_test_binaria_para_02, y_test_class_para_02, w_test_para_02,X_apred_para_02, y_apred_para_02= split_train_binario(df_02,[MES_02],MES_TEST,MES_A_PREDECIR)
    X_train_03, y_train_binaria_03,y_train_class_03, w_train_03, X_test_para_03, y_test_binaria_para_03, y_test_class_para_03, w_test_para_03,X_apred_para_03, y_apred_para_03 = split_train_binario(df_03,[MES_03],MES_TEST,MES_A_PREDECIR)


## 4. Carga de mejores Hiperparametros


    logger.info("Ingreso de hiperparametros de una Bayesiana ya realizada")
        
            
 
    bayesiana_fecha_hora= '2025-10-05_23-29-49'

    name_best_params_file=f"best_params_binaria_{bayesiana_fecha_hora}.json"
    name_best_iter_file=f"best_iter_binaria_{bayesiana_fecha_hora}.json"

    try:
        with open(bestparams_path+name_best_params_file, "r") as f:
            best_params = json.load(f)
            logger.info(f"Correcta carga de los best params : {best_params}")

        with open(best_iter_path+name_best_iter_file, "r") as f:
            best_iter = json.load(f)
            logger.info(f"Correcta carga de la best iter : {best_iter}")
    except Exception as e:
        logger.error(f"No se pudo encontrar los best params ni best iter por el error {e}")
        raise
    return 
# ## 5. Primer Entrenamiento lgbm con la mejor iteracion y los mejores hiperparametros en [01,02,03] y evaluamos en 04    
# name_1rst_train="1rst_train"
# model_lgbm = entrenamiento_lgbm(X_train , y_train_binaria,w_train ,best_iter,best_params , fecha,name_1rst_train)
# grafico_feature_importance(model_lgbm,X_train,name_1rst_train,fecha)
# y_pred_lgbm=prediccion_test_lgbm(X_test , y_test_binaria ,model_lgbm) 

# #Evaluacion umbral fijo
# name_umbral_fijo = name_1rst_train + "_umbral_fijo"
# umbral_fijo=UMBRAL
# ganancia = ganancia_prob_umbral_fijo(y_pred_lgbm , y_test_binaria)
# evaluacion_public_private(X_test , y_test_binaria,y_pred_lgbm ,umbral_fijo, name_umbral_fijo,fecha)

# #Evaluacion umbral movil
# name_umbral_movil=name_1rst_train + "_umbral_movil"
# umbrales= prediccion_lgbm_umbral_movil(X_test , y_test_binaria , y_test_class,model_lgbm,name_umbral_movil,fecha)
# umbral_optimo= umbrales["umbral_optimo"]
# ganancia = ganancia_prob_umbral_fijo(y_pred_lgbm , y_test_binaria,1,umbral_optimo)
# evaluacion_public_private(X_test , y_test_binaria,y_pred_lgbm ,umbral_optimo, name_umbral_movil,fecha)

# ## 6. FINAL TRAIN con mejores hiperp, mejor iter y mejor umbral
# name_final_train="final_train"
# # X_train_final= pd.concat([X_train , X_test],axis=0)
# # logger.info(f"meses en train {X_train_final['foto_mes'].unique()}")
# # logger.info(f"train shape {X_train_final.shape}")
# # y_train_binaria_final = pd.concat([y_train_binaria , y_test_binaria],axis=0)
# # w_train_final=pd.concat([w_train,w_test],axis=0)

# MES_TRAIN.append(MES_TEST)
# X_train_final, y_train_binaria_final,y_train_class_final, w_train_final, X_test, y_test_binaria, y_test_class, w_test,X_apred, y_apred = split_train_binario(df,MES_TRAIN,MES_TEST,MES_A_PREDECIR)
# # umbral_optimo=0.03083123681364618 
# umbral_optimo =0.025
# model_lgbm_final = entrenamiento_lgbm(X_train_final , y_train_binaria_final,w_train_final ,best_iter,best_params , fecha,name_final_train)
# grafico_feature_importance(model_lgbm_final,X_train_final,name_final_train,fecha)
# y_apred=X_apred[["numero_de_cliente"]]

# y_apred_final=prediccion_apred(X_apred ,y_apred,model_lgbm_final,umbral_optimo,fecha,comentario)

# logger.info(f">>> Ejecucion finalizada. Revisar logs para mas detalles. {nombre_log}")

