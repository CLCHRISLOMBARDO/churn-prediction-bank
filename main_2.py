#main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime 
import logging
import json


from src.config import *
from src.loader import cargar_datos
from src.constr_lista_cols import contruccion_cols
from src.feature_engineering import feature_engineering_delta, feature_engineering_lag , feature_engineering_ratio,feature_engineering_linreg,feature_engineering_max_min
from src.preprocesamiento import split_train_binario, submuestreo, imputacion
from src.optimizacion_rf import optim_hiperp_binaria 
from src.random_forests import  entrenamiento_rf,distanceMatrix
from src.embedding import embedding_umap
from src.cluster import clustering_kmeans ,cluster_distribution,score_cluster

## ---------------------------------------------------------Configuraciones Iniciales -------------------------------
## PATH
path_input_data = PATH_INPUT_DATA
path_output_data=PATH_OUTPUT_DATA
path_output_optim = PATH_OUTPUT_OPTIMIZACION
db_path = path_output_optim + 'db/'
bestparms_path = path_output_optim+'best_params/'
path_output_umap=PATH_OUTPUT_UMAP

path_output_segmentacion = PATH_OUTPUT_SEGMENTACION


## Carga de variables
n_subsample=N_SUBSAMPLE
n_completo=N_COMPLETO
mes_train = MES_TRAIN_SEGM
n_trials=N_TRIALS

## config basic logging
os.makedirs("logss",exist_ok=True)

fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
nombre_log = f"log_{fecha}.log"

logging.basicConfig(
    level=logging.INFO, #Puede ser INFO o ERROR
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(f"logss/{nombre_log}", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

## --------------------------------------------------------Funcion main ------------------------------------------

def main():
    logger.info("Inicio de ejecucion main_2.")

    ## 0. load datos
    df=cargar_datos(path_input_data)
    print(df.head())

    ## 1. Contruccion de las columnas
    columnas=contruccion_cols(df)
    cols_lag_delta_max_min_regl=columnas[0]
    cols_ratios=columnas[1]


    ## 2. Feature Engineering
    df=feature_engineering_lag(df,cols_lag_delta_max_min_regl,2)
    df=feature_engineering_delta(df,cols_lag_delta_max_min_regl,2)
    df=feature_engineering_max_min(df,cols_lag_delta_max_min_regl)
    df=feature_engineering_ratio(df,cols_ratios)
    df=feature_engineering_linreg(df,cols_lag_delta_max_min_regl)



    ## 2. Preprocesamiento para entrenamiento

    # split X_train, y_train
    X_train,y_train = split_train_binario(df,mes_train)
    # imputacion X_train
    X_train_imp = imputacion(X_train)
    # submuestreo
    X_train_imp,y_train = submuestreo(X_train_imp,y_train, n_completo)

    X_train_sample_imp ,y_train_sample = submuestreo(X_train_imp,y_train, n_subsample)

                # Guardo df
    try:
        X_train_sample_imp.to_csv(path_output_data + f"X_train_sample_imp_{fecha}.csv") 
        y_train_sample.to_csv(path_output_data + f"y_train_sample_{fecha}.csv") 
        logger.info(f"X shape {X_train_sample_imp.shape}, y shape{y_train_sample} guardado en csv")
    except Exception as e:
        logger.error(f"Error al guardar el df : {e}")
        raise


    ## 3. Optimizacion Hiperparametros y entrenamiento rf - Guardo las cosas en sus funciones respectivas, pero creo que tendria que hacerlo aca
    
    # b- Modelo completo
    name_rf_completo=f"_completo_{fecha}"
    with open(bestparms_path+"best_params_auc_binaria_completo_2025-09-20_10-14-27.json", "r") as f:
        best_params_completo = json.load(f)
    model_rf_completo=entrenamiento_rf(X_train_imp , y_train ,best_params_completo,name=name_rf_completo)
    class_index=np.where(model_rf_completo.classes_==1)[0][0]
    proba_baja_completo=model_rf_completo.predict_proba(X_train_sample_imp)[:,class_index] #Predigo solo el subsampleo que es el que voy a graficar
    distancia_con_completo_sampleado=distanceMatrix(model_rf_completo,X_train_sample_imp) # Calculo la dist solo con el subsampleo

    # 4. Embedding - UMAP
    embedding_comple = embedding_umap(distancia_con_completo_sampleado)

    # 5. Grafico del embedding coloreado por los predicts
    #b-completo
    plt.scatter(embedding_comple[:,0], embedding_comple[:,1], c=proba_baja_completo)
    plt.colorbar()
    file_image=f"embedding_umap{name_rf_completo}.png"
    plt.savefig(path_output_umap+file_image, dpi=300, bbox_inches="tight")
    plt.close()

    # 7. Clusters
    clusters=[5]
    embeddings = [embedding_comple]
    names_file = [name_rf_completo]
    names=["model_2_completo"]

    proba_bajas=[proba_baja_completo ]

    
    results_clusters_score = {}
    for emb,name_file , name , proba_baja in zip(embeddings,names_file,names,proba_bajas):
        results_clusters_score[name] = {}
        for k in clusters:
            cluster_i=clustering_kmeans(k, y_train_sample,emb,name_file)
            class_distribution_by_cluster = cluster_distribution(cluster_i  ,proba_baja )
            vandonngen_score=score_cluster(class_distribution_by_cluster)

            cluster_i.to_csv(path_output_segmentacion+name+"_"+f"cluster_{k}_{fecha}.csv")
            class_distribution_by_cluster.to_csv(path_output_segmentacion+name+"_"+f"cluster_distribution_{k}_{fecha}.csv")
            results_clusters_score[name][k] = vandonngen_score
         
    try:
        file_name = f"score_cluster_by_model_{fecha}.json"
        with open(PATH_OUTPUT_SEGMENTACION+file_name, "w", encoding="utf-8") as f:
            json.dump(results_clusters_score, f, indent=4, ensure_ascii=False)
        logger.info(f"json {file_name} de resultado de clusters guardados en {PATH_OUTPUT_SEGMENTACION}")
        logger.info("Fin del cluster")
    except Exception as e:
        logger.error(f"No se pudo guardar el resultado de los clusters por : {e}")


    logger.info(f">>> Ejecucion finalizada. Revisar logs para mas detalles. {nombre_log}")


if __name__ =="__main__":
    main()