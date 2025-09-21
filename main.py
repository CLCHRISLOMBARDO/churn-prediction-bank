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
print("ya cargo todo")
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
os.makedirs("logs",exist_ok=True)

fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
nombre_log = f"log_{fecha}.log"

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
    logger.info("Inicio de ejecucion.")

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
        X_train_sample_imp.to_csv(path_output_data + "X_train_sample_imp.csv") 
        y_train_sample.to_csv(path_output_data + "y_train_sample.csv") 
        logger.info(f"X shape {X_train_sample_imp.shape}, y shape{y_train_sample} guardado en csv")
    except Exception as e:
        logger.error(f"Error al guardar el df : {e}")
        raise


    ## 3. Optimizacion Hiperparametros y entrenamiento rf - Guardo las cosas en sus funciones respectivas, pero creo que tendria que hacerlo aca
    # a- Modelo sampleado
    name_rf_sample=f"_sampleado_{fecha}"
    study_rf_sample = optim_hiperp_binaria(X_train_sample_imp , y_train_sample ,n_trials , name=name_rf_sample)
    best_params_sample=study_rf_sample.best_params
    model_rf_sample=entrenamiento_rf(X_train_sample_imp , y_train_sample ,best_params_sample,name=name_rf_sample)
    class_index = np.where(model_rf_sample.classes_ == 1)[0][0]
    proba_baja_sample=model_rf_sample.predict_proba(X_train_sample_imp)[:,class_index]
    distancia_sample = distanceMatrix(model_rf_sample,X_train_sample_imp)
    
    # b- Modelo completo
    name_rf_completo=f"_completo_{fecha}"
    study_rf_completo = optim_hiperp_binaria(X_train_imp , y_train , n_trials , name=name_rf_completo) 
    best_params_completo=study_rf_completo.best_params
    model_rf_completo=entrenamiento_rf(X_train_imp , y_train ,best_params_completo,name=name_rf_completo)
    class_index=np.where(model_rf_completo.classes_==1)[0][0]
    proba_baja_completo=model_rf_completo.predict_proba(X_train_sample_imp)[:,class_index] #Predigo solo el subsampleo que es el que voy a graficar
    distancia_con_completo_sampleado=distanceMatrix(model_rf_completo,X_train_sample_imp) # Calculo la dist solo con el subsampleo

    # 4. Embedding - UMAP
    embedding_sample=embedding_umap(distancia_sample)
    embedding_comple = embedding_umap(distancia_con_completo_sampleado)

    # 5. Grafico del embedding coloreado por los predicts
    #a- Sampleado
    plt.scatter(embedding_sample[:,0], embedding_sample[:,1], c=proba_baja_sample)
    plt.colorbar()
    file_image=f"embedding_umap{name_rf_sample}.png"
    plt.savefig(path_output_umap+file_image, dpi=300, bbox_inches="tight")
    plt.close()

    #b-completo
    plt.scatter(embedding_comple[:,0], embedding_comple[:,1], c=proba_baja_completo)
    plt.colorbar()
    file_image=f"embedding_umap{name_rf_completo}.png"
    plt.savefig(path_output_umap+file_image, dpi=300, bbox_inches="tight")
    plt.close()

    # 7. Clusters
    clusters=[4,5,6,7]
    embeddings = [embedding_sample,embedding_comple]
    names_file = [name_rf_sample , name_rf_completo]
    names=["model_1_sample","model_2_completo"]

    proba_bajas=[proba_baja_sample ,proba_baja_completo ]

    
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
            # results_clusters_by_model[name]["cluster_i"]=cluster_i
            # results_clusters_by_model[name]["cross_table"]=class_distribution_by_cluster
            # results_clusters_by_model[name]["score"]=vandonngen_score
            # results_clusters_by_model[name]["important_features_by_cluster"]={}
            # np.savetxt("array.csv", cluster_i, delimiter=",", fmt="%.4f")
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