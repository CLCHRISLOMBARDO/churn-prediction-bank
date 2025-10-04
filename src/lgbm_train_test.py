#random_forest.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import StratifiedShuffleSplit


import logging
from time import time
import datetime

import pickle
import json

from src.config import PATH_OUTPUT_LGBM , SEMILLA ,GANANCIA , ESTIMULO

ganancia_acierto = GANANCIA
costo_estimulo=ESTIMULO

output_path = PATH_OUTPUT_LGBM
model_path=PATH_OUTPUT_LGBM + 'model/'
prediccion_final_path = PATH_OUTPUT_LGBM + 'final_prediction/'
graf_train_path=PATH_OUTPUT_LGBM +'grafico_train/'
umbrales_path=PATH_OUTPUT_LGBM +'umbrales/'



logger = logging.getLogger(__name__)



def entrenamiento_lgbm(X_train:pd.DataFrame ,y_train_binaria:pd.Series,w_train:pd.Series, best_iter:int, best_parameters:dict[str, object], fecha:str,name:str)->lgb.Booster:
    # name es para identificar 1rt_train o final_train
    name=f"{fecha}_{name}_lgbm"
    logger.info(f"Comienzo del entrenamiento del lgbm : {name}")
        
    best_iter = best_iter
    print(f"Mejor cantidad de árboles para el mejor model {best_iter}")
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'first_metric_only': True,
        'boost_from_average': True,
        'feature_pre_filter': False,
        'max_bin': 31,
        'num_leaves': best_parameters['num_leaves'],
        'learning_rate': best_parameters['learning_rate'],
        'min_data_in_leaf': best_parameters['min_data_in_leaf'],
        'feature_fraction': best_parameters['feature_fraction'],
        'bagging_fraction': best_parameters['bagging_fraction'],
        'seed': SEMILLA,
        'verbose': 0
    }

    train_data = lgb.Dataset(X_train,
                            label=y_train_binaria,
                            weight=w_train)

    model_lgbm = lgb.train(params,
                    train_data,
                    num_boost_round=best_iter)


    try:
        filename=model_path+f'{name}.txt'
        model_lgbm.save_model(filename )                         
        logger.info(f"Modelo {name} guardado en {filename}")
        logger.info("Fin del entrenamiento del LGBM")
    except Exception as e:
        logger.error(f"Error al intentar guardar el modelo {name}, por el error {e}")
        return
    return model_lgbm
    

def ganancia_prob_umbral_fijo(y_pred:pd.Series, y_true:pd.Series ,prop=1,threshold:int=0.025)->float:
    # logger.info(f"comienzo funcion ganancia con threshold = {threshold}")
    ganancia = np.where(y_true == 1, ganancia_acierto, 0) - np.where(y_true == 0, costo_estimulo, 0)
    # logger.info(f"fin evaluacion modelo.")
    return ganancia[y_pred >= threshold].sum() / prop

def prediccion_test_lgbm(X_test:pd.DataFrame , y_test_binaria:pd.Series , model_lgbm:lgb.Booster)-> pd.Series:
    logger.info("comienzo prediccion en test del modelo")
    y_pred_lgbm = model_lgbm.predict(X_test)
    logger.info("Fin de la prediccion en test del modelo")
    return y_pred_lgbm


def prediccion_lgbm_umbral_movil(X_test:pd.DataFrame , y_test_binaria:pd.Series,y_test_class:pd.Series , model_lgbm:lgb.Booster,name:str,fecha:str)->dict:
    name=f"{fecha}_{name}_lgbm"
    
    logger.info("Comienzo de la prediccion con umbral movil")


    piso=4000
    techo=20000
    

    y_pred_lgm = model_lgbm.predict(X_test)
    ganancia = np.where(y_test_class=="BAJA+2" , ganancia_acierto,0) - np.where(y_test_class!="BAJA+2" , costo_estimulo,0)
    print(ganancia) # BORRAR ----------------------------------------------------------------

    try:
        idx_sorted = np.argsort(y_pred_lgm)[::-1]
        y_pred_sorted = y_pred_lgm[idx_sorted]

        ganancia_sorted = ganancia[idx_sorted]
        ganancia_acumulada=np.cumsum(ganancia_sorted)

        max_ganancia_acumulada = np.max(ganancia_acumulada)

        indx_max_ganancia_acumulada = np.where(ganancia_acumulada ==max_ganancia_acumulada)[0][0]

        umbral_optimo = y_pred_sorted[indx_max_ganancia_acumulada]
    except Exception as e:
        logger.error(f"Hubo un error por {e}")
        raise
    logger.info("Comienzo de los graficos de curvas de ganancia")
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(y_pred_sorted[piso:techo] ,ganancia_acumulada[piso:techo] ,label="Ganancia LGBM")
        plt.xlabel('Predicción de probabilidad')
        plt.ylabel('Ganancia')
        plt.title("Curva Ganancia respecto a probabilidad")
        plt.axvline(x=0.025 , color="green" , linestyle="--" ,label="Punto de Corte a 0.025")
        plt.axvline(x=umbral_optimo , color="red" , linestyle="--" ,label="Punto de Corte optimo")
        plt.legend()
        plt.savefig(graf_train_path+f"{name}_grafico_umbral_optimo_probabilidad.png", bbox_inches='tight')
        logger.info("Creacion de los graficos Curva Ganancia respecto a probabilidad")
    except Exception as e:
        logger.error(f"Error al tratar de crear los graficos Curva Ganancia respecto a probabilidad --> {e}")


    try:
        plt.figure(figsize=(10, 6))
        plt.plot(range(piso,len(ganancia_acumulada[piso:techo])+piso) ,ganancia_acumulada[piso:techo] ,label="Ganancia LGBM")
        plt.xlabel('Clientes')
        plt.ylabel('Ganancia')
        plt.title("Curva Ganancia con numero de clientes")
        plt.axvline(x=indx_max_ganancia_acumulada , color="red" , linestyle="--" ,label="Punto de Corte optimo")
        plt.axhline(y=max_ganancia_acumulada , color="red",linestyle="--" ,label="Ganancia Acumulada Maxima" )
        plt.legend()
        plt.savefig(graf_train_path+f"{name}_grafico_umbral_optimo_numero_cliente.png", bbox_inches='tight')
        logger.info("Creacion de los graficos Curva Ganancia respecto al cliente")
    except Exception as e:
        logger.error(f"Error al tratar de crear los graficos Curva Ganancia respecto a probabilidad --> {e}")

    logger.info(f"Umbral_prob optimo = {umbral_optimo}")
    logger.info(f"Numero de cliente optimo : {indx_max_ganancia_acumulada}")
    logger.info(f"Ganancia maxima con el punto optimo : {max_ganancia_acumulada}")
    umbrales = {
    "umbral_optimo": float(umbral_optimo),
    "cliente": int(indx_max_ganancia_acumulada),
    "ganancia_max": float(max_ganancia_acumulada)
    }
    try:
        with open(umbrales_path+f"{name}_umbral.json", "w") as f:
            json.dump(umbrales, f, indent=4)
    except Exception as e:
        logger.error(f"Error al intentar guardar el dict de umbral como json --> {e}")
    logger.info(f"Los datos de umbrales moviles son : {umbrales}")
    logger.info("Fin de la prediccion de umbral movil")

    return umbrales 


def evaluacion_public_private(X_test:pd.DataFrame , y_test_binaria:pd.Series , y_pred_model:pd.Series,umbral:str,name:str,fecha:str ):
    logger.info("Comienzo de los histogramas de public and private")
    name=f"{fecha}_{name}_lgbm"
    sss=StratifiedShuffleSplit(n_splits=50,test_size=0.3,random_state=SEMILLA)
    modelos={"lgbm":y_pred_model}
    rows=[]
    for private_index , public_index in sss.split(X_test , y_test_binaria):
        row={}
        for name , y_pred in modelos.items():
            y_true_private = y_test_binaria.iloc[private_index]
            y_pred_private = y_pred[private_index]
            y_true_public = y_test_binaria.iloc[public_index]
            y_pred_public = y_pred[public_index]

            row[name+"_public"] = ganancia_prob_umbral_fijo(y_pred_public, y_true_public, 0.3,umbral)
            row[name+"_private"] =ganancia_prob_umbral_fijo(y_pred_private, y_true_private, 0.3,umbral)

        rows.append(row)

    df_lb = pd.DataFrame(rows)
    df_lb_long = df_lb.reset_index()
    df_lb_long = df_lb_long.melt(id_vars=['index'], var_name='model_type', value_name='ganancia')
    df_lb_long[['modelo', 'tipo']] = df_lb_long['model_type'].str.split('_', expand=True)
    df_lb_long = df_lb_long[['ganancia', 'tipo', 'modelo']]
    logger.info("Comienzo del grafico de los histogramas")
    try:
        g = sns.FacetGrid(df_lb_long, col="tipo", row="modelo", aspect=2)
        g.map(sns.histplot, "ganancia", kde=True)
        plt.savefig(graf_train_path+f"{fecha}_grafico_ganancia_hist.png", bbox_inches='tight')
    except Exception as e:
        logger.error(f"Error al intentar hacer el grafico de los histogramas {e}")

    logger.info("Fin de los histogramas de public and private")


def grafico_feature_importance(model_lgbm:lgb.Booster,name:str,fecha:str,threshold:int=0.025):
    logger.info("Comienzo del grafico de feature importance")
    name=f"{fecha}_{name}_lgbm"
    try:
        lgb.plot_importance(model_lgbm, figsize=(10, 20))
        plt.savefig(graf_train_path+f"{name}_grafico_feature_importance_plot.png", bbox_inches='tight')
    except Exception as e:
        logger.error(f"Error al intentar graficar los feat importances: {e}")
    logger.info("Fin del grafico de feature importance")



def prediccion_apred(X_apred:pd.DataFrame , y_apred:pd.DataFrame , model_lgbm:lgb.Booster, umbral:float,fecha:str)->pd.DataFrame:
    name=fecha+"_predicciones"
    logger.info(f"Comienzo de las predicciones del mes {X_apred['foto_mes'].unique()} ")
    y_pred=model_lgbm.predict(X_apred)
    y_apred["prediction"] = y_pred
    y_apred["prediction"]=y_apred["prediction"].apply(lambda x : 1 if x >= umbral else 0)
    logger.info(f"cantidad de bajas predichas : {(y_apred["prediction"]==1).sum()}")
    y_apred=y_apred.set_index("numero_de_cliente")
    file_name=prediccion_final_path+name+".csv"
    try:
        y_apred.to_csv(file_name)
        logger.info(f"predicciones guardadas en {file_name}")
    except Exception as e:
        logger.error(f"Error al intentar guardar las predicciones --> {e}")
        raise

    return y_apred

    