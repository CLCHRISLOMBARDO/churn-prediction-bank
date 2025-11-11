#lgbm_train_test.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
import xgboost as xgb

import logging
from time import time
import datetime

import pickle
import json
import os

from src.config import GANANCIA,ESTIMULO

ganancia_acierto = GANANCIA
costo_estimulo=ESTIMULO

logger = logging.getLogger(__name__)

def entrenamiento_xgb(X_train: pd.DataFrame,y_train_binaria: pd.Series,w_train: pd.Series,best_iter: int,best_parameters: dict[str, object],name: str,output_path: str,semilla: int) -> xgb.Booster:
    name = f"{name}_model_XGB"
    logger.info(f"Comienzo del entrenamiento del XGB : {name} en el mes train : {X_train['foto_mes'].unique()}")
    logger.info(f"Mejor cantidad de árboles para el mejor model {best_iter}")
    
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "seed": int(semilla),
        "verbosity": 0,
        **best_parameters
    }

    # 2) Pasá feature_names al DMatrix
    dtrain = xgb.DMatrix(
        data=X_train,
        label=y_train_binaria,
        weight=w_train)

    model_xgb = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(best_iter)
    )

    logger.info(f"comienzo del guardado en {output_path}")
    try:
        filename = output_path + f"{name}.txt"
        model_xgb.save_model(filename)
        logger.info(f"Modelo {name} guardado en {filename}")
        logger.info(f"Fin del entrenamiento del XGB en el mes train : {X_train['foto_mes'].unique()}")
    except Exception as e:
        logger.error(f"Error al intentar guardar el modelo {name}, por el error {e}")
        return
    return model_xgb



def grafico_feature_importance(model_xgb,X_train:pd.DataFrame,name:str,output_path:str):
    logger.info("Comienzo del grafico de feature importance")
    name=f"{name}_feature_importance"
    try:
        xgb.plot_importance(model_xgb, figsize=(10, 20))
        plt.savefig(output_path+f"{name}_grafico.png", bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"Error al intentar graficar los feat importances: {e}")
    logger.info("Fin del grafico de feature importance")

    importances = model_xgb.feature_importance()
    feature_names = X_train.columns.tolist()
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    importance_df = importance_df.sort_values('importance', ascending=False)
    importance_df["importance_%"] = (importance_df["importance"] /importance_df["importance"].sum())*100
    # importance_df[importance_df['importance'] > 0]
    logger.info("Guardado de feat import en excel")
    try :
        importance_df.to_excel(output_path+f"{name}_data_frame.xlsx" ,index=False)
        logger.info("Guardado feat imp en excel con EXITO")
    except Exception as e:
        logger.error(f"Error al intentar guardar los feat imp en excel por {e}")

def prediccion_test_xgb(X:pd.DataFrame ,  model_xgb: xgb.Booster)-> pd.Series:
    mes=X["foto_mes"].unique()
    logger.info(f"comienzo prediccion del modelo en el mes {mes}")
    y_pred_xgb = model_xgb.predict(X)
    logger.info("Fin de la prediccion del modelo")
    return y_pred_xgb


# Los calculos de curvas de ganancias e histograma estan en el lgbm
