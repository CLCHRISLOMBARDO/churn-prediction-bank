#random_forest.py
import pandas as pd
import numpy as np
import lightgbm as lgb


import logging
from time import time
import datetime

import pickle
import json

from src.config import PATH_OUTPUT_LGBM , SEMILLA ,GANANCIA , ESTIMULO

output_path = PATH_OUTPUT_LGBM
ganancia_acierto = GANANCIA
costo_estimulo=ESTIMULO

logger = logging.getLogger(__name__)



def entrenamiento_lgbm(X_train:pd.DataFrame ,y_train_binaria:pd.Series,w_train:pd.Series, best_iter:int, best_parameters:dict[str, object], name:str)->lgb.Booster:
    name=f"lgbm_model_{name}"
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
        filename=output_path+f'{name}.txt'
        model_lgbm.save_model(filename )                         
        logger.info(f"Modelo {name} guardado en {output_path}")
        logger.info("Fin del entrenamiento del LGBM")
    except Exception as e:
        logger.error(f"Error al intentar guardar el modelo {name}, por el error {e}")
        return
    return model_lgbm
    

def ganancia_prob(y_pred:pd.Series, y_true:pd.Series ,prop=1,threshold:int=0.025)->float:
    logger.info("comienzo funcion ganancia con threshold = 0.025")
    ganancia = np.where(y_true == 1, ganancia_acierto, 0) - np.where(y_true == 0, costo_estimulo, 0)
    return ganancia[y_pred >= threshold].sum() / prop

def evaluacion_lgbm(X_test:pd.DataFrame , y_test:pd.Series , model_lgbm:lgb.Booster)-> pd.Series:
    logger.info("comienzo evaluacion modelo")
    y_pred_lgm = model_lgbm.predict(X_test)
    ganancia = ganancia_prob(y_pred_lgm , y_test)
    logger.info("fin evaluacion modelo")
    print(f"ganancia:{ganancia}")

    return y_pred_lgm 
