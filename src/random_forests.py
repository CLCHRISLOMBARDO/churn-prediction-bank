#random_forest.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import logging
from time import time
import datetime

import pickle
import json

from src.config import PATH_OUTPUT_RF ,N_ESTIMATORS, SEMILLA

output_path = PATH_OUTPUT_RF
n_estimators = N_ESTIMATORS

logger = logging.getLogger(__name__)



def entrenamiento_rf(X:pd.DataFrame|np.ndarray ,y:pd.Series|np.ndarray , best_parameters:dict[str, object], name:str)->RandomForestClassifier:
    name=f"rf_model_{name}"
    logger.info(f"Comienzo del entrenamiento del rf : {name}")
    
    model_rf = RandomForestClassifier(
        n_estimators=n_estimators,
        #**study.best_params,
        **best_parameters,
        max_samples=0.7,
        random_state=SEMILLA,
        n_jobs=12,
        oob_score=True )
    model_rf.fit(X,y)
    try:
        filename=output_path+f'{name}.sav'
        pickle.dump(model_rf, open(filename, 'wb'))
        logger.info(f"Modelo {name} guardado en {output_path}")
        logger.info("Fin del entrenamiento del random forest")
    except Exception as e:
        logger.error(f"Error al intentar guardar el modelo {name}, por el error {e}")
        return
    return model_rf
    

def distanceMatrix(model:RandomForestClassifier, X:pd.DataFrame|np.ndarray)->np.ndarray:
    logger.info(f"Comienzo del calculo de las distancias")
    terminals = model.apply(X)
    nTrees = terminals.shape[1]

    a = terminals[:,0]
    proxMat = 1*np.equal.outer(a, a)

    for i in range(1, nTrees):
        a = terminals[:,i]
        proxMat += 1*np.equal.outer(a, a)
    proxMat = proxMat / nTrees
    logger.info(f"Fin del calculo de las distancias")
    return proxMat.max() - proxMat
