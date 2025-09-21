#preprocesamiento.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from typing import Tuple
from src.config import SEMILLA

import logging


logger = logging.getLogger(__name__)

def split_train_binario(df:pd.DataFrame|np.ndarray , mes_train:int) ->Tuple[pd.DataFrame|np.ndarray , pd.Series|np.ndarray]:
    logger.info("Creacion label binario")
    f = df["foto_mes"] == mes_train
    df = df.loc[f]
    X_train=df.drop(columns="clase_ternaria")
    y_train_ternaria = df["clase_ternaria"].copy()
    y_train=y_train_ternaria.map(lambda x : 0 if x =="Continua" else 1)
    logger.info(f"X_train shape : {X_train.shape} / y_train shape : {y_train.shape}")
    logger.info(f"cantidad de baja y continua:{np.unique(y_train,return_counts=True)}")
    logger.info("Finalizacion label binario")
    return X_train , y_train

def imputacion(X: pd.DataFrame|np.ndarray , strategy:str="median") -> pd.DataFrame:
    logger.info("Comienzo la imputacion en X_train")

    imputer=SimpleImputer(missing_values=np.nan , strategy=strategy)
    X_imp = imputer.fit_transform(X)
    X_imp=pd.DataFrame(X_imp , columns=X.columns , index=X.index)

    logger.info(f"nan en X_train : {X_imp.isna().sum().sum()}")
    logger.info("Finalizacion de la imputacion en X_train")
    return X_imp

def submuestreo(X_train:pd.DataFrame|np.ndarray , y_train:pd.DataFrame|np.ndarray , n_sample_continua:int) ->Tuple[pd.DataFrame|np.ndarray ,  pd.Series|np.ndarray]:
    logger.info("Comienzo de Submuestreo continua")
    np.random.seed(SEMILLA)
    continua_sample = y_train[y_train ==0].sample(n_sample_continua).index
    bajas_1_2 = y_train[y_train ==1].index
    rf_index = continua_sample.union(bajas_1_2)
    X_train_sample = X_train.loc[rf_index]
    y_train_sample = y_train.loc[rf_index]
    logger.info(f"X_train_sample shape : {X_train_sample.shape} / y_train_sample shape : {y_train_sample.shape}")
    logger.info("Finalizacion de Submuestreo continua")
    return X_train_sample , y_train_sample

