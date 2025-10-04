#preprocesamiento.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from typing import Tuple
from src.config import SEMILLA
import logging
logger = logging.getLogger(__name__)

def split_train_binario(df:pd.DataFrame|np.ndarray , mes_train:list[int],mes_test:int,mes_apred:int) ->Tuple[pd.DataFrame,pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series, pd.Series,pd.DataFrame,pd.DataFrame]:
    logger.info("Creacion label binario")

    df['clase_peso'] = 1.0

    df.loc[df['clase_ternaria'] == 'BAJA+2', 'clase_peso'] = 1.00002
    df.loc[df['clase_ternaria'] == 'BAJA+1', 'clase_peso'] = 1.00001

    df['clase_binaria'] = 0
    df['clase_binaria'] = np.where(df['clase_ternaria'] == 'Continua', 0, 1)
    train_data = df[df['foto_mes'].isin(mes_train)]
    test_data = df[df['foto_mes'] == mes_test]
    apred_data = df[df['foto_mes'] == mes_apred]

    # TRAIN
    X_train = train_data.drop(['clase_ternaria', 'clase_peso', 'clase_binaria'], axis=1)
    y_train_binaria = train_data['clase_binaria']
    y_train_class=train_data["clase_ternaria"]
    w_train = train_data['clase_peso']

    # TEST
    X_test = test_data.drop(['clase_ternaria', 'clase_peso','clase_binaria'], axis=1)
    y_test_binaria = test_data['clase_binaria']
    y_test_class = test_data['clase_ternaria']
    w_test = test_data['clase_peso']

    # A PREDECIR
    X_apred = apred_data.drop(['clase_ternaria', 'clase_peso','clase_binaria'], axis=1)
    y_apred=X_apred[["numero_de_cliente"]] # DF
  

    logger.info(f"X_train shape : {X_train.shape} / y_train shape : {y_train_binaria.shape} de los meses : {X_train['foto_mes'].unique()}")
    logger.info(f"X_test shape : {X_test.shape} / y_test shape : {y_test_binaria.shape}  del mes : {X_test['foto_mes'].unique()}")
    logger.info(f"X_apred shape : {X_apred.shape} / y_apred shape : {y_apred.shape}  del mes : {X_apred['foto_mes'].unique()}")

    logger.info(f"cantidad de baja y continua en train:{np.unique(y_train_binaria,return_counts=True)}")
    logger.info(f"cantidad de baja y continua en test:{np.unique(y_test_binaria,return_counts=True)}")
    logger.info("Finalizacion label binario")
    return X_train, y_train_binaria,y_train_class, w_train, X_test, y_test_binaria, y_test_class, w_test ,X_apred , y_apred 



