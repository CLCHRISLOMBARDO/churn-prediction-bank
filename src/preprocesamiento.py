#preprocesamiento.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from typing import Tuple
from src.config import SEMILLA, PATH_DATA_BASE_DB
import logging
import duckdb
from src.config import SUBSAMPLEO
logger = logging.getLogger(__name__)


def split_train_test_apred(n_exp:int|str,mes_train:list[int],mes_test:list[int],mes_apred:int,semilla:int=SEMILLA,subsampleo:float=SUBSAMPLEO)->Tuple[pd.DataFrame,pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series, pd.Series,pd.DataFrame,pd.DataFrame]:
    logger.info("Comienzo del slpiteo de TRAIN - TEST - APRED")
    
    # TRAIN DATA
    sql_train = f"""
    WITH clientes_train AS (
        SELECT numero_de_cliente, clase_ternaria, foto_mes
        FROM df
        WHERE foto_mes IN {tuple(mes_train)}
    ),
    clientes_minoritarios AS (
        SELECT DISTINCT numero_de_cliente
        FROM clientes_train
        WHERE clase_ternaria <> 'Continua'
    ),
    clientes_mayoritarios AS (
        SELECT DISTINCT numero_de_cliente
        FROM clientes_train
        WHERE clase_ternaria = 'Continua'
    ),
    clientes_mayoritarios_sample AS (
        SELECT numero_de_cliente
        FROM clientes_mayoritarios
        USING SAMPLE {subsampleo}% (REPEATABLE ({semilla}))
    ),
    clientes_finales AS (
        SELECT numero_de_cliente FROM clientes_minoritarios
        UNION
        SELECT numero_de_cliente FROM clientes_mayoritarios_sample
    )
    SELECT df.*
    FROM df
    JOIN clientes_finales USING (numero_de_cliente)
    WHERE foto_mes IN {tuple(mes_train)};
    """
    sql_test=f"""select *
                from df
                where foto_mes IN {tuple(mes_test)}"""
    sql_apred=f"""select *
                from df
                where foto_mes = {mes_apred}"""
    conn=duckdb.connect(PATH_DATA_BASE_DB)
    train_data = conn.execute(sql_train).df()
    test_data = conn.execute(sql_test).df()
    apred_data = conn.execute(sql_apred).df()
    conn.close()
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



def split_train_test_apred_python(df:pd.DataFrame|np.ndarray , mes_train:list[int],mes_test:list[int],mes_apred:int,semilla:int=SEMILLA,subsampleo:float=None) ->Tuple[pd.DataFrame,pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series, pd.Series,pd.DataFrame,pd.DataFrame]:
    logger.info(f"mes train={mes_train}  -  mes test={mes_test} - mes apred={mes_apred} ")

    train_data = df[df['foto_mes'].isin(mes_train)]
    test_data = df[df['foto_mes'].isin(mes_test)]
    apred_data = df[df['foto_mes'] == mes_apred]

    if subsampleo is not None:
        train_data=undersampling(train_data , subsampleo,semilla)

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



def undersampling(df:pd.DataFrame ,undersampling_rate:float , semilla:int) -> pd.DataFrame:
    logger.info("Comienzo del subsampleo")
    np.random.seed(semilla)
    clientes_minoritaria = df.loc[df["clase_ternaria"] != "Continua", "numero_de_cliente"].unique()
    clientes_mayoritaria = df.loc[df["clase_ternaria"] == "Continua", "numero_de_cliente"].unique()

    logger.info(f"Clientes minoritarios: {len(clientes_minoritaria)}")
    logger.info(f"Clientes mayoritarios: {len(clientes_mayoritaria)}")

    n_sample = int(len(clientes_mayoritaria) * undersampling_rate)
    clientes_mayoritaria_sample = np.random.choice(clientes_mayoritaria, n_sample, replace=False)

    # Unimos los IDs seleccionados
    clientes_finales = np.concatenate([clientes_minoritaria, clientes_mayoritaria_sample])

    df_train_undersampled = df[df["numero_de_cliente"].isin(clientes_finales)].copy()

    logger.info(f"Shape original: {df.shape}")
    logger.info(f"Shape undersampled: {df_train_undersampled.shape}")

    df_train_undersampled = df_train_undersampled.sample(frac=1, random_state=semilla).reset_index(drop=True)
    return df_train_undersampled

