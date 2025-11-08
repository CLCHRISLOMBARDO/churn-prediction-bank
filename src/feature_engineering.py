#feature_engineering.py
import pandas as pd
import numpy as np
import duckdb
import logging

#### FALTA AGREGAR LOS PUNTOS DE CONTROL PARA VISUALIZAR QUE ESTEN BIEN
from src.config import FILE_INPUT_DATA , PATH_DATA_BASE_DB
logger = logging.getLogger(__name__)


def feature_engineering_lag(df:pd.DataFrame ,columnas:list[str],cant_lag:int=1 ) -> pd.DataFrame:
    """
    Genera variables de lag para los atributos especificados utilizando SQL.
  
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list
        Lista de atributos para los cuales generar lags. Si es None, no se generan lags.
    cant_lag : int, default=1
        Cantidad de lags a generar para cada atributo
  
    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de lag agregadas
    """
    logger.info(f"Comienzo Feature de lag")


    # Armado de la consulta SQL


    if columnas[0]+"_lag_1" in df.columns:
        logger.info("Ya se hizo lags")
        return
    logger.info("Todavia no se hizo lags")
    sql = "CREATE or REPLACE table df as "
    sql +="(SELECT *"
    for attr in columnas:
        if attr in df.columns:
            for i in range(1,cant_lag+1):
                sql+= f",lag({attr},{i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_lag_{i}"
        else:
            print(f"No se encontro el atributo {attr} en df")
    sql+=" FROM df)"

    # Ejecucion de la consulta SQL
    conn = duckdb.connect(PATH_DATA_BASE_DB)
    conn.execute(sql)
    conn.close()
    logger.info(f"ejecucion lag finalizada")
    return

def feature_engineering_delta(df:pd.DataFrame , columnas:list[str],cant_lag:int=1 ) -> pd.DataFrame:
    """
    Genera variables de delta para los atributos especificados utilizando SQL.
  
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list
        Lista de atributos para los cuales generar lags. Si es None, no se generan lags.
    cant_lag : int, default=1
        Cantidad de delta a generar para cada atributo
  
    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de lag agregadas
    """
    if "delta_1_"+columnas[0] in df.columns:
        logger.info("Ya se hizo deltas")
        return
    logger.info(f"columnas con lags ? : {len(columnas)}")
    columnas_sin_lags = [c for c in  columnas if "lag_" not in c]
    logger.info(f"sacamos columnas sin lags ?: {len(columnas_sin_lags)}")
    logger.info("Todavia no se hizo deltas")
    logger.info(f"Comienzo feature de delta")
    sql = "CREATE or REPLACE table df as "
    sql+="(SELECT *"
    for attr in columnas_sin_lags:
        if attr in df.columns:
            for i in range(1,cant_lag+1):
                sql+= f", {attr}-{attr}_lag_{i} as delta_{i}_{attr}"
        else:
            print(f"No se encontro el atributo {attr} en df")
    sql+=" FROM df)"

    conn = duckdb.connect(PATH_DATA_BASE_DB)
    conn.execute(sql)
    conn.close()
    logger.info(f"ejecucion delta finalizada.  df shape: {df.shape}")
    return 

def feature_engineering_max_min(df:pd.DataFrame|np.ndarray , columnas:list[str]) -> pd.DataFrame|np.ndarray:
    """
    Genera variables de max y min para los atributos especificados por numero de cliente  utilizando SQL.
  
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list
        Lista de atributos para los cuales generar min y max. Si es None, no se generan lags.
    cant_lag : int, default=1
        Cantidad de delta a generar para cada atributo
  
    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de lag agregadas
    """
    logger.info(f"Comienzo feature max min. df shape: {df.shape}")
      
    sql="SELECT *"
    for attr in columnas:
        if attr in df.columns:
            sql+=f", MAX({attr}) OVER (PARTITION BY numero_de_cliente) as MAX_{attr}, MIN({attr}) OVER (PARTITION BY numero_de_cliente) as MIN_{attr}"
        else:
            print(f"El atributo {attr} no se encuentra en el df")
    
    sql+=" FROM df"

    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df=con.execute(sql).df()
    con.close()
    logger.info(f"ejecucion max min finalizada. df shape: {df.shape}")
    return df




def feature_engineering_ratio(df:pd.DataFrame|pd.Series, columnas:list[list[str]] )->pd.DataFrame:
    """
    Genera variables de ratio para los atributos especificados utilizando SQL.
  
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list[list]
        Lista de pares de columnas de monto y cantidad relacionados para generar ratios. Si es None, no se generan lags.
    cant_lag : int, default=1
        Cantidad de delta a generar para cada atributo
  
    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de ratios agregadas"""
    logger.info(f"Comienzo feature ratio. df shape: {df.shape}")
    sql="SELECT *"
    for par in columnas:
        if par[0] in df.columns and par[1] in df.columns:
            sql+=f", if({par[1]}=0 ,0,{par[0]}/{par[1]}) as ratio_{par[0]}_{par[1]}"
        else:
            print(f"no se encontro el par de atributos {par}")

    sql+=" FROM df"

    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df=con.execute(sql).df()
    con.close()
    logger.info(f"ejecucion ratio finalizada. df shape: {df.shape}")
    return df

def feature_engineering_linreg(df : pd.DataFrame|np.ndarray , columnas:list[str]) ->pd.DataFrame|np.ndarray:
    logger.info(f"Comienzo feature reg lineal. df shape: {df.shape}")
    sql="SELECT *"
    try:

        for attr in columnas:
            if attr in df.columns:
                sql+=f", regr_slope({attr} , cliente_antiguedad ) over ventana_3 as slope_{attr}"
            else :
                print(f"no se encontro el atributo {attr}")
        sql+=" FROM df window ventana_3 as (partition by numero_de_cliente order by foto_mes rows between 3 preceding and current row)"
    except Exception as e:
        logger.error(f"Error en la regresion lineal : {e}")
        raise
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df=con.execute(sql).df()
    con.close()
    logger.info(f"ejecucion reg lineal finalizada. df shape: {df.shape}")
    return df

def feature_engineering_max_min_2(df : pd.DataFrame|np.ndarray , columnas:list[str]) ->pd.DataFrame|np.ndarray:
    logger.info(f"Comienzo feature max min. df shape: {df.shape}")
    sql="SELECT *"
    try:

        for attr in columnas:
            if attr in df.columns:
                sql+=f", max({attr}  ) over ventana_3 as max_{attr} ,min({attr}) over ventana_3 as min_{attr}"
            else :
                print(f"no se encontro el atributo {attr}")
        sql+=" FROM df window ventana_3 as (partition by numero_de_cliente order by foto_mes rows between 3 preceding and current row)"
    except Exception as e:
        logger.error(f"Error en la max min : {e}")
        raise
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df=con.execute(sql).df()
    con.close()
    logger.info(f"ejecucion max min finalizada. df shape: {df.shape}")
    return df

def feature_engineering_normalizacion(df:pd.DataFrame , columnas:list[str]) -> pd.DataFrame:
    logger.info(f"Comienzo de la normalizacion de las cols seleccionadas , df shape {df.shape}")
    for attr in columnas:
        max_attr = df[attr].max()
        min_attr = df[attr].min()
        dif_max_min = max_attr - min_attr
        df[attr]= (df[attr] - min_attr) / dif_max_min
    logger.info(f"Finalizaion de la normalizacion df shape : {df.shape}")
    return df

def feature_engineering_rank(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """
    Genera variables de ranking para los atributos especificados utilizando SQL.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list[str]
        Lista de atributos para los cuales generar rankings.

    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de ranking agregadas
    """

    if not columnas:
        raise ValueError("La lista de columnas no puede estar vacía")

    columnas_validas = [col for col in columnas if col in df.columns]
    if not columnas_validas:
        raise ValueError("Ninguna de las columnas especificadas existe en el DataFrame")

    logger.info(f"Realizando feature engineering RANK para {len(columnas_validas)} columnas: {columnas_validas}")

    logger.info(f"Antes del ranking : la media de la columna {columnas_validas[0]} en 04 es de {df.loc[df['foto_mes']==202104,columnas_validas[0]].mean()}")

    rank_expressions = [
        f"PERCENT_RANK() OVER (PARTITION BY foto_mes ORDER BY {col}) AS {col}_rank"
        for col in columnas_validas
    ]

    sql = f"""
    SELECT *,
           {', '.join(rank_expressions)}
    FROM df
    """

    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()
    logger.info(f"Despues del ranking : la media de la columna {columnas_validas[0]} en 04 es de {df.loc[df['foto_mes']==202104,'mcuentas_saldo_rank'].mean()}")
    logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")
    return df
    

def feature_engineering_drop_cols(df:pd.DataFrame , columnas:list[str]) -> pd.DataFrame:
    logger.info(f"Comienzo dropeo de {len(columnas)} columnas. df shape --> {df.shape} ")
    try:
        df=df.drop(columns=columnas)
        logger.info(f"Fin del dropeo de {len(columnas)} columnas. df shape --> {df.shape}")
    except Exception as e:
        logger.error(f"Error al intentar borrar las colunas --> {e}")
        raise
    return df
