#feature_engineering.py
import pandas as pd
import numpy as np
import duckdb
import logging

#### FALTA AGREGAR LOS PUNTOS DE CONTROL PARA VISUALIZAR QUE ESTEN BIEN

logger = logging.getLogger(__name__)

def feature_engineering_lag(df:pd.DataFrame , columnas:list[str],cant_lag:int=1 ) -> pd.DataFrame:
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
    logger.info(f"Comienzo Feature de lag. df shape: {df.shape}")

    # Armado de la consulta SQL
    sql="SELECT *"
    for attr in columnas:
        if attr in df.columns:
            for i in range(1,cant_lag+1):
                sql+= f",lag({attr},{i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_lag_{i}"
        else:
            print(f"No se encontro el atributo {attr} en df")
    sql+=" FROM df"

    # Ejecucion de la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df=con.execute(sql).df()
    con.close()

    logger.info(f"ejecucion lag finalizada.  df shape: {df.shape}")
    return df

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
    # Armado de la consulta SQL
    logger.info(f"Comienzo feature de delta.  df shape: {df.shape}")
    sql="SELECT *"
    for attr in columnas:
        if attr in df.columns:
            for i in range(1,cant_lag+1):
                sql+= f", {attr}-{attr}_lag_{i} as delta_{i}_{attr}"
        else:
            print(f"No se encontro el atributo {attr} en df")
    sql+=" FROM df"

    # Ejecucion de la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df=con.execute(sql).df()
    con.close()
    logger.info(f"ejecucion delta finalizada.  df shape: {df.shape}")
    return df

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

    # Ejecucion de la consulta SQL
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