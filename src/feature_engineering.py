#feature_engineering.py
import pandas as pd
import numpy as np
import duckdb
import logging

#### FALTA AGREGAR LOS PUNTOS DE CONTROL PARA VISUALIZAR QUE ESTEN BIEN
from src.config import FILE_INPUT_DATA , PATH_DATA_BASE_DB
logger = logging.getLogger(__name__)


def feature_engineering_drop_cols(df:pd.DataFrame , columnas:list[str]) :
    if columnas is None:
        logger.info(f"No se realiza el dropeo de columnas. Solo la creacion de la tabla df")
    else:
        logger.info(f"Comienzo dropeo de {len(columnas)} columnas.")
    
   
    sql = "create or replace table df as "
    if columnas is None:
        sql+="""SELECT *
                from df_completo"""
    else:
        sql+= "SELECT * EXCLUDE("
        for i,c in enumerate(columnas):
            if i==0:
                sql+=f" {c}"
            else:
                sql+=f",{c}"
        sql+= ") from df_completo"
    
    # columnas_faltantes = [c for c in columnas if c not in df.columns]
    # if len(columnas_faltantes)>0:
    #     logger.error(f"{columnas_faltantes} no esta en el df columns")
    #     raise
 
    try:
        conn=duckdb.connect(PATH_DATA_BASE_DB)
        conn.execute(sql)
        conn.close()
        logger.info(f"Fin del dropeo de columnas.")
    except Exception as e:
        logger.error(f"Error al intentar crear en la base de datos --> {e}")
        raise
    return

def feature_engineering_lag(df:pd.DataFrame ,columnas:list[str],orden_lag:int=1 ):
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
    orden_lag_ya_realizado=1
    marca_nueva_real=0
    while marca_nueva_real ==0 and orden_lag_ya_realizado <= orden_lag:
        if any(c.endswith(f"_lag_{orden_lag_ya_realizado}") for c in df.columns):
            logger.info(f"Ya se hizo lag_{orden_lag_ya_realizado}")
            orden_lag_ya_realizado+=1
        else:
            marca_nueva_real=1
    if orden_lag_ya_realizado > orden_lag:
        logger.info(f"Ya se hicieron todos los lags pedidos hasta orden {orden_lag_ya_realizado-1}")
        return
    
    logger.info(f"Ya se hicieron los lags hasta orden {orden_lag_ya_realizado-1}. Falta hasta orden {orden_lag}")
    sql = "CREATE or REPLACE table df_completo as "
    sql +="(SELECT *"
    for attr in columnas:
        if attr in df.columns:
            for i in range(orden_lag_ya_realizado,orden_lag+1):
                sql+= f",lag({attr},{i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_lag_{i}"
        else:
            logger.warning(f"No se encontro el atributo {attr} en df")
    sql+=" FROM df_completo)"

    # Ejecucion de la consulta SQL
    conn = duckdb.connect(PATH_DATA_BASE_DB)
    conn.execute(sql)
    conn.close()
    logger.info(f"ejecucion lag finalizada")
    return

def feature_engineering_delta(df:pd.DataFrame , columnas:list[str],orden_delta:int=1 ) :
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
    logger.info(f"Comienzo feature de delta")
    orden_delta_ya_realizado=1
    marca_nueva_real=0
    while marca_nueva_real ==0 and orden_delta_ya_realizado <= orden_delta:
        if any(c.endswith(f"_delta_{orden_delta_ya_realizado}")  for c in df.columns):
            logger.info(f"Ya se hizo delta_{orden_delta_ya_realizado}_")
            orden_delta_ya_realizado+=1
        else:
            marca_nueva_real=1
    if orden_delta_ya_realizado > orden_delta:
        logger.info(f"Ya se hicieron todos los deltas pedidos hasta orden {orden_delta_ya_realizado-1}")
        return
    logger.info(f"Ya se hicieron los deltas hasta orden {orden_delta_ya_realizado-1}. Falta hasta orden {orden_delta}")

    
    sql = "CREATE or REPLACE table df_completo as "
    sql+="(SELECT *"
    for attr in columnas:
        if attr in df.columns:
            for i in range(orden_delta_ya_realizado,orden_delta+1):
                sql += (
                f", TRY_CAST({attr} AS DOUBLE) "
                f"- TRY_CAST({attr}_lag_{i} AS DOUBLE) AS {attr}_delta_{i}")
                # sql+= f", {attr}-{attr}_lag_{i} as delta_{i}_{attr}"
        else:
            logger.warning(f"No se encontro el atributo {attr} en df")
    sql+=" FROM df_completo)"
    conn = duckdb.connect(PATH_DATA_BASE_DB)
    conn.execute(sql)
    conn.close()
    logger.info(f"ejecucion delta finalizada")
    return 

def feature_engineering_ratio(df:pd.DataFrame|pd.Series, columnas:list[list[str]] ):
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
    logger.info(f"Comienzo feature ratio")

    if any(c.endswith(f"_ratio") for c in df.columns):
        logger.info("Ya se hizo ratios")
        return
    logger.info("Todavia no se hizo ratios")
    sql="CREATE or REPLACE table df_completo as "
    sql+="(SELECT *"
    for par in columnas:
        if par[0] in df.columns and par[1] in df.columns:
            sql+=f", if({par[1]}=0 ,0,{par[0]}/{par[1]}) as {par[0]}_{par[1]}_ratio"
        else:
            print(f"no se encontro el par de atributos {par}")

    sql+=" FROM df_completo)"

    conn = duckdb.connect(PATH_DATA_BASE_DB)
    conn.execute(sql)
    conn.close()

    logger.info(f"ejecucion ratio finalizada.")
    return 

def feature_engineering_linreg(df : pd.DataFrame|np.ndarray , columnas:list[str],ventana:int=3) :
    logger.info(f"Comienzo feature reg lineal")

    if any(c.endswith("_slope") for c in df.columns):
        logger.info("Ya se hizo slope")
        return
    logger.info("Todavia no se hizo slope")
    sql = "Create or replace table df_completo as "
    sql+="(SELECT *"
    try:
        for attr in columnas:
            if attr in df.columns:
                sql+=f", regr_slope({attr} , cliente_antiguedad ) over ventana_{ventana} as {attr}_slope"
            else :
                print(f"no se encontro el atributo {attr}")
        sql+=f" FROM df_completo window ventana_{ventana} as (partition by numero_de_cliente order by foto_mes rows between {ventana} preceding and current row))"
    except Exception as e:
        logger.error(f"Error en la regresion lineal : {e}")
        raise
    conn = duckdb.connect(PATH_DATA_BASE_DB)
    conn.execute(sql)
    conn.close()
    logger.info(f"ejecucion reg lineal finalizada")
    return 

def feature_engineering_max_min(df : pd.DataFrame|np.ndarray , columnas:list[str],ventana:int=3) :
    logger.info(f"Comienzo feature max min. df shape: {df.shape}")
    palabras_max_min=["_max","_min"]
    if any(any(c.endswith(p) for p in palabras_max_min) for c in df.columns):
        logger.info("Ya se hizo max min")
        return
    
    sql="CREATE or REPLACE table df_completo as "
    sql+="(SELECT *"
    try:

        for attr in columnas:
            if attr in df.columns:
                sql+=f", max({attr}  ) over ventana_{ventana} as {attr}_max ,min({attr}) over ventana_{ventana} as {attr}_min"
            else :
                print(f"no se encontro el atributo {attr}")
        sql+=f" FROM df_completo window ventana_{ventana} as (partition by numero_de_cliente order by foto_mes rows between {ventana} preceding and current row))"
    except Exception as e:
        logger.error(f"Error en la max min : {e}")
        raise
    conn=duckdb.connect(PATH_DATA_BASE_DB)
    conn.execute(sql)
    conn.close()
    logger.info(f"ejecucion max min finalizada. ")
    return 


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
    FROM df_completo
    """

    con = duckdb.connect(PATH_DATA_BASE_DB)
    con.execute(sql)
    con.close()
    logger.info(f"Despues del ranking : la media de la columna {columnas_validas[0]} ")
    logger.info(f"Feature engineering completado")
    return 
    


def feature_engineering_max_min_2(df:pd.DataFrame|np.ndarray , columnas:list[str]) -> pd.DataFrame|np.ndarray:
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

