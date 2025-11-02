import duckdb
import polars as pl
import pandas as pd
from src.config import *
import logging
logger=logging.getLogger(__name__)

def contador_targets(df:pd.DataFrame | pl.DataFrame):
    logger.info("Inicio control de la cantidad de los targets")
    sql="""
        select foto_mes , 
        COUNT(*) FILTER(where clase_ternaria = 'BAJA+1') as "BAJA+1",
        COUNT(*) FILTER(where clase_ternaria = 'BAJA+2') as "BAJA+2",
        COUNT(*) FILTER(where clase_ternaria = 'Continua') as "Continua"
        from df
        group by foto_mes"""
    # Ejecucion de la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    contador_de_targets=con.execute(sql).df()
    contador_de_targets=contador_de_targets.sort_values(by="foto_mes",ascending=True)
    con.close()
    logger.info(contador_de_targets)
    print(contador_de_targets)
    logger.info("Fin control de la cantidad de los targets")


def creacion_clase_ternaria(df:pd.DataFrame | pl.DataFrame ,competencia:str|int) ->pd.DataFrame | pl.DataFrame :
    logger.info("Inicio de la creacion del target")
    output_file = PATH_INPUT_DATA + f"competencia_0{competencia}.csv"
    sql= f"""
    with df2 as (
    SELECT foto_mes , numero_de_cliente,
    lead(foto_mes  , 1 ) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) as foto_mes_1,
    lead(foto_mes  , 2 ) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) as foto_mes_2
    FROM df)
    SELECT * EXCLUDE (foto_mes_1,foto_mes_2),
    if (foto_mes < 202106 , if(foto_mes <202105 ,
    if(df2.foto_mes_1 IS NULL,'BAJA+1', 
    if(df2.foto_mes_2 IS NULL,'BAJA+2','Continua')) ,
    if(df2.foto_mes_1 IS NULL,'BAJA+1',NULL)) ,NULL) as clase_ternaria
    from df
    LEFT JOIN df2 USING (numero_de_cliente,foto_mes)
    """
    # Ejecucion de la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df=con.execute(sql).df()
    con.close()
    df.to_parquet(output_file)
    contador_targets(df)

    logger.info(f"Fin de la creacion de target")
    return df


