import duckdb
import polars as pl
import pandas as pd
from src.config import *
import logging

logger =logging.getLogger(__name__)

def create_data_base():
    logger.info(f"Creacion de la base de datos en : {FILE_INPUT_DATA_CRUDO}")
    sql = f"""
    create or replace table df as 
    select *
    from read_csv_auto('{FILE_INPUT_DATA_CRUDO}')"""

    conn=duckdb.connect(PATH_DATA_BASE_DB)
    conn.execute(sql)
    conn.close()

def contador_targets():
    logger.info("Inicio control de la cantidad de los targets")
    sql="""
        select foto_mes , 
        COUNT(*) FILTER(where clase_ternaria = 'BAJA+1') as "BAJA+1",
        COUNT(*) FILTER(where clase_ternaria = 'BAJA+2') as "BAJA+2",
        COUNT(*) FILTER(where clase_ternaria = 'Continua') as "Continua"
        from df
        group by foto_mes"""
    # Ejecucion de la consulta SQL
    conn = duckdb.connect(PATH_DATA_BASE_DB)
    logger.info(conn.execute(sql))
    conn.close()


    # contador_de_targets=con.execute(sql).df()
    # contador_de_targets=contador_de_targets.sort_values(by="foto_mes",ascending=True)
    # con.close()
    # logger.info(contador_de_targets)
    # print(contador_de_targets)
    logger.info("Fin control de la cantidad de los targets")


def creacion_clase_ternaria() :
    logger.info("Inicio de la creacion del target")

    sql= f"""CREATE or REPLACE table df as
    (with df2 as (
    SELECT foto_mes , numero_de_cliente,
    lead(foto_mes  , 1 ) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) as foto_mes_1,
    lead(foto_mes  , 2 ) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) as foto_mes_2
    FROM df)
    SELECT * EXCLUDE (foto_mes_1,foto_mes_2),
    if (foto_mes < 202108 , if(foto_mes <202107 ,
    if(df2.foto_mes_1 IS NULL,'BAJA+1', 
    if(df2.foto_mes_2 IS NULL,'BAJA+2','Continua')) ,
    if(df2.foto_mes_1 IS NULL,'BAJA+1',NULL)) ,NULL) as clase_ternaria
    from df
    LEFT JOIN df2 USING (numero_de_cliente,foto_mes))
    """
    # Ejecucion de la consulta SQL
    conn = duckdb.connect(PATH_DATA_BASE_DB)
    conn.execute(sql)
    conn.close()
    # con.register("df", df)
    # df=con.execute(sql).df()
    # con.close()
    # df.to_csv(output_file,index=False)
    logger.info(f"Fin de la creacion de target")
    return 


def lanzar_creacion_clase_ternaria():
    logger.info("Lanzamiento de la creacion de la clase ternaria target")
    create_data_base()
    
    # try:
    #     df=pd.read_csv(FILE_INPUT_DATA_CRUDO)
    #     logger.info("Cargado del dataset crudo con exito")
        
    # except Exception as e:
    #     logger.error(f"Error al cargar el data set por : {e}")
    
    creacion_clase_ternaria()
    contador_targets()

    return 


