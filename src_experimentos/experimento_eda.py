import duckdb 
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import *
from src.eda import mean_por_mes , crear_reporte_pdf



def lanzar_eda():
    df= pl.read_csv(PATH_INPUT_DATA, infer_schema_length=10000)
    media_por_mes = mean_por_mes(df)

    crear_reporte_pdf(media_por_mes, xcol='foto_mes', columnas_y=media_por_mes.columns,
                  name_pdf="reporte_medias_por_mes.pdf",
                  titulo="Medias por mes — Scatter por variable")
