from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import polars as pl
import duckdb 
import logging

from src.config import PATH_OUTPUT_EDA

logger = logging.getLogger(__name__)
def mean_por_mes(df:pd.DataFrame|pl.DataFrame ) ->pl.DataFrame|pd.DataFrame:
    logger.info("Comienzo del eda de media por foto_mes")
    if isinstance(df , pl.DataFrame):
        num_cols=df.select(pl.selectors.numeric()).columns
    elif isinstance(df , pd.DataFrame):
        num_cols = df.select_dtypes(include="number").columns
    
    sql='select foto_mes'

    for c in num_cols:
        sql+=f', AVG({c}) as {c}_mean'
    sql+=' from df group by foto_mes'

    con = duckdb.connect(database=":memory:")
    con.register("df",df)
    medias_por_mes = con.execute(sql).df()
    con.close()
    logger.info("Fin del eda de media por foto_mes")
    return medias_por_mes

def crear_reporte_pdf(df, xcol, columnas_y, name_pdf, titulo="Reporte de gráficos"):
    """
    Genera un PDF con una página por gráfico 
    """

    logger.info("Comienzo de la creacion del reporte")

    salida_pdf = PATH_OUTPUT_EDA+name_pdf
    with PdfPages(salida_pdf) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))  
        fig.text(0.5, 0.6, titulo, ha="center", va="center", fontsize=20)
        fig.text(0.5, 0.5, f"Variables: {len(columnas_y)}", ha="center", va="center")
        fig.text(0.5, 0.4, f"Eje X: {xcol}", ha="center", va="center")
        fig.text(0.5, 0.2, "Generado con matplotlib", ha="center", va="center", fontsize=9)
        pdf.savefig(fig); plt.close(fig)

        for col in columnas_y:
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            sns.lineplot(data=df , x = xcol , y =col,ax=ax)
            sns.scatterplot(data=df , x = xcol , y =col,ax=ax)
            ax.set_title(f"{col} vs {xcol}")
            ax.set_xlabel(xcol)
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        d = pdf.infodict()
        d['Title'] = titulo
        d['Author'] = "Tu nombre"
        d['Subject'] = "Reporte automático de gráficos"
        d['Keywords'] = "matplotlib, reporte, gráficos"
        d['Creator'] = "Python + Matplotlib"
        logger.info("Fin de la creacion del reporte")


