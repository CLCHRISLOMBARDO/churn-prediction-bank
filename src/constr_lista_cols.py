#constr_lista_cols.py
import pandas as pd
import numpy as np
import logging

logger=logging.getLogger(__name__)

def contruccion_cols(df:pd.DataFrame|np.ndarray)->list[list]:
    logger.info("Comienzo de la extraccion de la seleccion de las columnas")
    # Columnas categoricas y numericas
    cat_cols =[]
    num_cols=[]
    col_drops=["numero_de_cliente","foto_mes","active_quarter","clase_ternaria","cliente_edad","cliente_antiguedad"
           ,"Visa_fultimo_cierre","Visa_fultimo_cierre","Master_fultimo_cierre","Visa_Fvencimiento",
           "Master_Fvencimiento"]
    for c in df.columns:
        if (df[c].nunique() <= 5):
            cat_cols.append(c)
        else:
            num_cols.append(c)
    lista_t=[c for c in list(map(lambda x : x if x[0]=='t' and x not in col_drops else np.nan ,df.columns )) if pd.notna(c)]
    lista_c=[c for c in list(map(lambda x : x if x[0]=='c' and x not in col_drops else np.nan ,df.columns )) if pd.notna(c)]
    lista_m=[c for c in list(map(lambda x : x if x[0]=='m' and x not in col_drops else np.nan ,df.columns )) if pd.notna(c)]
    lista_r=[c for c in df.columns if c not in (lista_t + lista_c + lista_m +col_drops )]


    # # Columnas lags y delta
    cols_lag_delta_max_min_regl=lista_m + lista_c+ lista_r

    # # Columnas para regresion lineal y max-min
    # lista_regl_max_min = lista_m + lista_c+ lista_r+lista_r

    # # Columnas para los ratios
    cols_ratios=[]
    for c in lista_c:
        i=0
        while i < len(lista_m) and c[1:] != lista_m[i][1:]:
            i+=1
        if i < len(lista_m):
            cols_ratios.append([lista_m[i],c ])




    logger.info(f"columnas para lags y deltas ")
    logger.info(f"columnas para ratios :")
    logger.info("Finalizacion de la construccion de las columnas")

    return [cols_lag_delta_max_min_regl ,cols_ratios ]


def contrs_cols_dropear_feat_imp(df:pd.DataFrame , file:str , threshold:float)->list[str]:
    logger.info(f"Comienzo de la seleccion de columnas a dropear")
    importance_df=pd.read_excel(file)
    f = importance_df["importance_%"]<=threshold
    cols_menos_importantes=list(importance_df.loc[f,'feature'].unique())
    cols_no_dropear=["foto_mes","numero_de_cliente"]
    cols_dropear=[c for c in  cols_menos_importantes if c not in cols_no_dropear]
    logger.info(f"Fin de la seleccion de columnas a dropear. Se eliminaran {len(cols_dropear)} columnas")

    return cols_dropear
    


