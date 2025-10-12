#lgbm_train_test.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
import xgboost as xgb

import logging
from time import time
import datetime

import pickle
import json
import os

from src.config import GANANCIA,ESTIMULO,prediccion_final_path

ganancia_acierto = GANANCIA
costo_estimulo=ESTIMULO

logger = logging.getLogger(__name__)

def entrenamiento_xgb(
    X_train: pd.DataFrame,
    y_train_binaria: pd.Series,
    w_train: pd.Series,
    best_iter: int,
    best_parameters: dict[str, object],
    name: str,
    output_path: str,
    semilla: int
) -> xgb.Booster:
    name = f"{name}_model_XGB"
    logger.info(f"Comienzo del entrenamiento del XGB : {name} en el mes train : {X_train['foto_mes'].unique()}")

    # 1) Definí las features reales (excluí columnas administrativas si las tenés)
    cols_drop = [c for c in ["foto_mes"] if c in X_train.columns]
    feature_names = [c for c in X_train.columns if c not in cols_drop]
    X_train_feat = X_train[feature_names]

    logger.info(f"Mejor cantidad de árboles para el mejor model {best_iter}")
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "seed": int(semilla),
        "verbosity": 0,
        **best_parameters
    }

    # 2) Pasá feature_names al DMatrix
    dtrain = xgb.DMatrix(
        data=X_train_feat,
        label=y_train_binaria,
        weight=w_train,
        feature_names=feature_names
    )

    model_xgb = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(best_iter)
    )

    # 3) Guardá metadata útil dentro del Booster (como strings)
    model_xgb.set_attr(feature_names_json=json.dumps(feature_names))

    logger.info(f"comienzo del guardado en {output_path}")
    try:
        filename = output_path + f"{name}.txt"
        model_xgb.save_model(filename)
        logger.info(f"Modelo {name} guardado en {filename}")
        logger.info(f"Fin del entrenamiento del XGB en el mes train : {X_train['foto_mes'].unique()}")
    except Exception as e:
        logger.error(f"Error al intentar guardar el modelo {name}, por el error {e}")
        return

    return model_xgb
def prediccion_test_xgb(X: pd.DataFrame, model_xgb: xgb.Booster) -> pd.Series:
    mes = X["foto_mes"].unique() if "foto_mes" in X.columns else "N/A"
    logger.info(f"comienzo prediccion del modelo en el mes {mes}")

    # 1) Recuperar feature_names del modelo (si están)
    feat_json = model_xgb.attr("feature_names_json")
    if feat_json:
        feature_names = json.loads(feat_json)
    else:
        # fallback: si el modelo las tiene adentro (depende de versión)
        feature_names = getattr(model_xgb, "feature_names", None)
        if feature_names is None:
            # último recurso: usamos todas menos 'foto_mes'
            feature_names = [c for c in X.columns if c != "foto_mes"]

    # 2) Chequeo de columnas y orden
    faltantes = set(feature_names) - set(X.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas para predecir con XGB: {sorted(faltantes)[:10]} ...")
    X_aligned = X.loc[:, feature_names]

    # 3) DMatrix + predict
    dtest = xgb.DMatrix(X_aligned, feature_names=feature_names)
    y_pred = model_xgb.predict(dtest)  # para 'binary:logistic' ya son probabilidades

    logger.info("Fin de la prediccion del modelo")
    return  y_pred


# def entrenamiento_xgb(X_train:pd.DataFrame , y_train_binaria:pd.Series , w_train:pd.Series , best_iter:int , best_parameters:dict[str,object] , name:str,output_path:str ,semilla:int)->xgb.Booster:
#     name=f"{name}_model_XGB"
#     logger.info(f"Comienzo del entrenamiento del XGB : {name} en el mes train : {X_train['foto_mes'].unique()}")
#     best_iter = best_iter
#     logger.info(f"Mejor cantidad de árboles para el mejor model {best_iter}")
#     params = {
#         'objective': 'binary:logistic',
#         'eval_metric': 'logloss',       
#         'tree_method': 'hist',
#         'seed': semilla,
#         'verbosity': 0,
#         **best_parameters
#     }
#     train_data = xgb.DMatrix(data=X_train,
#                             label=y_train_binaria,
#                             weight=w_train)
#     model_xgb = xgb.train(params,
#                     train_data,
#                     num_boost_round=best_iter)
#     logger.info(f"comienzo del guardado en {output_path}")
#     try:
#         filename=output_path+f'{name}.txt'
#         model_xgb.save_model(filename )                         
#         logger.info(f"Modelo {name} guardado en {filename}")
#         logger.info(f"Fin del entrenamiento del XGB en el mes train : {X_train['foto_mes'].unique()}")
#     except Exception as e:
#         logger.error(f"Error al intentar guardar el modelo {name}, por el error {e}")
#         return
#     return model_xgb






def grafico_feature_importance_xgb(model_xgb, X_train: pd.DataFrame,name: str,output_path: str,importance_type: str = "gain",top_n: int | None = None):
    logger.info("Comienzo del grafico de feature importance del XGB")
    os.makedirs(output_path, exist_ok=True)
    base_name = f"{name}_feature_importance"

    # ---- Gráfico ----
    try:
        fig = plt.figure(figsize=(10, 20))
        ax = fig.add_subplot(111)
        # Para Booster y para scikit-wrappers, plot_importance funciona igual
        xgb.plot_importance(
            model_xgb,
            ax=ax,
            importance_type=importance_type,
            max_num_features=top_n,
            show_values=False,
            grid=True
        )
        fig.tight_layout()
        fig.savefig(os.path.join(output_path, f"{base_name}_grafico.png"), bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error al intentar graficar los feat importances: {e} del XGB")
    logger.info("Fin del grafico de feature importance")

    # ---- Datos (DataFrame) ----
    try:
        # Caso 1: Booster nativo
        if isinstance(model_xgb, xgb.Booster):
            # dict: { 'f0': score, 'f1': score, ... } o con nombres reales si se entrenó con feature_names
            scores = model_xgb.get_score(importance_type=importance_type)
            if not scores:
                logger.warning("get_score devolvió vacío. ¿Entrenaste con DMatrix sin features o modelo vacío?")
            # Mapear claves a nombres de columnas si vienen como f0, f1, ...
            feature_names = []
            importances = []
            for k, v in scores.items():
                if k.startswith("f") and k[1:].isdigit():
                    idx = int(k[1:])
                    if 0 <= idx < X_train.shape[1]:
                        feature_names.append(X_train.columns[idx])
                    else:
                        feature_names.append(k)  # fallback
                else:
                    feature_names.append(k)    # ya sería el nombre real
                importances.append(v)

            imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})

        # Caso 2: Wrapper scikit-learn (XGBClassifier / XGBRegressor)
        elif hasattr(model_xgb, "feature_importances_"):
            imp = model_xgb.feature_importances_
            imp_df = pd.DataFrame({"feature": X_train.columns, "importance": imp})

        else:
            raise TypeError(
                "Tipo de modelo no soportado. Se espera xgboost.Booster o un wrapper con .feature_importances_."
            )

        # Normalizar/ordenar y (opcional) recortar top_n
        imp_df = imp_df.groupby("feature", as_index=False)["importance"].sum()
        imp_df = imp_df.sort_values("importance", ascending=False)
        imp_df["importance_%"] = (imp_df["importance"] / imp_df["importance"].sum() * 100.0).fillna(0.0)

        if top_n is not None:
            imp_df = imp_df.head(top_n)

        # Guardar Excel
        out_xlsx = os.path.join(output_path, f"{base_name}_data_frame.xlsx")
        imp_df.to_excel(out_xlsx, index=False)
        logger.info(f"Guardado feat imp en excel con EXITO: {out_xlsx}")

    except Exception as e:
        logger.error(f"Error al intentar guardar los feat imp en excel por {e}")
        
# def grafico_feature_importance_xgb(model_xgb:xgb.Booster,X_train:pd.DataFrame,name:str,output_path:str):
    # logger.info("Comienzo del grafico de feature importance del XGB")
    # name=f"{name}_feature_importance"
    # try:
    #     xgb.plot_importance(model_xgb, figsize=(10, 20))
    #     plt.savefig(output_path+f"{name}_grafico.png", bbox_inches='tight')
    # except Exception as e:
    #     logger.error(f"Error al intentar graficar los feat importances: {e} del XGB")
    # logger.info("Fin del grafico de feature importance")

    # importances = model_xgb.feature_importance()
    # feature_names = X_train.columns.tolist()
    # importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    # importance_df = importance_df.sort_values('importance', ascending=False)
    # importance_df["importance_%"] = (importance_df["importance"] /importance_df["importance"].sum())*100
    # # importance_df[importance_df['importance'] > 0]
    # logger.info("Guardado de feat import en excel")
    # try :
    #     importance_df.to_excel(output_path+f"{name}_data_frame.xlsx" ,index=False)
    #     logger.info("Guardado feat imp en excel con EXITO")
    # except Exception as e:
    #     logger.error(f"Error al intentar guardar los feat imp en excel por {e}")

# def prediccion_test_xgb(X:pd.DataFrame ,  model_xgb:xgb.Booster)-> pd.Series:
#     mes=X["foto_mes"].unique()
#     logger.info(f"comienzo prediccion del modelo en el mes {mes}")
#     y_pred_xgb = model_xgb.predict(X)
#     logger.info("Fin de la prediccion del modelo")
#     return y_pred_xgb
