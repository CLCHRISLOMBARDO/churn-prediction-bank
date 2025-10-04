#optimizacion.py
import pandas as pd
import numpy as np
import lightgbm as lgb

from joblib import Parallel, delayed
import optuna
from optuna.study import Study
from time import time
import os

import pickle
import json
import logging
from optuna.samplers import TPESampler # Para eliminar el componente estocastico de optuna
from optuna.visualization import plot_param_importances, plot_contour,  plot_slice, plot_optimization_history

from src.config import PATH_OUTPUT_OPTIMIZACION, GANANCIA,ESTIMULO,SEMILLA ,N_BOOSTS ,N_FOLDS

output_path = PATH_OUTPUT_OPTIMIZACION
db_path = output_path + 'db/'
bestparams_path = output_path+'best_params/'
best_iter_path = output_path+'best_iters/'
graf_bayesiana_path = output_path+'grafico_bayesiana/'


ganancia_acierto = GANANCIA
costo_estimulo = ESTIMULO

logger = logging.getLogger(__name__)


def lgb_gan_eval(y_pred, data):
    weight = data.get_weight()
    ganancia = np.where(weight == 1.00002, ganancia_acierto, 0) - np.where(weight < 1.00002, costo_estimulo, 0)
    ganancia = ganancia[np.argsort(y_pred)[::-1]]
    ganancia = np.cumsum(ganancia)

    return 'gan_eval', np.max(ganancia) , True

def optim_hiperp_binaria(X_train:pd.DataFrame ,y_train_binaria:pd.Series,w_train:pd.Series, n_trials:int, name:str)-> Study:
    logger.info("Comienzo optimizacion hiperp binario")
    name ="binaria"+name


    def objective(trial):

        num_leaves = trial.suggest_int('num_leaves', 8, 100)
        learning_rate = trial.suggest_float('learning_rate', 0.005, 0.3) # mas bajo, más iteraciones necesita
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 1, 1000)
        feature_fraction = trial.suggest_float('feature_fraction', 0.1, 1.0)
        bagging_fraction = trial.suggest_float('bagging_fraction', 0.1, 1.0)

        params = {
            'objective': 'binary',
            'metric': 'custom',
            'boosting_type': 'gbdt',
            'first_metric_only': True,
            'boost_from_average': True,
            'feature_pre_filter': False,
            'max_bin': 31,
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'min_data_in_leaf': min_data_in_leaf,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'seed': SEMILLA,
            'verbose': -1
        }
        train_data = lgb.Dataset(X_train,
                                label=y_train_binaria,
                                weight=w_train)
        cv_results = lgb.cv(
            params,
            train_data,
            num_boost_round=N_BOOSTS, # modificar, subit y subir... y descomentar la línea inferior
            #early_stopping_rounds= int(50 + 5 / learning_rate),
            feval=lgb_gan_eval,
            stratified=True,
            nfold=N_FOLDS,
            seed=SEMILLA,
            callbacks=[
                lgb.early_stopping(stopping_rounds=int(50 + 5/learning_rate), verbose=False),
                lgb.log_evaluation(period=200),
                ]
        )
        max_gan = max(cv_results['valid gan_eval-mean'])
        best_iter = cv_results['valid gan_eval-mean'].index(max_gan) + 1

        # Guardamos cual es la mejor iteración del modelo
        trial.set_user_attr("best_iter", best_iter)

        return max_gan * N_FOLDS
    
    storage_name = "sqlite:///" + db_path + "optimization_lgbm.db"
    study_name = f"study_lgbm_{name}"    


    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True
        #sampler=TPESampler(seed=SEMILLA)
    )

    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    best_iter=study.best_trial.user_attrs["best_iter"]
    
    # Guardo best iter
    try:
        with open(best_iter_path + f"best_iter_{name}.json","w") as f:
            json.dump(best_iter , f ,indent=4)
        logger.info(f"best_iter_{name}.json guardado en {best_iter_path} ")
    except Exception as e:
        logger.error(f"Error al tratar de guardar el json de best iter por el error :{e}")

    # Guardo best params
    try:
        with open(bestparams_path+f"best_params{name}.json", "w") as f:
            json.dump(best_params, f, indent=4) 
        logger.info(f"best_params{name}.json guardado en {bestparams_path}")
        logger.info(f"Finalizacion de optimizacion hiperp binario con study name {study_name}.")
    except Exception as e:
        logger.error(f"Error al tratar de guardar el json de los best parameters por el error :{e}")
    return study

def graficos_bayesiana(study:Study, name: str):
    logger.info("Comienzo de la creacion de graficos")
    try:
        fig1 = plot_optimization_history(study)
        fig1.write_image(graf_bayesiana_path+f"{name}_graficos_opt_history.png")

        fig2 = plot_param_importances(study)
        fig2.write_image(graf_bayesiana_path+f"{name}_graficos_param_importances.png")

        fig3 = plot_slice(study)
        fig3.write_image(graf_bayesiana_path+f"{name}_graficos_slice.png")

        fig4 = plot_contour(study)
        fig4.write_image(graf_bayesiana_path+f"{name}_graficos_contour_all.png")

        fig5 = plot_contour(study, params=["num_leaves", "learning_rate"])
        fig5.write_image(graf_bayesiana_path+f"{name}_graficos_contour_specific.png")

        logger.info(f" Gráficos guardados en {output_path}")
    except Exception as e:
        logger.error(f"Error al generar las gráficas: {e}")




def _ganancia_prob(y_hat:pd.Series|np.ndarray , y:pd.Series|np.ndarray ,prop=1,class_index:int =1,threshold:int=0.025)->float:
    logger.info("comienzo funcion ganancia con threhold = 0.025")
    @np.vectorize
    def _ganancia_row(predicted , actual , threshold=0.025):
        return (predicted>=threshold) * (ganancia_acierto if actual=="BAJA+2" else -costo_estimulo)
    logger.info("Finalizacion funcion ganancia con threhold = 0.025")
    return _ganancia_row(y_hat[:,class_index] ,y).sum() /prop


# def optim_hiperp_ternaria(X:pd.DataFrame|np.ndarray ,y:pd.Series|np.ndarray , n_trials:int , name:str)-> Study:
    
#     logger.info("Inicio de optimizacion hiperp ternario")
#     name ="ternaria"+name
#     def objective(trial):
#         max_depth = trial.suggest_int('max_depth', 2, 32)
#         min_samples_split = trial.suggest_int('min_samples_split', 2, 2000)
#         min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 200)
#         max_features = trial.suggest_float('max_features', 0.05, 0.7)

#         model = RandomForestClassifier(
#             n_estimators=N_ESTIMATORS,
#             max_depth=max_depth,
#             min_samples_split=min_samples_split,
#             min_samples_leaf=min_samples_leaf,
#             max_features=max_features,
#             max_samples=0.7,
#             random_state=SEMILLA,
#             n_jobs=12,
#             oob_score=True
#         )

#         model.fit(X, y)

#         return _ganancia_prob(model.oob_decision_function_, y)

#     storage_name = "sqlite:///" + db_path + "optimization_tree.db"
#     study_name = f"rf_ganancia_{name}"  

#     study = optuna.create_study(
#         direction="maximize",
#         study_name=study_name,
#         storage=storage_name,
#         load_if_exists=True,
#         sampler=TPESampler(seed=SEMILLA)
#     )

#     study.optimize(objective, n_trials=n_trials)

#     best_params = study.best_trial.params
    
#     try:
#         with open(bestparms_path+f"best_params_ganancia_{name}.json", "w") as f:
#             json.dump(best_params, f, indent=4) 
#             logger.info(f"best_params_ganancia_{name}.json guardado en outputs/optimizacion_rf/best_params/")
#         logger.info("Finalizacion de optimizacion hiperp binario.")
#     except Exception as e:
#         logger.error(f"Error al tratar de guardar el json de los best parameters por el error :{e}")
#     return study


