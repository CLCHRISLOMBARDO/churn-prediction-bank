import logging
from src.config import *


def creacion_directorios():
    ## Creacion de las carpetas
            #LOGS PATHS
    os.makedirs(PATH_LOGS,exist_ok=True)
            #OUTPUT PATHS
    os.makedirs(PATH_OUTPUT_BAYESIAN,exist_ok=True)
    os.makedirs(PATH_OUTPUT_FINALES,exist_ok=True)
    os.makedirs(PATH_OUTPUT_EXPERIMENTOS,exist_ok=True)
    os.makedirs(PATH_OUTPUT_EDA , exist_ok=True)
            #BAYESIANA
    os.makedirs(path_output_bayesian_db,exist_ok=True)
    os.makedirs(path_output_bayesian_bestparams,exist_ok=True)
    os.makedirs(path_output_bayesian_best_iter,exist_ok=True)
    os.makedirs(path_output_bayesian_graf,exist_ok=True)
            #FINALES
    os.makedirs(path_output_finales_model,exist_ok=True)
    os.makedirs(path_output_finales_feat_imp,exist_ok=True)
    os.makedirs(path_output_prediccion_final,exist_ok=True)
            #EXPERIMENTOS
    os.makedirs(path_output_exp_model,exist_ok=True)
    os.makedirs(path_output_exp_feat_imp,exist_ok=True)
    os.makedirs(path_output_exp_graf_gan_hist_grilla,exist_ok=True)
    os.makedirs(path_output_exp_graf_gan_hist_total,exist_ok=True)
    os.makedirs(path_output_exp_graf_gan_hist_semillas,exist_ok=True)
    os.makedirs(path_output_exp_graf_curva_ganancia,exist_ok=True)
    os.makedirs(path_output_exp_umbral,exist_ok=True)


