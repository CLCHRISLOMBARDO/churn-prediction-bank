#config.py
## Configuracion General
SEMILLA= 773767
SEMILLAS=[259621, 282917, 413417, 773767, 290827]
N_SEMILLAS= 49
## INPUT PATHS
# PATH_INPUT_DATA="data/competencia_01.csv"
PATH_INPUT_DATA="~/buckets/b1/datasets/competencia_01.csv"

## LOG PATH
PATH_LOGS= "logs/"

## OUTPUTS PATHS
PATH_OUTPUT_DATA="outputs/data_outputs/"
PATH_OUTPUT_BAYESIAN="outputs/bayesian_outputs/"
PATH_OUTPUT_FINALES = "outputs/finales_outputs/"
PATH_OUTPUT_EXPERIMENTOS = "outputs/experimentos_outputs/"


## PATH_OUTPUT_BAYESIAN
path_output_bayesian_db = PATH_OUTPUT_BAYESIAN + 'db/'
path_output_bayesian_bestparams= PATH_OUTPUT_BAYESIAN+'best_params/'
path_output_bayesian_best_iter = PATH_OUTPUT_BAYESIAN+'best_iters/'
path_output_bayesian_graf = PATH_OUTPUT_BAYESIAN+'grafico_bayesiana/'

## PATH_OUTPUT_FINALES
path_output_finales_model=PATH_OUTPUT_FINALES + 'model/'
path_output_finales_feat_imp=PATH_OUTPUT_FINALES +'feature_importances/'
path_output_finales_graf_gan_hist_grilla=PATH_OUTPUT_FINALES +'graf_ganancias_hist_grilla/'
path_output_finales_graf_gan_hist_total=PATH_OUTPUT_FINALES +'graf_ganancias_hist_total/'
path_output_finales_graf_gan_hist_semillas=PATH_OUTPUT_FINALES +'graf_ganancias_hist_semillas/'
path_output_finales_graf_curva_ganancia=PATH_OUTPUT_FINALES +'graf_curva_ganancia/'
path_output_finales_umbral=PATH_OUTPUT_FINALES +'umbrales/'
path_output_prediccion_final = PATH_OUTPUT_FINALES + 'final_prediction/'

# PATH_OUTPUT_EXPERIMENTOS
path_output_exp_model=PATH_OUTPUT_EXPERIMENTOS + "exp_model/"
path_output_exp_feat_imp=PATH_OUTPUT_EXPERIMENTOS + "exp_feat_importances/"
path_output_exp_graf_gan_hist_grilla=PATH_OUTPUT_EXPERIMENTOS + "exp_graf_ganancias_hist_grilla/"
path_output_exp_graf_gan_hist_total=PATH_OUTPUT_EXPERIMENTOS + "exp_graf_ganancias_hist_total/"
path_output_exp_graf_gan_hist_semillas=PATH_OUTPUT_EXPERIMENTOS + "exp_graf_ganancias_hist_semillas/"
path_output_exp_graf_curva_ganancia=PATH_OUTPUT_EXPERIMENTOS + "exp_graf_curva_ganancia/"
path_output_exp_umbral=PATH_OUTPUT_EXPERIMENTOS + "exp_umbrales/"


## Submuestra - solo uso por el momento el de segmentacion
MES_TRAIN =[202101,202102,202103]
MES_TEST =[202104]
MES_A_PREDECIR=202106
MES_01=202101
MES_02=202102
MES_03=202103
MES_04=202104
MES_05=202105

## OPTIMIZACION LGBM
UMBRAL=0.025
GANANCIA=780000
ESTIMULO = 20000
N_TRIALS= 30
N_BOOSTS=1000
N_FOLDS=5





