#config.py
## Configuracion General
SEMILLA= 14

## INPUT FILES
PATH_INPUT_DATA="data/competencia_01.csv"

## OUTPUTS FILES
PATH_OUTPUT_DATA="outputs/data_outputs/"
PATH_OUTPUT_RF="outputs/model_rf/"
PATH_OUTPUT_OPTIMIZACION="outputs/optimizacion_rf/"
PATH_OUTPUT_UMAP="outputs/umap/"
PATH_OUTPUT_CLUSTER="outputs/clusters/"
PATH_OUTPUT_SEGMENTACION = "outputs/segmentacion/"

## Submuestra - solo uso por el momento el de segmentacion
N_SUBSAMPLE = 2000
N_COMPLETO=10000
MES_TRAIN_SEGM =202104
MES_VALIDACION=202105
MES_TEST =202103

## OPTIMIZACION
GANANCIA=780000
ESTIMULO = 20000
N_TRIALS=25

## Random Forest
N_ESTIMATORS=1000





