import pandas as pd
import numpy as np
from datetime import datetime

from src.class_competition import *
from src.class_preprocessing import *
from src.parameters import Parameters
from sklearn.model_selection import train_test_split

import warnings;
warnings.simplefilter('ignore')

def train_classification(X_vect, y_encoded):
        # División de los datos en conjunto de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X_vect, y_encoded, test_size=0.2, random_state=42)
        
        dict_models, dict_pred, dict_metrics = evaluate_models_classification(X_train, X_test, y_train, y_test)
        _winner_metrics, model = class_competition(dict_models, dict_pred, dict_metrics)
        bm = _winner_metrics.pop('model_name')
        
def fn_classification(df_class, X_vect):
    
    winner_metrics_temp = pd.read_excel(os.path.join(Parameters.results_path, 'winner_metrics_class.xlsx'))
    df_col_dummies = pd.read_excel(os.path.join(Parameters.results_path, 'df_col_dummies.xlsx'))
    X_vect = pd.concat([df_col_dummies, X_vect], axis = 0).fillna(False)
    bm = winner_metrics_temp.iloc[-1]['model_name']
    
    models_path = os.path.join(os.path.dirname(Parameters.results_path),'models')
    model_file = max([file for file in os.listdir(models_path) if file.startswith(bm)])
    
    # Cargar el modelo
    model_file_path = os.path.join(models_path, model_file)
    with open(model_file_path, 'rb') as f:
        modelo_cargado = pickle.load(f)

    classifications = modelo_cargado.predict(X_vect)
    df_class['Class'] = classifications
    df_class['date_run'] = pd.to_datetime(datetime.now()).strftime('%Y-%m')
    df_class_0 = pd.read_excel(os.path.join(Parameters.results_path,'df_class.xlsx'))
    df_class = pd.concat([df_class_0, df_class], axis = 0)
    df_class.to_excel(os.path.join(Parameters.results_path, 'df_class.xlsx'), index=False)
    
    return df_class