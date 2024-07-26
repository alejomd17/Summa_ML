# Librerías principales
# # ====================================================================================================
import numpy as np
import math

# Sarima
from pmdarima.arima import auto_arima

# Estacionalidad
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Random Forest
from sklearn.ensemble import RandomForestRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import grid_search_forecaster

# Lasso
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline

# XGB
from xgboost import XGBRegressor


import warnings
warnings.filterwarnings('ignore')


class Models():
    def sarima_model(df_train, df_test, steps):
        sarima = auto_arima(df_train.tolist(),
                                        start_p=0, start_q=0,
                                        test='adf',
                                        seasonal=True, 
                                        # stationary=True,
                                        m=12,
                                        lags=12,
                                        # start_P=0, start_Q=0,
                                        # max_p=3, max_q=3, 
                                        # max_P=5, max_Q=5,
                                        # d=1, D=1,
                                        # d_range=range(2), D_range=range(2),
                                        trace=True,
                                        error_action='ignore',
                                        suppress_warnings=True,
                                        # stepwise=True,
                                        random_state=123,   
                                        n_fits=50,
                                        # trend=None,
                                        # method='nm',
                                        # information_criterion='aic'
                            )
        pred_sarima= sarima.predict(n_periods=steps)
        
        return sarima, pred_sarima

    def lasso_model(df_train, df_test, steps):
        model_lasso = ForecasterAutoreg(
            regressor = make_pipeline(Lasso(random_state=123)),  # Este valor será remplazado en el grid search
            lags = 12)

        param_grid = {'lasso__alpha': np.logspace(-5, 5, 10)}

        lags_grid = [6, 12, 18] #36

        results_grid = grid_search_forecaster(
            forecaster          = model_lasso,
            y                   = df_train,  # Train and validation data
            param_grid          = param_grid,
            lags_grid           = lags_grid,
            steps               = steps,  # debería ser steps
            refit=False,
            metric              = 'mean_squared_error',
            # Model is trained with trainign data
            initial_train_size  = int(len(df_train) * 0.9),
            # fixed_train_size   = False,
            return_best=True,
            verbose=False
        )

        pred_lasso = model_lasso.predict(steps=steps)
        
        return model_lasso, pred_lasso
    
    def randomforest_model(df_train, df_test, steps):
        forecaster_rf = ForecasterAutoreg(
            regressor = RandomForestRegressor(random_state=123), # Este valor será remplazado en el grid search
            lags = 12)


        # Lags utilizados como predictores
        lags_grid = [3, 6, 12]

        # Hiperparámetros del regresor
        param_grid = {'n_estimators': [100, 500],
                'max_depth': [3, 5, 10]}
            
        resultados_grid = grid_search_forecaster(
                                forecaster         = forecaster_rf,
                                y                  = df_train,
                                param_grid         = param_grid,
                                lags_grid          = lags_grid,
                                steps              = steps,
                                refit              = False,
                                metric             = 'mean_squared_error',
                                initial_train_size = int(len(df_train)*0.5),
                                # fixed_train_size   = False,
                                return_best        = True,
                                verbose            = False
                        )

        pred_rf = forecaster_rf.predict(steps=steps)
        return forecaster_rf, pred_rf
        

    def xgboost_model(df_train, df_test, steps):
        forecaster_xgb = ForecasterAutoreg(
            regressor = XGBRegressor(random_state=123), # Este valor será remplazado en el grid search
            lags = 12)
        param_grid = {
            'n_estimators': [100, 300, 500],
            'max_depth': [3, 5, 10],
            'learning_rate': [0.01, 0.1]
            }


        # Lags used as predictors
        lags_grid = [8, 12, 24]


        results_grid = grid_search_forecaster(
                forecaster         = forecaster_xgb,
                y                  = df_train, # Train and validation data
                param_grid         = param_grid,
                lags_grid          = lags_grid,
                steps              = steps,
                refit              = False,
                metric             = 'mean_squared_error',
                initial_train_size = int(len(df_train)*0.5), # Model is trained with trainign data
                # fixed_train_size   = False,
                return_best        = True,
                verbose            = False
                )

        pred_xgb = forecaster_xgb.predict(steps=steps)
        
        return forecaster_xgb, pred_xgb