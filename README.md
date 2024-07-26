<img src="./assets/logo_summa.png" alt="logo_summa" width="170"/>

# Desarrollo prueba técnica Profesional Machine Learning - SUMMA
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white&labelColor=3776AB)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com)

Este Repo **Summa_ML** es el desarrollo de la prueba técnica para **Profesional Machine Learning -SUMMA**
En este se desarrollaron los ejercicios practicos:
   
1. [Regresión: Pronóstico de la demanda](#regresión-pronóstico-de-la-demanda)
2. [Clasificación: Predicción de la clase](#clasificacion_predicción-de-la-clase)
3. [API](#API)
4. [Completar to_predict](#completar-to_predict)
5. [Docker](#docker)
6. [Teoría](#teoría)

## Introducción al Repo
#### Clona el repositorio:
   ```bash
   git clone https://github.com/alejomd17/Summa_ML.git
  ```
#### Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
#### Explora el Repo
<img src="./assets/Repo.JPG" alt="code" width="500"/>

   ```bash
 # Carpeta code
Se encuentran alojados los scripts con los que se dio solución a los ejercicios
* El jupyter summa_ml_full.ipynb resuelve los ejercicios de Regresión, Clasificación y Completar to_predict, este jupyter hace las veces de un Orchestrator que va utilizando diferentes modulos, que se encuentran en la carpeta src
* El script api_class_compras.py construye la API usando FastApi
```
<img src="./assets/code_folder.JPG" alt="code" width="200"/><img src="./assets/src_folder.JPG" alt="code" width="200"/>

   ```bash
 # Carpeta data
Se encuentran alojados los inputs y los outputs
* En la carpeta inputs, están los archivos iniciales brindados por Summa
* En la carpeta outputs, están las soluciones de los ejercicios, consta de tres carpetas:
** plots: donde queda alojada la gráfica de la predicción
** models: donde quedan alojados los modelos versionados ganadores de la predicción y la clasificación
** results: donde quedan alojados los archivos (excel) que dan respuesta de la predicción, clasificación y las métricas con las que se ganó en cada una, así cómo un archivo auxiliar para la clasificación, se explicará más adelante
   ```
<img src="./assets/data_folder.JPG" alt="code" width="300"/><img src="./assets/input_folder.JPG" alt="code" width="300"/><img src="./assets/output_folder.JPG" alt="code" width="300"/>
<img src="./assets/models_folder.JPG" alt="code" width="300"/><img src="./assets/plots_fodler.JPG" alt="code" width="300"/><img src="./assets/Results_folder.JPG" alt="code" width="300"/>

   ```bash
 # Carpeta devops
Se encuentran alojado el dockerfile y su respectiva app, para despliegue
   ```
<img src="./assets/devops_folder.JPG" alt="code" width="300"/><img src="./assets/app_folder.JPG" alt="code" width="300"/>


## Desarrollo de la prueba
## 1. Regresión: Pronóstico de la demanda
El desarrollo de este ejercicio se desarrolla en el script de jupyter [summa_ml_full](./code/summa_ml_full.ipynb) dentro de la carpeta [code](./code)\
Cuenta también con los modulos dentro de la carpeta [code/src](./code/src)

<img src="./assets/code_folder.JPG" alt="code" width="300"/><img src="./assets/src_folder.JPG" alt="code" width="300"/><img src="./assets/regression.JPG" alt="code" width="500"/>

En este, se busca encontrar un pronóstico sobre la demanda para los períodos 2022-05, 2022-06 y 2022-07.
Exporta los modulos y librerías creadas en [code/src](./code/src)
   ```bash
# Librerías
# =================
import os
import pandas as pd

import warnings;
warnings.simplefilter('ignore')

# Modulos
# =================
from src.parameters import Parameters
from src.pred_preprocessing import clean_atipicis, scalated_dataframe
from src.pred_predictions import fn_prediction, plot_pred
   ```
Se realiza la lectura del archivo input [dataset_demand_acumulate.csv](./data/input/dataset_demand_acumulate.csv)
   ```bash
# Datos
# =================
df_demanda   = pd.read_csv(os.path.join(Parameters.input_path, 'dataset_demand_acumulate.csv'))
   ```
Y se establecen parametros que son puestos a disposición en [parameters.py](./code/src/parameters.py)
   ```bash
# Parametros
# =================
steps                   = Parameters.steps   -----> 4 Pasos, que es el tiempo con el que se hará el testeo para encontrar el mejor modelo
col_pronos              = Parameters.col_pronos---> str: Columna Demand
col_y                   = Parameters.col_y--------> str: Columna yearmonth
scales                  = Parameters.scales-------> 4 escalas: Normal, [0,1], [-1,1] y logaritmica
   ```
Una vez establecido todo esto, se procede a ejecutar.
#### 1. Limpieza de datos
Los datos comienzan a ser limpiados, utilizando la función *clean_data* de [pred_preprocessing](./code/src/pred_preprocessing.py)
Esta función, evalúa la serie de tiempo recibida y determina que valores fueron atípicos según la diferencia entre la media y 2 veces la desviación estandar, y crea las columnas atipics (1 si es atípico, 0 si no), la columna Demand_with_atipics (que es la original) y renueva la columna Demand (en la que los datos que son atípicos, se interpola para eliminar los atipicos. Esta última, será la columna que se utilizará para el resto del ejercicio.
   ```bash
df_clean_atipics = clean_atipicis(df_demanda, col_pronos, col_y)
   ```
#### 2. Escalamiento de los datos
Se escalan los datos, es decir, se transforman según cada una de las escalas establecidas anteriormente (scales), utilizando la función *scalated_dataframe* de [pred_preprocessing](./code/src/pred_preprocessing.py)
Esto se crea con el fin de evaluar cada una de las posibles escalamiento de los datos para comparar entre ellas y elegir la que mejor respuesta otorgue.
   ```bash
df_standarized   = scalated_dataframe(df_clean_atipics, col_pronos)
   ```
#### 3. Predicción
El dataframe creado con todas las escalas se pasa por la función *fn_prediction* de [pred_predictions](./code/src/pred_predictions.py)
Esta función crea los dataframes de train y de test y evalúa cada una de los datos escalados.
   ```bash
df_pred = fn_prediction(df_standarized, steps, scales,col_pronos, col_y)
   ```
Los dataframe de train, test, escalados, así como los steps, col_pronos y scale son evaluados en la función *evaluate_models* de [pred_competition](./code/src/pred_competition.py)
En esta función, se pasan los argumentos mencionados por diferentes modelos, que están contemplados en el modulo [pred_models](./code/src/pred_models.py)
Para los cuales, devuelve tanto el modelo construído, como la predicción usando el df_test.

   ```bash
Los modelos utilizados son:
* Lasso
* Sarima
* Random Forest
* XGBoost
   ```
Cada uno de estos modelos se construyen sus métricas y los resultados son almacenados en 3 diccionarios, uno para los modelos (dict_models), uno para las metricas (dict_model) y otro para las predicciones (dict_pred)

Se itera y se agrupa por cada una de las escalas y finalmente se hace una competencia de modelos para elegir entre todos los modelos el que mejor métricas tenga, lo cuál generará los mejores pronosticos posibles. Esta competencia se lleva con la función *model_competition* de [pred_competition](./code/src/pred_competition.py) en donde se elige los resultados que tengan al menos 2 de 3 de las mejores métricas.

   ```bash
Las métricas utilizados son:
* RMSE
* MAPE
* R2
   ```
Esas métricas y pronósticos se almacenan en los resultados, en el archivo [winner_metrics_pred.xlsx](./data/output/results/winner_metrics_pred.xlsx)

<img src="./assets/winner_metrics_pred.JPG" alt="code" width="300"/>

Y el modelo ganador, que luego será el que se utilizará para los nuevos pronósticos se guarda como modelo versionado en la carpeta de modelos, para el ejercicio el ganador fue un modelo Random Forest con los datos escalados logaritmicamente [Demand_randomforest_Demand_scallg](./data/output/models/Demand_randomforest_Demand_scallg-1.0.pkl)

<img src="./assets/models_folder.JPG" alt="code" width="300"/>

Finalmente, ya con el modelo ganador obtenido, podremos realizar el pronóstico de los nuevos meses, utilizando el modelo ganador, y en caso de requerirlo, los resultados se "reescalan", es decir, se escalan inversamente para volver a tener los datos según su nominal original. Utilizamos la función *de_escalate* de [pred_preprocessing](./code/src/pred_preprocessing.py)

#### 4. Visualización y guardado de los resultados
Obtenido el df_pred, lo ilustraremos según los requerimientos del ejercicio.
Esto se realiza usando *plot_pred* de [pred_predictions](./code/src/pred_predictions.py)
   ```bash
plot_pred(df_demanda, df_pred, col_pronos, col_y)
   ```
En este, se crea un dataframe en donde se almacena la data real, los pronósticos del testeo y los resultados de la predicción, así cómo una fecha de corrida y el nombre de la variable, el cuál es el resultado final [df_plot.xlsx](./data/output/results/df_plot.xlsx)

<img src="./assets/df_plot.JPG" alt="code" width="300"/>

Y por último, se crea una visualización que se almacena en [pred_Demand.png](./data/output/plots/pred_Demand.png)

<img src="./data/output/plots/pred_Demand.png" alt="code" width="500"/>


## 2. Clasificación: Predicción de la clase
El desarrollo de este ejercicio también se desarrolla en el script de jupyter [summa_ml_full](./code/summa_ml_full.ipynb) dentro de la carpeta [code](./code)\
Cuenta también con los modulos dentro de la carpeta [code/src](./code/src)

<img src="./assets/code_folder.JPG" alt="code" width="300"/><img src="./assets/src_folder.JPG" alt="code" width="300"/><img src="./assets/classification.JPG" alt="code" width="500"/>

En este, se busca encontrar un pronóstico sobre la demanda para los períodos 2022-05, 2022-06 y 2022-07.
Exporta los modulos y librerías creadas en [code/src](./code/src)
   ```bash
# Librerías
# =================
import os
import pandas as pd

import warnings;
warnings.simplefilter('ignore')

# Modulos
# =================
from src.parameters import Parameters
from src.class_preprocessing import data_to_class
from src.class_classification import train_classification, fn_classification
   ```
Se realiza la lectura del archivo input [dataset_alpha_betha.csv](./data/input/dataset_alpha_betha.csv)
   ```bash
# Datos
# =================
df_variables    = pd.read_csv(os.path.join(Parameters.input_path, 'dataset_alpha_betha.csv'))
   ```
Una vez establecido todo esto, se procede a ejecutar.
#### 1. Entrenamiento
Se dispone a entrenar el modelo que clasifique, según sus caracteristicas, si una compra es [Alpha, Betha].
Primero, debemos obtener cuales son las variables predictoras(X) y predichas (y). Utilizamos entonces la función *data_to_class* de [class_preprocessing](./code/src/class_preprocessing.py)
   ```bash
X, y            = data_to_class(df_variables, train = True)
   ```
En este, se hace la limpieza, transformación de las caracteristicas para tener un dataset mas homogeneo y poder convertir las variables a categoricas, según sea el caso.
Una vez obtenidos todos y cada uno de los posibles variables dummies que podamos, vamos a guardar un archivo auxiliar que nos ayudará cuando el desarrollo este en productivo en la API y en el Docker. Este archivo es [df_col_dummies.xlsx](./data/output/results/df_col_dummies.xlsx)

<img src="./assets/cols_dummies.JPG" alt="code" width="300"/>

Cuando obtenemos la X, y, podemos realizar el entrenamiento para obtener el modelo con la función *train_classification* de [class_classification](./code/src/class_classification.py)
   ```bash
train_classification(X, y)
   ```
En esta función se particionan los dataframes en X_train, X_test, y_train, y_test y se evalúan en la función *evaluate_models_classification* de [class_competition](./code/src/class_competition.py)
Similar a como se hizo con la predicción se evalúan varios modelos, que generan ciertas métricas y pronósticos que se almacenan en los diccionarios respectivos (dict_models, dict_pred, dict_metrics)

   ```bash
Los modelos utilizados son:
* Logistic Regression
* Logistic Regression Multivariable
* KNeighbors
* DecisionTree
* SVC  Vectores de soporte
* Random Forest
* Gaussian NB

Las métricas utilizados son:
* Precision
* Recall
* f1-score
   ```
Luego, se hace la competencia de modelos para elegir entre todos los modelos el que mejor métricas tenga, lo cuál generará los mejores pronosticos posibles. Esta competencia se realiza con la función *class_competition* de [class_competition](./code/src/class_competition.py) en donde se elige los resultados que tengan al menos 2 de 3 de las mejores métricas.

Esas métricas y pronósticos ganadores se almacenan en los resultados, en el archivo [winner_metrics_class.xlsx](./data/output/results/winner_metrics_class.xlsx)

<img src="./assets/winner_metrics_class.JPG" alt="code" width="300"/>

Y el modelo ganador, que luego será el que se utilizará para los nuevos pronósticos se guarda como modelo versionado en la carpeta de modelos, para el ejercicio el ganador fue un modelo Sarima con los datos escalados logaritmicamente [lrm](./data/output/models/lrm-1.0.pkl)

<img src="./assets/models_folder.JPG" alt="code" width="300"/>

#### 2. Solución con to_predict
> [!NOTE]
> Acá se soluciona el ejercicio 4 de completar el to_predict

Con el modelo entrenado, podemos hacer la predicción con los datos de [to_predict.csv](./data/input/to_predict.csv)
Utilizamos también como soporte las columnas dummies mencionadas anteriormente y llenamos con los datos pronosticados en demandas del ejercicio 1 los datos de la demanda.
Construimos los X, y. y finalmente hacemos el pronóstico usando la función *fn_classification* del modulo [class_classification](./code/src/class_classification.py) 

   ```bash
df_to_pred                            = pd.read_csv(os.path.join(Parameters.input_path, 'to_predict.csv'))
df_pred_demand                        = pd.read_excel(os.path.join(Parameters.results_path, 'df_plot.xlsx'))
df_to_pred['Demand']                  = df_pred_demand[['year_month','pred']][-3:]['pred'].values
X, y                                  = data_to_class(df_to_pred)
df_class, df_class_consolidate        = fn_classification(df_to_pred, X)
   ```
Con esta última función, el dataframe se organiza para que quede con las mismas variables dummies originales y buscamos el último modelo ganador versionado con el cuál hacemos el pronóstico de los datos de [to_predict.csv](./data/input/to_predict.csv)
Estos resultados se almacenan en un dataframe junto con la fecha de corrida, el cuál es el resultado final y se guarda en [df_class.xlsx](./data/output/results/df_class.xlsx)

<img src="./assets/df_class.JPG" alt="code" width="500"/>

## 3. API
La API esta diseñada con [FastAPI](https://fastapi.tiangolo.com/)
Se desarrollo un ejemplo manual del funcionamiento del API en el script de jupyter [summa_ml_full](./code/summa_ml_full.ipynb) dentro de la carpeta [code](./code)\
Usando el modulo [manual_api_class_compras.py](./code/src/manual_api_class_compras.py) el cuál, es una representación igual de la API diseñada.

<img src="./assets/api.JPG" alt="code" width="400"/>

La API es entonces la llamada [api_class_compras.py](./code/api_class_compras.py) dentro de la carpeta code, donde se usa la función *class_compras*
En esta, se deben pasar los datos de cada una de las columnas, salvo la Clase que será la que se predice.
Dado esto, se construye un dataframe, similar a los que se usan (Como el to_predict) y se pasa por el predictor, es decir, se utiliza, como ya lo habíamos visto, la función *fn_classification* del modulo [class_classification](./code/src/class_classification.py) 
Esta API devuelve un JSON con dos salidas, la primera, es el resultado de la predicción [Alpha, Betha], y la otra es el DataFrame ya con dicho resultado.

### Para ejecutar la API en Bash
Para ejecutar la API en Bash, es decir, en el Navegador, debemos correr el script que contiene la API, puesto que esta ya corre el uvicorn en la terminal *127.0.0.1:8000*
   ```bash
Summa_ML (main)
py code/api_class_compras.py
   ```
<img src="./assets/fastapi_bash.JPG" alt="code" width="400"/>

Esto nos disponibiliza el host mencionado, y entramos directamente a la documentación de la API
   ```bash
http://127.0.0.1:8000/docs
   ```
En esta, podemos ver la documentación de la API, y podemos ver los requerimientos que tiene

<img src="./assets/fastapi_ex.JPG" alt="code" width="400"/>

Así como los resultados de la misma, con un caso de ejemplo:

<img src="./assets/fastapi_res.JPG" alt="code" width="400"/>

Misma validación puede hacerse también en otras plataformas como Postman.

## 4. Completar to_predict
> [!NOTE]
> El pronostico del to_predict, ya fue desarrollado con el ejercicio #2, y puede hacerse de nuevo individualmente con la API

Con el modelo entrenado, podemos hacer la predicción con los datos de [to_predict.csv](./data/input/to_predict.csv)
Utilizamos también como soporte las columnas dummies mencionadas anteriormente y llenamos con los datos pronosticados en demandas del ejercicio 1 los datos de la demanda.
Construimos los X, y. y finalmente hacemos el pronóstico usando la función *fn_classification* del modulo [class_classification](./code/src/class_classification.py) 
<img src="./assets/api.JPG" alt="code" width="400"/>
   ```bash
df_to_pred                            = pd.read_csv(os.path.join(Parameters.input_path, 'to_predict.csv'))
df_pred_demand                        = pd.read_excel(os.path.join(Parameters.results_path, 'df_plot.xlsx'))
df_to_pred['Demand']                  = df_pred_demand[['year_month','pred']][-3:]['pred'].values
X, y                                  = data_to_class(df_to_pred)
df_class, df_class_consolidate        = fn_classification(df_to_pred, X)
   ```
Con esta última función, el dataframe se organiza para que quede con las mismas variables dummies originales y buscamos el último modelo ganador versionado con el cuál hacemos el pronóstico de los datos de [to_predict.csv](./data/input/to_predict.csv)
Estos resultados se almacenan en un dataframe junto con la fecha de corrida, el cuál es el resultado final y se guarda en [df_class.xlsx](./data/output/results/df_class.xlsx)

<img src="./assets/df_class.JPG" alt="code" width="500"/>

## Docker
Para construir la imagen y el contenedor del Docker, procedemos a copiar la API generada dentro de la carpeta donde se almacenará la app que ejecute la imagen.
Entonces, se crea la carpeta [devops](./devops/) y la aplicación a la carpeta [app](./devops/app). Para este ejercicio, la API pasa a llamarse [main.py](./devops/app/main.py) y se modifican las rutas que así lo requieran.

<img src="./assets/devops_folder.JPG" alt="code" width="300"/><img src="./assets/app_folder.JPG" alt="code" width="300"/>

Dentro de [devops](./devops/) se crea el [dockerfile](./devops/dockerfile) y se copian los requirements.txt que necesita la app para funcionar.

<img src="./assets/docker_file.JPG" alt="code" width="300"/>

docker_bash_create_Image.JPG

Creamos la imagen del Docker en bash
   ```bash
Summa_ML (main)
docker build -t api_compras .
   ```
<img src="./assets/fastapi_bash.JPG" alt="code" width="400"/>

Ya con esto queda creada la imagen en Docker y se puede disponer de un contenedor para ejecutar la app

<img src="./assets/docker_image.JPG" alt="code" width="400"/>

<img src="./assets/docker_container.JPG" alt="code" width="400"/>

> [!NOTE]
> Con esto queda contruido el contenedor, y la API ya funciona tal como se mostro en el ejercicio 3 (donde se creo la API con FastAPI)

## 6. Teoría
> [!NOTE]  
> Las respuestas de la parte teórica se encuentran en [teoria_.pdf](./devops/dockerfile), lo que esta escrito a continuación, es lo mismo que esta en dicho archivo.

### 1. 
En la empresa GA, en el área de compras necesitan CLASIFICAR y organizar los correos que llegan a la bandeja de entrada entre 4 tipos de correos (Compras cementos, Compras energía, Compras concretos y correos generales o de otra índole). Esta tarea se le encomienda a usted, como es el Gestor SR en temas de analítica e IA puede solicitar al área interesada los recursos humanos que necesite para llevar a cabo este proyecto, también puede solicitar en tecnología todo lo que necesite, además tiene las bandejas de entrada de correos históricos de los analistas que reciben estas solicitudes con aproximadamente: 5500 correos de compras cementos, 2700 correos de compras de energía, 1100 correos de compras concretos y 12876 correos generales o de otra índole.
Explique como resolvería este problema, metodología, algoritmos, modelos, arquitectura del proyecto etc.
   ```bash
Para empezar, se debe estructurar esto como un proyecto, con su respectiva metodología, repositorios, ambientes y miembros.
Procedemos al entendimiento del negocio, objetivos y posibles soluciones. así como de los datos, para poder identificar que datos sirven, y como se deben obtener.
Una vez identificados las fuentes de los datos y los datos, se comienza con un proceso de guardado de la información de tal suerte que tenga una estructura adecuada para empezar. Se puede hacer uso de APIs de lecturas de correos, o si esta en un correo de microsoft hacer uso de Azure, usando por ejemplo Graph, o en su defecto un RPA.
Junto a esto se define el espacio donde se van a almacenar y si los datos serán estructurados o no estructurados.
Finalmente podemos proceder con un modelo de clasificación de texto empleando modelos de clasificación como KNN, Naive Bayesiano Supervisado o vectores de soporte SVM y random forest, arboles de decisión, redes neuronales e incluso PLN.  
Otro recurso a usarse pueden ser los PLN, LLM y se puede auxiliar de librerías como Transformers de HuggingFace con datos ya entrenados, o una elaboración propia de un diccionario con palabras que tienen posibilidad de aparecer en determinado correo y se categorizan.

Obtenido el resultado, se disponibiliza el modelo ya sea con una API, docker o ya en Azure con un procesamiento orchestado desde Data Factory o Databricks, o que se habilite azure function o logic app.

Por último, si es necesario, se construye un front ya sea cómo plataforma web o en tableros de PBI.

1. Caso de negocio
2. Entendimiento del negocio y data
3. Obtener información, limpiarla y almacenarla.
4. Modelación
5. Orchestation, llevar a productivo, disponibilizar el modelo.
6. Exposición o visualización (Si aplica)
   ```
### 2.
Seis meses después de haber desplegado un modelo de regresión en producción, los usuarios se dan cuenta que las predicciones que este está dando no son tan acertadas, se le encarga a usted como Gestor SR en temas de IA que revise que puede estar sucediendo.
¿Cree que el modelo esté sufriendo Drift?
¿Cómo puede validarlo?
¿De ser así, que haría usted para corregir esto?
Explique sus respuestas.
  ```bash
Si, el modelo puede estar perdiendo predicción dado que esta dejando de lado los eventos y situaciones que vienen ocurriendo en los últimos 6 meses que afectan el comportamiento de la predicción, por tanto, es importante tener un tiempo de reentramiento más corto para que estos efectos no sigan ocurriendo. La forma de evidenciar si está o no perdiendo efectividad es calculando las predicciones versus la realidad y calcular el valor de algunas métricas de acierto de predicción y compararlas con las que dieron en la última puesta en producción en su entrenamiento, si es evidente su caída, podemos asumir que el modelo perdió asertividad.
Como lo mencioné anteriormente, una de las formas de asegurar que esto no ocurra, es reentrenando los modelos con menor tiempo, para así posibilitar que los modelos aprendan lo que sucedió recientemente y calculen un pronóstico mucho más semejante a la realidad, puesto que es probable que ocurriesen eventos de datos atípicos, tendencias, estaciones o incluso situaciones geopolíticas.
Si es muy difícil poder lograr que se reentrene por los altos costos, se debería hacer una especie de alerta que muestre el nivel de acierto que va teniendo semana a semana y poder haciendo ajustes que no requieran modelación (por ejemplo, la suma de datos atípicos, etc).
   ```
### 3.
3.	Su equipo de trabajo está trabajando en un chatbot con generación de texto utilizando el modelo GPT-3.5, según cómo funciona este modelo, ¿cómo haría usted para hacer que las respuestas del chatbot estén siempre relacionadas a conseguir cierta información particular del usuario y no empiece a generar texto aleatorio sobre cualquier tema? 
Explique su respuesta.
  ```bash
Según se indica, el chatbot es propio del desarrollo, por ende, se puede personalizar, ajustar los parametros, mantener prompts específicos y dando contexto persistentemente.
Por último, se puede hacer fine tunning o Few-shot Learning
   ```