from fastapi import FastAPI, HTTPException, File, UploadFile, Form
import pandas as pd
from io import StringIO
from PredictionModel import Model
from DataModel import DataModel, columns
from CustomTransformers import ExpandContractions, Tokenizer, NormalizeText, StemAndLemmatize, CombineOriginalStemLemma, TextVectorizer
import dill
from typing import List
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import mimetypes
from fastapi.middleware.cors import CORSMiddleware 
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas las solicitudes de cualquier origen
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos los headers
)

@app.get("/")
def read_root():
   return {"Hello": "World"}

@app.post("/predict")
def make_predictions(dataModels: List[DataModel]): 
    # Convertir los datos recibidos a un DataFrame
    df = pd.DataFrame([data.model_dump() for data in dataModels])

    with open("model-group20.dill", "rb") as f:
        model = dill.load(f)

    # Realizar las predicciones y calcular las probabilidades
    predicciones = model.predict(df)
    probabilidades = model.predict_proba(df)  

    # Crear una lista de diccionarios para devolver tanto la predicción como la probabilidad
    resultados = [
        {
            "prediccion": int(pred),  # La predicción de la clase
            "probabilidades": prob.tolist()  # Las probabilidades para todas las clases
        }
        for pred, prob in zip(predicciones, probabilidades)
    ]

    return {"resultados": resultados}

def load_previous_data():
    try:
        with open("HistoryData.csv", "rb") as file:
            df_old = pd.read_csv(file)
    
        X_old = df_old.drop('sdg', axis=1)  # Características antiguas
        y_old = df_old['sdg']  # Etiquetas antiguas
        return X_old, y_old
    except FileNotFoundError:
        return None, None 

@app.post("/retrain/")
async def retrain(
    file: UploadFile = File(...),
    retraining_option: str = Form('svc'),  # Por defecto es 'svc'
):
    try:
        # Obtener el tipo de archivo
        file_type = mimetypes.guess_type(file.filename)[0]
        

        # Leer el archivo según sea CSV o Excel
        if file_type == 'text/csv' or file.filename.endswith('.csv'):
            content = await file.read()
            content= io.BytesIO(content)
            df = pd.read_csv(content)
        elif file_type in ['application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'] or file.filename.endswith('.xlsx'):
            content = await file.read()
            df = pd.read_excel(content)
        else:
            raise HTTPException(status_code=400, detail="El archivo debe ser un CSV o Excel válido")

        # Validar las columnas necesarias
        expected_columns = columns()
        if not all(col in df.columns for col in expected_columns):
            raise HTTPException(status_code=400, detail=f"El archivo debe contener las columnas: {expected_columns}")

        # Separar las características (X) y la columna objetivo (y)
        X = df.drop("sdg", axis=1)
        y = df["sdg"]

        if retraining_option not in ['naive_bayes_incremental', 'svc', 'svc_add_new']:
            raise HTTPException(status_code=400, detail="Opción de reentrenamiento no válida. Usa 'naive_bayes_incremental', 'svc' o 'svc_add_new'.")

        if retraining_option in ['svc_add_new', 'naive_bayes_incremental']:
            X_old, y_old = load_previous_data()
        else:
            X_old, y_old = None, None

         # Crear una instancia del modelo
        model = Model(retraining_option=retraining_option, columns=df.columns)

        # Reentrenar el modelo y obtener métricas
        metrics = model.retrain_model(X, y, X_old=X_old, y_old=y_old)

        return {
            "message": f"Modelo reentrenado con la opcion de {retraining_option} y guardado con éxito.",
            "metrics": metrics 
        }

    except Exception as e:
        pass
        raise HTTPException(status_code=500, detail=f"Error durante el reentrenamiento: {str(e)}")