import dill
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import numpy as np
import pandas as pd

# Custom transformers
from CustomTransformers import (
    ExpandContractions, Tokenizer, NormalizeText, 
    StemAndLemmatize, CombineOriginalStemLemma, TextVectorizer
)

class Model:
    def __init__(self, retraining_option='svc', columns=None):
       
        self.retraining_option = retraining_option
        self.columns = columns
        self.pipeline = self.create_pipeline()  

    def create_pipeline(self):

        steps = [
            ('expand_contractions', ExpandContractions()),
            ('tokenize', Tokenizer()),
            ('normalize_text', NormalizeText()),
            ('stem_and_lemmatize', StemAndLemmatize()),
            ('combine_original_stem_lemma', CombineOriginalStemLemma()),
            ('text_vectorizer', TextVectorizer()) 
        ]

        # Modificar el pipeline según la opción de reentrenamiento
        if self.retraining_option == 'naive_bayes_incremental':        
            steps.append(('model', MultinomialNB()))
        elif self.retraining_option == 'svc_add_new':
            steps.append(('model', SVC(probability=True)))
        else:
            steps.append(('model', SVC(probability=True)))
        return Pipeline(steps)

    def load_model(self, path):
        try:
            with open(path, 'rb') as f:
                model = dill.load(f)
            return model
        except FileNotFoundError:
            return None  
        
    # Método para reentrenar el pipeline y guardar el modelo, devolviendo métricas
    def retrain_model(self, X, y, save_path="model-group20.dill", X_old=None, y_old=None):
       
        if self.retraining_option == 'naive_bayes_incremental':
            # Cambiar el modelo a Naive Bayes si no está ya configurado
            if not isinstance(self.pipeline.named_steps['model'], MultinomialNB):
                self.pipeline.named_steps['model'] = MultinomialNB()

            # Transformar los datos nuevos con las partes previas del pipeline (sin incluir el modelo)
            steps_before_model = Pipeline(self.pipeline.steps[:-1])
            X_transformed = steps_before_model.fit_transform(X)

            # Ajustar el modelo Naive Bayes de manera incremental
            classes = np.unique(y) 
            self.pipeline.named_steps['model'].partial_fit(X_transformed, y, classes=classes)

            # Guardar el dataframe con la variable objetivo en un csv
            X_combined = pd.concat([X_old, X], axis=0)
            y_combined = np.concatenate([y_old, y], axis=0)
            X_combined['sdg'] = y_combined
            X_combined.to_csv('HistoryData.csv', index=False)

        elif self.retraining_option == 'svc_add_new':
            # Asegurarse de que los datos nuevos y viejos tengan las mismas columnas
            if not np.array_equal(X_old.columns, X.columns):
                raise ValueError("Las columnas de los datos antiguos y nuevos no coinciden")
            
            X_combined = pd.concat([X_old, X], axis=0)
            y_combined = np.concatenate([y_old, y], axis=0)

            #

            self.pipeline.fit(X_combined, y_combined)

            #guardar el dataframe combinado con la variable objetivo en un csv
            X_combined['sdg'] = y_combined
            X_combined.to_csv('HistoryData.csv', index=False)


        else:

            self.pipeline.fit(X, y)

            # Guardar el dataframe con la variable objetivo en un csv
            X['sdg'] = y
            X.to_csv('HistoryData.csv', index=False)

        y_pred = self.pipeline.predict(X)

        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')

        with open(save_path, 'wb') as file:
            dill.dump(self.pipeline, file)

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }