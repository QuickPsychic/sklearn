# Integrantes Equipo:
# Juan Luis Ivan Estrella Lopez
# Axel Leonardo Fernandez Albarran


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score


# Cargar DataSet
data = pd.read_csv("diabetes.csv")

# Nomas para ver primeras lineas
data.head()

#Agregar columna target (en este caso cambiar outcome a que sea target)
data = data.rename(columns={'Outcome': 'target'})


# Arreglar datos ceros para que sean Nulos
columns_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
data[columns_with_zeros] = data[columns_with_zeros].replace(0, np.nan)

# Agregar valores en base a la mediana
imputer = SimpleImputer(strategy="median")
data[columns_with_zeros] = imputer.fit_transform(data[columns_with_zeros])

# Separar features (X) y target (y)
X = data.drop(columns=['target'])
y = data['target']

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scalers
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Modelo AdaBoostClassifier (
ada_model = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=1.0,
    random_state=42,
)

# Entrenar el modelo
ada_model.fit(X_train, y_train)

# Sacar predicciones
predictions = ada_model.predict(X_test)

print("Predicciones:", predictions)
print("Accuracy:", accuracy_score(y_test, predictions))



