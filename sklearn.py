# Integrantes Equipo:
# Juan Luis Ivan Estrella Lopez
# Axel Leonardo Fernandez Albarran


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


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

# Seguir Axel, que es separar features y target, scalers, split-train y luego el modelo


