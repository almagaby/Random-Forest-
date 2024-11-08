from flask import Flask, render_template, send_file
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

# Función para particionar el conjunto de datos
def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return train_set, val_set, test_set

# Cargar el DataSet
df = pd.read_csv('TotalFeatures-ISCXFlowMeter.csv')
df['calss'] = df['calss'].factorize()[0]  # Transformar clase a numérica

# Partición del DataSet
train_set, val_set, test_set = train_val_test_split(df)
X_train, y_train = train_set.drop('calss', axis=1), train_set['calss']
X_val, y_val = val_set.drop('calss', axis=1), val_set['calss']
X_test, y_test = test_set.drop('calss', axis=1), test_set['calss']

# Escalado de datos
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Entrenamiento y evaluación de modelos
clf_rnd = RandomForestClassifier(n_estimators=10, random_state=42)
clf_rnd.fit(X_train, y_train)
y_val_pred = clf_rnd.predict(X_val)

# Métricas del modelo
f1_score_val = f1_score(y_val, y_val_pred, average="weighted")

# Regresión Random Forest para la comparación
le = LabelEncoder()
y_val_numeric = le.fit_transform(y_val)
reg_rnd = RandomForestRegressor(n_estimators=10, random_state=42)
reg_rnd.fit(X_train_scaled, y_train)
y_val_pred_reg = reg_rnd.predict(X_val_scaled)

mse = mean_squared_error(y_val_numeric, y_val_pred_reg)
r2 = r2_score(y_val_numeric, y_val_pred_reg)

# Ruta para la gráfica de dispersión
@app.route('/grafico_dispersion.png')
def grafico_dispersion():
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val_numeric, y_val_pred_reg, alpha=0.5)
    plt.plot([min(y_val_numeric), max(y_val_numeric)],
             [min(y_val_numeric), max(y_val_numeric)], 'r--')
    plt.xlabel("Valores reales")
    plt.ylabel("Valores predichos")
    plt.title("Comparación de valores reales vs. predichos")

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')

# Ruta para la página principal
@app.route('/')
def index():
    return render_template('index.html', f1_score_val=f1_score_val, mse=mse, r2=r2)

if __name__ == '__main__':
    app.run(debug=True)
