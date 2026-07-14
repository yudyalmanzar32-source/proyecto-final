import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

# Configuración
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 1. Generación de Dataset (Mismo del notebook original, adaptado para contexto dominicano)
def generar_dataset(n=4500, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)

    edad = rng.integers(14, 21, n)
    genero = rng.choice(['M', 'F'], n)
    zona = rng.choice(['Urbana', 'Rural'], n, p=[0.65, 0.35])

    nivel_socioeconomico = rng.choice(['Bajo', 'Medio', 'Alto'], n, p=[0.35, 0.45, 0.20])
    educacion_padre = rng.choice(['Ninguno', 'Primaria', 'Secundaria', 'Superior'], n, p=[0.15, 0.35, 0.35, 0.15])
    educacion_madre = rng.choice(['Ninguno', 'Primaria', 'Secundaria', 'Superior'], n, p=[0.12, 0.33, 0.37, 0.18])
    apoyo_familiar = rng.choice(['Bajo', 'Medio', 'Alto'], n, p=[0.25, 0.45, 0.30])

    vulnerabilidad = (
        (nivel_socioeconomico == 'Bajo').astype(float) * 0.3 +
        (apoyo_familiar == 'Bajo').astype(float) * 0.3 +
        (zona == 'Rural').astype(float) * 0.2 +
        rng.normal(0, 0.2, n)
    )

    tasa_asistencia = np.clip(95 - vulnerabilidad * 35 + rng.normal(0, 6, n), 30, 100)
    promedio_calificaciones = np.clip(80 - vulnerabilidad * 25 + rng.normal(0, 8, n), 30, 100)
    reprobaciones_previas = np.clip((vulnerabilidad * 4 + rng.poisson(0.6, n)).astype(int), 0, 8)
    incidencias_disciplinarias = np.clip((vulnerabilidad * 3 + rng.poisson(0.4, n)).astype(int), 0, 15)
    cambios_escuela = np.clip((vulnerabilidad * 1.5 + rng.poisson(0.2, n)).astype(int), 0, 5)
    distancia_escuela = np.clip(rng.exponential(3, n) + (zona == 'Rural').astype(float) * 5, 0, 50)

    trabaja_estudiante = (rng.random(n) < (0.10 + vulnerabilidad * 0.25)).astype(int)
    acceso_internet = (rng.random(n) > (0.15 + vulnerabilidad * 0.35)).astype(int)
    actividades_extracurriculares = (rng.random(n) > (0.4 + vulnerabilidad * 0.2)).astype(int)
    beca = (rng.random(n) < (0.25 - vulnerabilidad * 0.10)).astype(int)
    beca = np.clip(beca, 0, 1)

    # Score de riesgo combinado
    riesgo_score = (
        -0.05 * tasa_asistencia
        -0.04 * promedio_calificaciones
        +0.55 * reprobaciones_previas
        +0.30 * incidencias_disciplinarias
        +0.40 * cambios_escuela
        +0.04 * distancia_escuela
        +0.9 * trabaja_estudiante
        -0.6 * acceso_internet
        -0.5 * beca
        +1.5 * (nivel_socioeconomico == 'Bajo').astype(int)
        +1.2 * (apoyo_familiar == 'Bajo').astype(int)
        + rng.normal(0, 1.5, n)
    )
    prob_desercion = 1 / (1 + np.exp(-(riesgo_score - np.median(riesgo_score))))
    desercion = (rng.random(n) < prob_desercion * 0.42).astype(int)

    df = pd.DataFrame({
        'edad': edad,
        'genero': genero,
        'zona': zona,
        'promedio_calificaciones': np.round(promedio_calificaciones, 1),
        'tasa_asistencia': np.round(tasa_asistencia, 1),
        'reprobaciones_previas': reprobaciones_previas,
        'nivel_socioeconomico': nivel_socioeconomico,
        'educacion_padre': educacion_padre,
        'educacion_madre': educacion_madre,
        'trabaja_estudiante': trabaja_estudiante,
        'distancia_escuela': np.round(distancia_escuela, 1),
        'acceso_internet': acceso_internet,
        'actividades_extracurriculares': actividades_extracurriculares,
        'apoyo_familiar': apoyo_familiar,
        'incidencias_disciplinarias': incidencias_disciplinarias,
        'cambios_escuela': cambios_escuela,
        'beca': beca,
        'desercion': desercion
    })

    # Inyección de problemas de calidad de datos controlados (para replicar fase de limpieza)
    idx_nulos_padre = rng.choice(n, int(n * 0.032), replace=False)
    df.loc[idx_nulos_padre, 'educacion_padre'] = np.nan

    idx_nulos_dist = rng.choice(n, int(n * 0.018), replace=False)
    df.loc[idx_nulos_dist, 'distancia_escuela'] = np.nan

    idx_dup = rng.choice(n, 27, replace=False)
    df = pd.concat([df, df.loc[idx_dup]], ignore_index=True)

    idx_outlier = rng.choice(len(df), 12, replace=False)
    df.loc[idx_outlier, 'tasa_asistencia'] = rng.uniform(101, 130, 12)

    return df

print("1. Generando dataset con problemas de calidad...")
df_raw = generar_dataset()

# 2. Limpieza de Datos (Fase de Data Preparation en CRISP-ML)
print("2. Aplicando limpieza de datos...")
df_clean = df_raw.copy()

# Eliminar duplicados exactos
df_clean = df_clean.drop_duplicates().reset_index(drop=True)

# Corregir valores fuera de rango en tasa_asistencia (capping a 100)
df_clean['tasa_asistencia'] = df_clean['tasa_asistencia'].clip(upper=100)

# Imputación de nulos
moda_padre = df_clean['educacion_padre'].mode()[0]
df_clean['educacion_padre'] = df_clean['educacion_padre'].fillna(moda_padre)

mediana_dist = df_clean['distancia_escuela'].median()
df_clean['distancia_escuela'] = df_clean['distancia_escuela'].fillna(mediana_dist)

# Guardar dataset limpio para análisis y uso en la app de Streamlit
os.makedirs('assets', exist_ok=True)
df_clean.to_csv('assets/datos_estudiantes.csv', index=False)
print(f"   Dataset limpio guardado. Dimensiones: {df_clean.shape}")

# Mapeos ordinales
ordinal_maps = {
    'nivel_socioeconomico': {'Bajo': 0, 'Medio': 1, 'Alto': 2},
    'educacion_padre': {'Ninguno': 0, 'Primaria': 1, 'Secundaria': 2, 'Superior': 3},
    'educacion_madre': {'Ninguno': 0, 'Primaria': 1, 'Secundaria': 2, 'Superior': 3},
    'apoyo_familiar': {'Bajo': 0, 'Medio': 1, 'Alto': 2},
}

# 3. Preprocesamiento para Clasificación (Predecir Deserción)
print("3. Preparando datos para el modelo Clasificador (Deserción)...")
X_clf = df_clean.drop(columns=['desercion'])
y_clf = df_clean['desercion']

# Codificación ordinal
for col, mapping in ordinal_maps.items():
    X_clf[col] = X_clf[col].map(mapping)

# One-hot encoding manual para nominales (genero, zona)
# Usaremos dummy columns fijas para asegurar reproducibilidad en Streamlit
X_clf['genero_M'] = (X_clf['genero'] == 'M').astype(int)
X_clf['zona_Urbana'] = (X_clf['zona'] == 'Urbana').astype(int)
X_clf = X_clf.drop(columns=['genero', 'zona'])

# Lista de columnas numéricas para escalar
num_features_clf = ['edad', 'promedio_calificaciones', 'tasa_asistencia', 
                    'reprobaciones_previas', 'distancia_escuela', 
                    'incidencias_disciplinarias', 'cambios_escuela']

# Dividir dataset
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.15, stratify=y_clf, random_state=RANDOM_STATE
)

# Escalamiento Min-Max
scaler_clf = MinMaxScaler()
X_train_clf_scaled = X_train_clf.copy()
X_test_clf_scaled = X_test_clf.copy()

X_train_clf_scaled[num_features_clf] = scaler_clf.fit_transform(X_train_clf[num_features_clf])
X_test_clf_scaled[num_features_clf] = scaler_clf.transform(X_test_clf[num_features_clf])

# Aplicar SMOTE en entrenamiento
smote = SMOTE(sampling_strategy=0.6667, random_state=RANDOM_STATE)
X_train_res_clf, y_train_res_clf = smote.fit_resample(X_train_clf_scaled, y_train_clf)

# Entrenar Regresión Logística
print("   Entrenando Regresión Logística...")
model_lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
model_lr.fit(X_train_res_clf, y_train_res_clf)

# Entrenar Random Forest (Mejor modelo clasificador)
print("   Entrenando Random Forest...")
model_rf = RandomForestClassifier(
    n_estimators=200, 
    max_depth=10, 
    min_samples_leaf=3, 
    class_weight='balanced_subsample', 
    random_state=RANDOM_STATE,
    n_jobs=-1
)
model_rf.fit(X_train_res_clf, y_train_res_clf)

# 4. Preprocesamiento para Regresión (Predecir Promedio de Calificaciones)
print("4. Preparando datos para el modelo Regresor (Promedio de Calificaciones)...")
# Aquí, la variable objetivo es promedio_calificaciones.
X_reg = df_clean.drop(columns=['promedio_calificaciones', 'desercion'])
y_reg = df_clean['promedio_calificaciones']

# Codificación ordinal
for col, mapping in ordinal_maps.items():
    X_reg[col] = X_reg[col].map(mapping)

# One-hot encoding nominal
X_reg['genero_M'] = (X_reg['genero'] == 'M').astype(int)
X_reg['zona_Urbana'] = (X_reg['zona'] == 'Urbana').astype(int)
X_reg = X_reg.drop(columns=['genero', 'zona'])

# Columnas numéricas del regresor
num_features_reg = ['edad', 'tasa_asistencia', 'reprobaciones_previas', 
                    'distancia_escuela', 'incidencias_disciplinarias', 'cambios_escuela']

# Dividir dataset
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.15, random_state=RANDOM_STATE
)

# Escalamiento
scaler_reg = MinMaxScaler()
X_train_reg_scaled = X_train_reg.copy()
X_test_reg_scaled = X_test_reg.copy()

X_train_reg_scaled[num_features_reg] = scaler_reg.fit_transform(X_train_reg[num_features_reg])
X_test_reg_scaled[num_features_reg] = scaler_reg.transform(X_test_reg[num_features_reg])

# Entrenar Regresión Lineal
print("   Entrenando Regresión Lineal...")
model_lin = LinearRegression()
model_lin.fit(X_train_reg_scaled, y_train_reg)

# 5. Serialización de Modelos y Objetos de Preprocesamiento
print("5. Guardando binarios de modelos en carpeta 'assets'...")
joblib.dump(model_lr, 'assets/modelo_logistico.joblib')
joblib.dump(model_rf, 'assets/modelo_rf.joblib')
joblib.dump(model_lin, 'assets/modelo_lineal.joblib')
joblib.dump(scaler_clf, 'assets/scaler_clf.joblib')
joblib.dump(scaler_reg, 'assets/scaler_reg.joblib')

# Guardar nombres de columnas y mapeos
config = {
    'columns_clf': list(X_clf.columns),
    'columns_reg': list(X_reg.columns),
    'num_features_clf': num_features_clf,
    'num_features_reg': num_features_reg,
    'ordinal_maps': ordinal_maps
}
joblib.dump(config, 'assets/config.joblib')

print("¡Proceso completado exitosamente! Todos los modelos y configuraciones han sido almacenados.")
