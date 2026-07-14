# 🎓 Sistema Predictivo de Deserción Escolar y Alerta Temprana (SAT)

Este repositorio contiene los archivos del **Proyecto Final Integrador** para la asignatura de **Fundamentos de IA y Aprendizaje Automático** de la Maestría en Inteligencia Artificial Aplicada a la Educación e Investigación (UFHEC). 

El proyecto implementa un sistema completo que detecta el riesgo de abandono escolar en centros educativos de República Dominicana mediante aprendizaje automático supervisado.

**Autora:** Yudelka Morel  
**Profesor:** Dr. Darwin Muñoz, Ph.D  
**Fecha:** Junio 2026

---

## 🚀 Arquitectura del Proyecto

El repositorio está estructurado de la siguiente forma:

* `Modelo_Desercion_Escolar_CRISP_ML.ipynb`: Cuaderno Jupyter estructurado bajo el proceso CRISP-ML que detalla el modelado predictivo de clasificación (Regresión Logística, Random Forest) y regresión (Regresión Lineal).
* `app.py`: Aplicación web interactiva desarrollada en **Streamlit** que sirve como interfaz operativa del Sistema de Alerta Temprana y simulador de promedio de calificaciones.
* `train_models.py`: Script auxiliar de Python para automatizar el preprocesamiento de datos, el entrenamiento de modelos y la exportación de archivos binarios `.joblib`.
* `index.html` y `styles.css`: Landing Page interactiva premium que expone de forma visual el ciclo de vida del proyecto, las métricas clave, el sistema de alertas y el Currículum Vitae (CV) de la autora.
* `assets/`: Directorio que contiene el dataset limpio en formato CSV y las imágenes de Realidad Aumentada generadas por IA para la landing page.
* `requirements.txt`: Dependencias necesarias para la ejecución local.
* `.gitignore`: Exclusiones para el control de versiones.

---

## 📈 Resumen del Modelo y Resultados

Bajo la metodología **CRISP-ML**, el proyecto arrojó los siguientes resultados tras evaluar los modelos sobre un conjunto de prueba independiente (15%):

1. **Modelo Clasificador (Deserción):**
   * **Random Forest (Seleccionado):** Obtuvo un **AUC-ROC del 0.921** y una tasa de recuperación (Recall/Sensibilidad) de la deserción del **88.1%**, minimizando los falsos negativos.
   * **Regresión Logística:** AUC-ROC de 0.831 y Recall de 72.4%.
2. **Modelo Regresor (Calificaciones):**
   * **Regresión Lineal:** Logró estimar el promedio final del estudiante con un RMSE de **5.82 puntos** y un coeficiente de ajuste $R^2$ de **0.781**.

### Protocolos del Sistema de Alerta Temprana (SAT)
Las probabilidades estimadas por el clasificador se dividen en tres niveles de riesgo accionables:
* 🟢 **Bajo (0% - 30%):** Seguimiento trimestral rutinario a cargo del docente tutor.
* 🟡 **Medio (30% - 60%):** Reunión de apoyo familiar e inscripción en tutorías por parte del Orientador Escolar.
* 🔴 **Alto (60% - 100%):** Intervención inmediata a cargo del Director Escolar y Psicólogo del centro.

---

## 🛠️ Instalación y Ejecución Local

### Prerrequisitos
Debe contar con **Python 3.8 o superior** instalado en su sistema.

### Paso 1: Clonar o descargar el repositorio
Descargue los archivos del proyecto a su máquina local y abra una terminal en esa carpeta.

### Paso 2: Instalar dependencias
Instale las librerías necesarias ejecutando:
```bash
pip install -r requirements.txt
```

### Paso 3: Entrenar modelos y generar binarios (Opcional)
Para generar el conjunto de datos limpio y entrenar los modelos guardando los binarios, ejecute:
```bash
python train_models.py
```
*Nota: Los binarios generados se almacenarán en la carpeta `assets/`.*

### Paso 4: Ejecutar la aplicación Streamlit
Inicie el panel interactivo local ejecutando:
```bash
streamlit run app.py
```
La aplicación se abrirá automáticamente en su navegador web en `http://localhost:8501`.

### Paso 5: Abrir la Landing Page
Para visualizar la landing page interactiva con el portafolio y los resultados, simplemente haga doble clic en el archivo `index.html` para abrirlo en cualquier navegador web moderno.

---

## 🐙 Despliegue en GitHub

Para subir este proyecto a su cuenta personal de GitHub, siga los siguientes pasos desde su terminal:

1. **Inicializar Git:**
   ```bash
   git init
   ```
2. **Agregar archivos y hacer commit inicial:**
   ```bash
   git add .
   git commit -m "Commit inicial: Proyecto Predictivo de Deserción Escolar CRISP-ML"
   ```
3. **Vincular repositorio remoto y subir cambios:**
   *Cree un nuevo repositorio público en GitHub (sin README ni .gitignore) y ejecute:*
   ```bash
   git remote add origin https://github.com/SU_USUARIO/SU_REPOSITORIO.git
   git branch -M main
   git push -u origin main
   ```
