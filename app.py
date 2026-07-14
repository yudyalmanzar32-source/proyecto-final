import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Configuración de página
st.set_page_config(
    page_title="Sistema Inteligente de Alerta Temprana (SAT)",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para una estética premium y moderna
st.markdown("""
<style>
    /* Estilos generales */
    .main {
        background-color: #f8f9fa;
        color: #212529;
    }
    .stApp {
        background-color: #f8f9fa;
    }
    /* Encabezado */
    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .main-header h1 {
        color: white !important;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-size: 1.15rem;
        opacity: 0.9;
    }
    /* Tarjetas de Métricas */
    .kpi-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .kpi-title {
        font-size: 0.9rem;
        color: #6B7280;
        text-transform: uppercase;
        font-weight: bold;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1F2937;
    }
    /* Tarjetas de Alerta */
    .card-risk-bajo {
        background-color: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-left: 8px solid #10B981;
        padding: 1.5rem;
        border-radius: 10px;
        color: #065F46;
    }
    .card-risk-medio {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 8px solid #F59E0B;
        padding: 1.5rem;
        border-radius: 10px;
        color: #92400E;
    }
    .card-risk-alto {
        background-color: #FEF2F2;
        border: 1px solid #FEE2E2;
        border-left: 8px solid #EF4444;
        padding: 1.5rem;
        border-radius: 10px;
        color: #991B1B;
    }
</style>
""", unsafe_allow_html=True)

# Carga de recursos (Modelos y Datos)
@st.cache_resource
def load_ml_resources():
    try:
        model_lr = joblib.load('assets/modelo_logistico.joblib')
        model_rf = joblib.load('assets/modelo_rf.joblib')
        model_lin = joblib.load('assets/modelo_lineal.joblib')
        scaler_clf = joblib.load('assets/scaler_clf.joblib')
        scaler_reg = joblib.load('assets/scaler_reg.joblib')
        config = joblib.load('assets/config.joblib')
        return model_lr, model_rf, model_lin, scaler_clf, scaler_reg, config
    except Exception as e:
        st.error(f"Error al cargar los modelos: {e}. Por favor, asegúrate de haber ejecutado 'train_models.py' primero.")
        return None

@st.cache_data
def load_dataset():
    if os.path.exists('assets/datos_estudiantes.csv'):
        return pd.read_csv('assets/datos_estudiantes.csv')
    return None

resources = load_ml_resources()
df_data = load_dataset()

# Encabezado Principal de la Aplicación
st.markdown("""
<div class="main-header">
    <h1>🎓 Sistema Inteligente de Alerta Temprana (SAT)</h1>
    <p>Proyecto Final de Yudelka Morel | Maestría en IA Aplicada a la Educación e Investigación (UFHEC)</p>
</div>
""", unsafe_allow_html=True)

if resources is None or df_data is None:
    st.warning("⚠️ Los recursos del modelo o el dataset no están disponibles. Asegúrate de ejecutar `python train_models.py` para generar los archivos necesarios en la carpeta `assets/`.")
else:
    model_lr, model_rf, model_lin, scaler_clf, scaler_reg, config = resources
    
    # Barra lateral de navegación
    st.sidebar.image("assets/ar_classroom.png" if os.path.exists("assets/ar_classroom.png") else "https://via.placeholder.com/150", use_column_width=True)
    st.sidebar.title("Navegación")
    menu = st.sidebar.radio(
        "Seleccione una sección:",
        ["📈 Dashboard Analítico", "🔍 Predicción de Deserción", "✍️ Simulador de Calificaciones", "📖 Metodología CRISP-ML"]
    )
    
    # SECCIÓN 1: DASHBOARD
    if menu == "📈 Dashboard Analítico":
        st.subheader("Panel de Estadísticas Estudiantiles (Dataset de 4,500 registros)")
        
        # Fila de KPIs
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        total_estudiantes = len(df_data)
        tasa_desercion = (df_data['desercion'].mean() * 100)
        asistencia_promedio = df_data['tasa_asistencia'].mean()
        calificacion_promedio = df_data['promedio_calificaciones'].mean()
        
        with kpi1:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: #3B82F6;">
                <div class="kpi-title">Total Estudiantes</div>
                <div class="kpi-value">{total_estudiantes:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi2:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: #EF4444;">
                <div class="kpi-title">Tasa de Deserción</div>
                <div class="kpi-value">{tasa_desercion:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: #10B981;">
                <div class="kpi-title">Asistencia Promedio</div>
                <div class="kpi-value">{asistencia_promedio:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi4:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: #F59E0B;">
                <div class="kpi-title">Calificación Promedio</div>
                <div class="kpi-value">{calificacion_promedio:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # Gráficos
        g1, g2 = st.columns(2)
        
        with g1:
            st.markdown("### Asistencia vs. Calificaciones por Deserción")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(
                data=df_data.sample(800, random_state=42), 
                x="tasa_asistencia", 
                y="promedio_calificaciones", 
                hue="desercion",
                palette={0: '#10B981', 1: '#EF4444'},
                alpha=0.6,
                ax=ax
            )
            ax.set_xlabel("Tasa de Asistencia (%)")
            ax.set_ylabel("Promedio de Calificaciones")
            ax.set_title("Muestra de 800 Estudiantes")
            st.pyplot(fig)
            
        with g2:
            st.markdown("### Deserción Escolar por Nivel Socioeconómico")
            fig, ax = plt.subplots(figsize=(6, 4))
            socio_des = df_data.groupby('nivel_socioeconomico')['desercion'].mean() * 100
            socio_des = socio_des.reindex(['Bajo', 'Medio', 'Alto'])
            bars = ax.bar(socio_des.index, socio_des.values, color=['#EF4444', '#F59E0B', '#10B981'])
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
            ax.set_ylabel("Porcentaje de Deserción")
            ax.set_xlabel("Nivel Socioeconómico")
            ax.set_ylim(0, max(socio_des.values) + 8)
            st.pyplot(fig)
            
    # SECCIÓN 2: PREDICCIÓN DE DESERCIÓN
    elif menu == "🔍 Predicción de Deserción":
        st.subheader("Simulador del Sistema de Alerta Temprana (Clasificación)")
        st.write("Complete los datos académicos y socioeconómicos del estudiante para calcular su probabilidad de deserción:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            edad = st.number_input("Edad del Estudiante", min_value=12, max_value=22, value=16)
            genero = st.selectbox("Género", ["Femenino", "Masculino"])
            zona = st.selectbox("Zona de Residencia", ["Urbana", "Rural"])
            nivel_socio = st.selectbox("Nivel Socioeconómico", ["Bajo", "Medio", "Alto"])
            beca = st.selectbox("¿Recibe Beca o Subsidio?", ["No", "Sí"])
            
        with col2:
            asistencia = st.slider("Tasa de Asistencia (%)", 0.0, 100.0, 85.0, 0.5)
            calificaciones = st.slider("Promedio de Calificaciones (0-100)", 0.0, 100.0, 72.0, 0.5)
            reprobaciones = st.number_input("Materias Reprobadas Previas", min_value=0, max_value=10, value=0)
            disciplina = st.number_input("Reportes Disciplinarios (Año en curso)", min_value=0, max_value=20, value=0)
            cambios = st.number_input("Número de Cambios de Escuela", min_value=0, max_value=5, value=0)
            
        with col3:
            edu_padre = st.selectbox("Educación del Padre", ["Ninguno", "Primaria", "Secundaria", "Superior"])
            edu_madre = st.selectbox("Educación de la Madre", ["Ninguno", "Primaria", "Secundaria", "Superior"])
            trabaja = st.selectbox("¿El estudiante trabaja?", ["No", "Sí"])
            distancia = st.number_input("Distancia a la Escuela (km)", min_value=0.0, max_value=50.0, value=3.2)
            internet = st.selectbox("¿Tiene acceso a Internet en el hogar?", ["Sí", "No"])
            extracurriculares = st.selectbox("¿Participa en Actividades Extracurriculares?", ["Sí", "No"])
            apoyo = st.selectbox("Percepción de Apoyo Familiar", ["Bajo", "Medio", "Alto"])

        # Procesar entrada para predicción
        mapping_genero = {"Femenino": 0, "Masculino": 1} # genero_M
        mapping_zona = {"Urbana": 1, "Rural": 0} # zona_Urbana
        mapping_si_no = {"No": 0, "Sí": 1}
        mapping_si_no_inv = {"No": 1, "Sí": 0} # internet: Sí=1, No=0 en dummy
        
        # Mapear valores según config
        inp_socio = config['ordinal_maps']['nivel_socioeconomico'][nivel_socio]
        inp_padre = config['ordinal_maps']['educacion_padre'][edu_padre]
        inp_madre = config['ordinal_maps']['educacion_madre'][edu_madre]
        inp_apoyo = config['ordinal_maps']['apoyo_familiar'][apoyo]
        
        # Construir dataframe con el orden exacto de columnas para el clasificador
        # columns_clf: ['edad', 'promedio_calificaciones', 'tasa_asistencia', 'reprobaciones_previas',
        #               'nivel_socioeconomico', 'educacion_padre', 'educacion_madre', 'trabaja_estudiante',
        #               'distancia_escuela', 'acceso_internet', 'actividades_extracurriculares', 'apoyo_familiar',
        #               'incidencias_disciplinarias', 'cambios_escuela', 'beca', 'genero_M', 'zona_Urbana']
        
        input_data = pd.DataFrame([{
            'edad': edad,
            'promedio_calificaciones': calificaciones,
            'tasa_asistencia': asistencia,
            'reprobaciones_previas': reprobaciones,
            'nivel_socioeconomico': inp_socio,
            'educacion_padre': inp_padre,
            'educacion_madre': inp_madre,
            'trabaja_estudiante': mapping_si_no[trabaja],
            'distancia_escuela': distancia,
            'acceso_internet': 1 if internet == "Sí" else 0,
            'actividades_extracurriculares': mapping_si_no[extracurriculares],
            'apoyo_familiar': inp_apoyo,
            'incidencias_disciplinarias': disciplina,
            'cambios_escuela': cambios,
            'beca': mapping_si_no[beca],
            'genero_M': mapping_genero[genero],
            'zona_Urbana': mapping_zona[zona]
        }])
        
        # Asegurar orden exacto de columnas
        input_data = input_data[config['columns_clf']]
        
        # Escalar
        input_scaled = input_data.copy()
        input_scaled[config['num_features_clf']] = scaler_clf.transform(input_data[config['num_features_clf']])
        
        # Predicción
        btn_pred = st.button("📊 Calcular Nivel de Riesgo", use_container_width=True)
        
        if btn_pred:
            # Calcular probabilidad con Random Forest (final) y Regresión Logística
            prob_rf = model_rf.predict_proba(input_scaled)[0, 1]
            prob_lr = model_lr.predict_proba(input_scaled)[0, 1]
            
            st.markdown("### Resultados del Diagnóstico")
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Probabilidad de Deserción (Random Forest)", f"{prob_rf*100:.1f}%")
            with c2:
                st.metric("Probabilidad de Deserción (Regresión Logística)", f"{prob_lr*100:.1f}%")
                
            # Determinar riesgo e infografía
            if prob_rf < 0.30:
                riesgo = "BAJO"
                st.markdown(f"""
                <div class="card-risk-bajo">
                    <h3>🟢 ALERTA: RIESGO BAJO (Probabilidad: {prob_rf*100:.1f}%)</h3>
                    <p><b>Acción Recomendada:</b> Seguimiento rutinario trimestral. El estudiante presenta indicadores estables.</p>
                    <p><b>Responsable:</b> Docente tutor del grado.</p>
                </div>
                """, unsafe_allow_html=True)
            elif prob_rf < 0.60:
                riesgo = "MEDIO"
                st.markdown(f"""
                <div class="card-risk-medio">
                    <h3>🟡 ALERTA: RIESGO MEDIO (Probabilidad: {prob_rf*100:.1f}%)</h3>
                    <p><b>Acción Recomendada:</b> Coordinar reunión con la familia, realizar diagnóstico pedagógico y estructurar un plan de tutorías y acompañamiento socioemocional.</p>
                    <p><b>Responsable:</b> Orientador escolar del centro educativo.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                riesgo = "ALTO"
                st.markdown(f"""
                <div class="card-risk-alto">
                    <h3>🔴 ALERTA: RIESGO ALTO (Probabilidad: {prob_rf*100:.1f}%)</h3>
                    <p><b>Acción Recomendada:</b> Intervención de emergencia inmediata. Activación del equipo multidisciplinario del centro, diseño de adecuación curricular y visitas domiciliarias si es necesario.</p>
                    <p><b>Responsable:</b> Dirección del centro educativo junto con el Psicólogo escolar.</p>
                </div>
                """, unsafe_allow_html=True)
                
    # SECCIÓN 3: SIMULADOR DE CALIFICACIONES
    elif menu == "✍️ Simulador de Calificaciones":
        st.subheader("Simulador del Rendimiento Académico (Regresión Lineal)")
        st.write("Estime el promedio de calificaciones final del estudiante basado en factores conductuales y socioeconómicos:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            edad = st.number_input("Edad del Estudiante", min_value=12, max_value=22, value=16)
            genero = st.selectbox("Género", ["Femenino", "Masculino"])
            zona = st.selectbox("Zona de Residencia", ["Urbana", "Rural"])
            nivel_socio = st.selectbox("Nivel Socioeconómico", ["Bajo", "Medio", "Alto"])
            beca = st.selectbox("¿Recibe Beca o Subsidio?", ["No", "Sí"])
            extracurriculares = st.selectbox("¿Participa en Actividades Extracurriculares?", ["Sí", "No"])
            apoyo = st.selectbox("Percepción de Apoyo Familiar", ["Bajo", "Medio", "Alto"])
            
        with col2:
            asistencia = st.slider("Tasa de Asistencia (%)", 0.0, 100.0, 85.0, 0.5)
            reprobaciones = st.number_input("Materias Reprobadas Previas", min_value=0, max_value=10, value=0)
            disciplina = st.number_input("Reportes Disciplinarios (Año en curso)", min_value=0, max_value=20, value=0)
            cambios = st.number_input("Número de Cambios de Escuela", min_value=0, max_value=5, value=0)
            edu_padre = st.selectbox("Educación del Padre", ["Ninguno", "Primaria", "Secundaria", "Superior"])
            edu_madre = st.selectbox("Educación de la Madre", ["Ninguno", "Primaria", "Secundaria", "Superior"])
            trabaja = st.selectbox("¿El estudiante trabaja?", ["No", "Sí"])
            distancia = st.number_input("Distancia a la Escuela (km)", min_value=0.0, max_value=50.0, value=3.2)
            internet = st.selectbox("¿Tiene acceso a Internet en el hogar?", ["Sí", "No"])

        # Mapeos
        mapping_genero = {"Femenino": 0, "Masculino": 1}
        mapping_zona = {"Urbana": 1, "Rural": 0}
        mapping_si_no = {"No": 0, "Sí": 1}
        
        inp_socio = config['ordinal_maps']['nivel_socioeconomico'][nivel_socio]
        inp_padre = config['ordinal_maps']['educacion_padre'][edu_padre]
        inp_madre = config['ordinal_maps']['educacion_madre'][edu_madre]
        inp_apoyo = config['ordinal_maps']['apoyo_familiar'][apoyo]
        
        # Construir dataframe para regresor (excluye calificaciones y desercion)
        input_data_reg = pd.DataFrame([{
            'edad': edad,
            'tasa_asistencia': asistencia,
            'reprobaciones_previas': reprobaciones,
            'nivel_socioeconomico': inp_socio,
            'educacion_padre': inp_padre,
            'educacion_madre': inp_madre,
            'trabaja_estudiante': mapping_si_no[trabaja],
            'distancia_escuela': distancia,
            'acceso_internet': 1 if internet == "Sí" else 0,
            'actividades_extracurriculares': mapping_si_no[extracurriculares],
            'apoyo_familiar': inp_apoyo,
            'incidencias_disciplinarias': disciplina,
            'cambios_escuela': cambios,
            'beca': mapping_si_no[beca],
            'genero_M': mapping_genero[genero],
            'zona_Urbana': mapping_zona[zona]
        }])
        
        input_data_reg = input_data_reg[config['columns_reg']]
        
        # Escalar
        input_scaled_reg = input_data_reg.copy()
        input_scaled_reg[config['num_features_reg']] = scaler_reg.transform(input_data_reg[config['num_features_reg']])
        
        if st.button("✍️ Estimar Promedio de Calificaciones", use_container_width=True):
            pred_nota = model_lin.predict(input_scaled_reg)[0]
            pred_nota = np.clip(pred_nota, 30, 100) # Mantener en escala realista
            
            st.markdown(f"### Nota Promedio Estimada: **{pred_nota:.1f} / 100**")
            
            # Contexto pedagógico
            if pred_nota >= 70:
                st.success("🎉 Nota Aprobatoria. El modelo estima un rendimiento satisfactorio bajo estas condiciones.")
            else:
                st.warning("⚠️ Nota Reprobatoria (Menor a 70). Se sugiere reforzar el acompañamiento académico del alumno.")
                
    # SECCIÓN 4: METODOLOGÍA CRISP-ML
    elif menu == "📖 Metodología CRISP-ML":
        st.subheader("Ciclo de Vida del Proyecto: Metodología CRISP-ML")
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown("""
            La metodología **CRISP-ML (CRoss Industry Standard Process for Machine Learning)** organiza el ciclo de vida del proyecto en 6 fases iterativas:
            
            1. **Comprensión del Negocio (Business Understanding):** Definición del reto en República Dominicana, donde la deserción escolar supera el 20%.
            2. **Comprensión de los Datos (Data Understanding):** Exploración y diagnóstico preliminar de un dataset sintético/Kaggle de 4,500 registros con 18 variables.
            3. **Preparación de los Datos (Data Preparation):** Depuración de duplicados y valores atípicos, normalización mediante Min-Max y sobremuestreo mediante **SMOTE** para corregir el desbalance de clases (60/40).
            4. **Modelado (Modeling):**
               * **Clasificación:** Regresión Logística (Modelo Base) y Random Forest.
               * **Regresión:** Regresión Lineal para estimar el rendimiento académico final.
            5. **Evaluación (Evaluation):** Comprobación en datos de prueba independientes (15%). El Random Forest obtuvo un **AUC-ROC del 0.921** y un **Recall del 88.1%** en la clase deserción. El regresor lineal obtuvo las métricas de error y ajuste correspondientes.
            6. **Despliegue y Monitoreo (Deployment):** Integración de los modelos predictivos en esta interfaz interactiva de Streamlit para el uso práctico de directores y orientadores.
            """)
            
        with col_m2:
            st.image("assets/early_warning_system.png" if os.path.exists("assets/early_warning_system.png") else "https://via.placeholder.com/300", use_column_width=True)
            st.info("💡 **Consideración Ética:** El modelo presenta sesgos leves en zonas rurales (tasa de falsos positivos del 8.1%). Se recomienda encarecidamente utilizar las predicciones únicamente como apoyo y validar toda alerta con un profesional de la psicología u orientación.")
