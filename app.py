import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# CONFIGURACIÓN GENERAL
# --------------------------------------------------
st.set_page_config(
    page_title="Insurance Company - EDA",
    layout="wide"
)

# --------------------------------------------------
# SIDEBAR IMAGEN
# --------------------------------------------------
st.sidebar.image("dmc.png", width="stretch")

# --------------------------------------------------
# CLASE POO
# --------------------------------------------------
class DataAnalyzer:

    def __init__(self, df):
        self.df = df

    def estadisticas(self):
        return self.df.describe()

# --------------------------------------------------
# SIDEBAR MENÚ
# --------------------------------------------------
st.sidebar.title("📌 Menú")

opcion = st.sidebar.selectbox(
    "Seleccione módulo",
    [
        "Módulo 1 – Home",
        "Módulo 2 – Carga del Dataset",
        "Módulo 3 – Análisis Exploratorio de Datos (EDA)",
        "Módulo 4 – Conclusiones Finales"
    ]
)

# --------------------------------------------------
# HOME
# --------------------------------------------------
def home():

    st.title("📊 Proyecto Final – Análisis de Dataset")

    st.write("**Estudiante:** Oscar Leonardo Montalván Villafuerte")
    st.write("**Curso:** Programación en Python")
    st.write("**Caso:** Insurance Company")
    st.write("**Año:** 2026")

    st.markdown("---")

    st.markdown("""
Esta aplicación permite realizar **análisis exploratorio de datos (EDA)** usando:

• Python  
• Streamlit  
• Pandas  
• Numpy  
• Matplotlib  
• Seaborn
""")

# --------------------------------------------------
# CARGA DATASET
# --------------------------------------------------
def carga_dataset():

    st.title("📂 Módulo 2 – Carga del Dataset")

    archivo = st.file_uploader("Cargar CSV", type=["csv"])

    if archivo is None:
        st.warning("Debe cargar un archivo CSV")
        return None

    df = pd.read_csv(archivo)

    st.success("Archivo cargado correctamente")

    st.subheader("Vista previa")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    col1.metric("Filas", df.shape[0])
    col2.metric("Columnas", df.shape[1])

    return df

# --------------------------------------------------
# EDA
# --------------------------------------------------
def eda(df):

    st.title("📊 Módulo 3 – Análisis Exploratorio de Datos")

    if df is None or df.empty:
        st.warning("Primero cargue el dataset")
        return

    # Columnas por tipo
    num = df.select_dtypes(include=['number']).columns.tolist()
    cat = df.select_dtypes(include=['object']).columns.tolist()

    analyzer = DataAnalyzer(df)

    # Crear pestañas
    tabs = st.tabs([
        "Ítem 1","Ítem 2","Ítem 3","Ítem 4","Ítem 5",
        "Ítem 6","Ítem 7","Ítem 8","Ítem 9","Ítem 10"
    ])

    # --------------------------------------------------
    # ÍTEM 1: Información general
    # --------------------------------------------------
    with tabs[0]:
        st.subheader("Información general")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Filas", df.shape[0])
        col2.metric("Columnas", df.shape[1])
        col3.metric("Numéricas", len(num))
        col4.metric("Categóricas", len(cat))

        st.write("Vista previa")
        st.dataframe(df.head())

        st.write("Tipos de datos")
        st.dataframe(df.dtypes)

        st.write("Valores nulos")
        st.bar_chart(df.isnull().sum())

    # --------------------------------------------------
    # ÍTEM 2: Clasificación de variables
    # --------------------------------------------------
    with tabs[1]:
        st.subheader("Clasificación de variables")
        st.write("Numéricas:", num)
        st.write("Categóricas:", cat)

        for columna in ["sourcing_channel", "residence_area_type"]:
            if columna in df.columns:
                st.write(columna)
                st.dataframe(df[columna].value_counts())

    # --------------------------------------------------
    # ÍTEM 3: Estadísticas descriptivas
    # --------------------------------------------------
    with tabs[2]:
        st.subheader("Estadísticas descriptivas")
        st.dataframe(analyzer.estadisticas())
        st.info("""
Media = promedio  
Mediana = valor central  
Desviación estándar = dispersión
""")

    # --------------------------------------------------
    # ÍTEM 4: Valores faltantes
    # --------------------------------------------------
    with tabs[3]:
        st.subheader("Valores faltantes")
        nulos = df.isnull().sum()
        st.dataframe(nulos)
        st.bar_chart(nulos)

    # --------------------------------------------------
    # ÍTEM 5: Distribución variables numéricas
    # --------------------------------------------------
    with tabs[4]:
        st.subheader("Distribución variables numéricas")
        if num:
            var = st.selectbox("Variable numérica", num, key="dist_num")
            fig, ax = plt.subplots()
            sns.histplot(df[var], kde=True, ax=ax)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No hay variables numéricas")

    # --------------------------------------------------
    # ÍTEM 6: Variables categóricas
    # --------------------------------------------------
    with tabs[5]:
        st.subheader("Variables categóricas")
        if cat:
            var = st.selectbox("Variable categórica", cat, key="dist_cat")
            conteo = df[var].value_counts()
            proporciones = df[var].value_counts(normalize=True)*100
            st.write("Conteos")
            st.dataframe(conteo)
            st.write("Proporciones (%)")
            st.dataframe(proporciones.round(2))
            st.bar_chart(conteo)
        else:
            st.info("No existen variables categóricas")

    # --------------------------------------------------
    # ÍTEM 7: Bivariado Numérico vs Categórico
    # --------------------------------------------------
    with tabs[6]:
        st.subheader("Bivariado Numérico vs Categórico")
        if num and cat:
            col1, col2 = st.columns(2)
            x = col1.selectbox("Numérica", num, key="biv_num")
            y = col2.selectbox("Categórica", cat, key="biv_cat")
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x=y, y=x, ax=ax)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Se requieren variables numéricas y categóricas")

    # --------------------------------------------------
    # ÍTEM 8: Bivariado Categórico vs Categórico
    # --------------------------------------------------
    with tabs[7]:
        st.subheader("Bivariado Categórico vs Categórico")
        ejemplos = ["residence_area_type","sourcing_channel"]
        target = "renewal"
        for v in ejemplos:
            if v in df.columns and target in df.columns:
                st.write(f"{v} vs {target}")
                tabla = pd.crosstab(df[v], df[target])
                st.dataframe(tabla)
                fig, ax = plt.subplots()
                tabla.plot(kind="bar", stacked=True, ax=ax)
                st.pyplot(fig)
                plt.close(fig)

    # --------------------------------------------------
    # ÍTEM 9: Correlación
    # --------------------------------------------------
    with tabs[8]:
        st.subheader("Correlación")
        columnas = st.multiselect("Variables", num, key="corr_vars")
        if len(columnas) >= 2:
            corr = df[columnas].corr()
            st.dataframe(corr)
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
            plt.close(fig)

    # --------------------------------------------------
    # ÍTEM 10: Hallazgos Clave
    # --------------------------------------------------
    with tabs[9]:
        st.subheader("📌 Hallazgos Clave")
        if {"renewal", "sourcing_channel", "residence_area_type"}.issubset(df.columns):
            resumen = pd.crosstab(
                [df["residence_area_type"], df["sourcing_channel"]], 
                df["renewal"]
            )
            st.write("### Resumen de renovaciones por canal y área de residencia")
            st.dataframe(resumen)
            fig, ax = plt.subplots(figsize=(10,5))
            resumen.plot(kind="bar", stacked=True, ax=ax)
            ax.set_title("Renovaciones según canal y área de residencia")
            ax.set_ylabel("Cantidad de clientes")
            ax.set_xlabel("Área de residencia y canal")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close(fig)
        st.info("""
**Insights principales derivados del EDA:**

1. Existen diferencias claras en la renovación entre segmentos de clientes según área de residencia y canal.  
2. Los clientes urbanos muestran mayor volumen de renovaciones que los rurales.  
3. Canales con menor volumen de clientes (como E) presentan baja actividad y podrían requerir revisión.  
4. El canal A es generalmente el más efectivo en términos de renovación.  
5. Los datos permiten priorizar canales y segmentos para futuras estrategias comerciales.
""")
        
        
# --------------------------------------------------
# CONCLUSIONES
# --------------------------------------------------
def conclusiones():
    st.title("🧠 Conclusiones Finales")

    st.info("""
1. **Diferencias entre segmentos de clientes:**  
   Se identifican diferencias significativas entre segmentos de clientes según variables demográficas como `residence_area_type` y `sourcing_channel`, lo que permite orientar estrategias comerciales diferenciadas.

2. **Renovación de pólizas:**  
   La variable `renewal` muestra asociación con factores de comportamiento y canales de adquisición, indicando que ciertos segmentos son más propensos a renovar pólizas.

3. **Comportamiento de variables numéricas:**  
   Algunas variables numéricas presentan alta dispersión y presencia de outliers, evidenciando heterogeneidad en los clientes y la necesidad de análisis estadísticos robustos para la toma de decisiones.

4. **Valores faltantes y calidad de datos:**  
   Se detectaron variables con valores faltantes, por lo que se recomienda imputación o limpieza de datos antes de aplicar modelos predictivos o análisis más avanzados.

5. **Relaciones bivariadas:**  
   El análisis bivariado (num vs num, num vs cat y cat vs cat) permitió identificar patrones relevantes, como la relación entre ingresos y renovación, así como diferencias entre canales de adquisición y renovación de clientes.
""")

# --------------------------------------------------
# FLUJO PRINCIPAL
# --------------------------------------------------
if opcion == "Módulo 1 – Home":
    home()

elif opcion == "Módulo 2 – Carga del Dataset":

    df = carga_dataset()

    st.session_state["df"] = df

elif opcion == "Módulo 3 – Análisis Exploratorio de Datos (EDA)":

    if "df" not in st.session_state:
        st.warning("Primero cargue el dataset")

    else:
        eda(st.session_state["df"])

elif opcion == "Módulo 4 – Conclusiones Finales":

    conclusiones()
