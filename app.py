import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Análisis Económico Gobierno Petro",
    layout="wide"
)


st.title("Análisis del Desempeño Económico durante el Gobierno de Gustavo Petro")
st.markdown("### Pontificia Universidad Javeriana")
st.markdown("#### Análisis Económico del Gobierno Actual")


@st.cache_data
def cargar_datos():
    try:
        ipc = pd.read_csv("ipc.csv")
        trm = pd.read_csv("TRM.csv")
        pib = pd.read_csv("pib.csv")
        desempleo = pd.read_csv("desempleo.csv")
        
        return ipc, trm, pib, desempleo
        
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        st.error("No se pueden cargar los archivos CSV necesarios. Cerrando la aplicación.")
        st.stop()  
        return None, None, None, None

ipc, trm, pib, desempleo = cargar_datos()

tab1, tab2, tab3 = st.tabs([
    "Análisis Univariado", 
    "Análisis Descriptivo",
    "Conclusión general"
])

def crear_histograma(datos, columna, titulo, ax):
    if columna in datos.columns:
        sns.histplot(datos[columna].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribución de {titulo}")
        ax.set_xlabel(titulo)
        ax.set_ylabel('Frecuencia')
    else:
        ax.text(0.5, 0.5, f'Columna {columna} no disponible', horizontalalignment='center', verticalalignment='center')

def crear_boxplot(datos, columna, titulo, ax):
    if columna in datos.columns:
        sns.boxplot(y=datos[columna].dropna(), ax=ax)
        ax.set_title(f"Boxplot de {titulo}")
        ax.set_ylabel(titulo)
    else:
        ax.text(0.5, 0.5, f'Columna {columna} no disponible', horizontalalignment='center', verticalalignment='center')

def mostrar_estadisticas(df, columna, titulo):
    if columna in df.columns:
        stats = df[columna].describe()
        st.write(f"### Estadísticas Descriptivas de {titulo}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Medidas de Tendencia Central:**")
            st.write(f"- Media: {stats['mean']:.4f}")
            st.write(f"- Mediana: {stats['50%']:.4f}")
            
 
            moda = df[columna].mode()
            if not moda.empty:
                st.write(f"- Moda: {moda.iloc[0]:.4f}")
            
        with col2:
            st.write("**Medidas de Dispersión:**")
            st.write(f"- Desviación Estándar: {stats['std']:.4f}")
            st.write(f"- Varianza: {(stats['std']**2):.4f}")
            st.write(f"- Rango: {stats['max'] - stats['min']:.4f}")
            st.write(f"- Coeficiente de Variación: {(stats['std']/stats['mean']*100):.2f}%")
        
        st.write("**Estadísticas Completas:**")
        st.dataframe(stats.to_frame().T)
        

        Q1 = stats['25%']
        Q3 = stats['75%']
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        
        outliers = df[(df[columna] < limite_inferior) | (df[columna] > limite_superior)]
        
        st.write(f"**Análisis de Valores Atípicos:**")
        st.write(f"- Número de outliers detectados: {len(outliers)}")
        st.write(f"- Porcentaje de outliers: {(len(outliers)/len(df)*100):.2f}%")
        
    else:
        st.write(f"Columna {columna} no disponible para {titulo}")


with tab1:
    st.header("Análisis univariado")
    st.subheader("Palabras más pronunciadas por el presidente en sus discursos")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("imagen1.png", caption="Análisis de frecuencia discurso 1")
    with col2:
        st.image("imagen2.png", caption="Análisis de frecuencia discurso 2")
    with col3:
        st.image("imagen3.png", caption="Análisis de frecuencia discurso 3")
        
    st.subheader("Análisis Univariado de Indicadores Económicos")


    indicador = st.selectbox(
        "Seleccione un indicador para ver su análisis univariado:",
        ["IPC", "TRM", "PIB", "Desempleo"]
    )
    
    if indicador == "IPC":
        st.subheader("Análisis del Índice de Precios al Consumidor (IPC)")
        
        st.write("#### Evolución Temporal del IPC")
        fig, ax = plt.subplots(figsize=(12, 6))
        crear_serie_temporal(ipc, None, "IPC", "IPC", ax)
        plt.tight_layout()
        st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            crear_histograma(ipc, "IPC", "IPC", ax)
            st.pyplot(fig)
            
            
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            crear_boxplot(ipc, "IPC", "IPC", ax)
            st.pyplot(fig)
        
        mostrar_estadisticas(ipc, "IPC", "IPC")
        
    elif indicador == "TRM":
        st.subheader("Análisis de la Tasa Representativa del Mercado (TRM)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            crear_histograma(trm, "TRM", "TRM", ax)
            st.pyplot(fig)
            
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            crear_boxplot(trm, "TRM", "TRM", ax)
            st.pyplot(fig)
        
        mostrar_estadisticas(trm, "TRM", "TRM")
        
    elif indicador == "PIB":
        st.subheader("Análisis del Producto Interno Bruto (PIB)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            crear_histograma(pib, "PIB", "PIB", ax)
            st.pyplot(fig)
            
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            crear_boxplot(pib, "PIB", "PIB", ax)
            st.pyplot(fig)
        
        mostrar_estadisticas(pib, "PIB", "PIB")
        
    elif indicador == "Desempleo":
        st.subheader("Análisis de la Tasa de Desempleo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            crear_histograma(desempleo, "desempleo", "desempleo", ax)
            st.pyplot(fig)
            
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            crear_boxplot(desempleo, "desempleo", "desempleo", ax)
            st.pyplot(fig)
        
        mostrar_estadisticas(desempleo, "desempleo", "desempleo")


with tab2:
    st.header("Análisis Descriptivo General")
    st.write("Esta sección presenta un resumen estadístico completo de todos los indicadores económicos.")
    
    st.subheader("Resumen Estadístico Comparativo")
    

    resumen_data = []
    

    indicadores_info = [
        (ipc, "IPC", "IPC"),
        (trm, "TRM", "TRM"), 
        (pib, "PIB", "PIB"),
        (desempleo, "desempleo", "Desempleo")
    ]
    
    for df, col, nombre in indicadores_info:
        if col in df.columns:
            stats = df[col].describe()
            resumen_data.append({
                'Indicador': nombre,
                'Media': stats['mean'],
                'Mediana': stats['50%'],
                'Desv. Estándar': stats['std'],
                'Mínimo': stats['min'],
                'Máximo': stats['max'],
                'CV (%)': (stats['std']/stats['mean']*100) if stats['mean'] != 0 else 0
            })
    
    if resumen_data:
        df_resumen = pd.DataFrame(resumen_data)
        st.dataframe(df_resumen)
        
 
        st.subheader("Comparación de Variabilidad")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_resumen, x='Indicador', y='CV (%)', ax=ax, palette='viridis')
        ax.set_title('Coeficiente de Variación por Indicador')
        ax.set_ylabel('Coeficiente de Variación (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
    

    st.subheader("Comparación de Distribuciones")
    

    indicadores_seleccionados = st.multiselect(
        "Seleccione los indicadores para comparar sus distribuciones:",
        ["IPC", "TRM", "PIB", "Desempleo"],
        default=["IPC", "TRM"]
    )
    
    if len(indicadores_seleccionados) >= 2:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for indicador in indicadores_seleccionados:
            if indicador == "IPC" and "IPC" in ipc.columns:
                datos_norm = (ipc["IPC"] - ipc["IPC"].min()) / (ipc["IPC"].max() - ipc["IPC"].min())
                sns.histplot(datos_norm, alpha=0.7, label=indicador, kde=True, ax=ax)
            elif indicador == "TRM" and "TRM" in trm.columns:
                datos_norm = (trm["TRM"] - trm["TRM"].min()) / (trm["TRM"].max() - trm["TRM"].min())
                sns.histplot(datos_norm, alpha=0.7, label=indicador, kde=True, ax=ax)
            elif indicador == "PIB" and "PIB" in pib.columns:
                datos_norm = (pib["PIB"] - pib["PIB"].min()) / (pib["PIB"].max() - pib["PIB"].min())
                sns.histplot(datos_norm, alpha=0.7, label=indicador, kde=True, ax=ax)
            elif indicador == "Desempleo" and "tasa" in desempleo.columns:
                datos_norm = (desempleo["desempleo"] - desempleo["desempleo"].min()) / (desempleo["desempleo"].max() - desempleo["desempleo"].min())
                sns.histplot(datos_norm, alpha=0.7, label=indicador, kde=True, ax=ax)
        
        ax.set_title('Comparación de Distribuciones Normalizadas')
        ax.set_xlabel('Valor Normalizado (0-1)')
        ax.set_ylabel('Frecuencia')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

with tab3:
    st.title("Conclusión general")
    st.write("Existe una relación evidente entre el discurso del presidente y el desempeño económico. Aunque el gobierno busca posicionar su narrativa como un proyecto de cambio estructural con justicia social, la forma en que se comunican estos objetivos influye fuertemente en la percepción económica y en la respuesta de los mercados. En general, los discursos han tenido un efecto mixto: cuando son técnicos y conciliadores, pueden generar confianza y estabilidad; sin embargo, cuando son ideológicos o confrontativos, tienden a generar incertidumbre y afectan negativamente al mercado. La coherencia entre la retórica y las acciones concretas será clave para mejorar la estabilidad y la confianza en la política económica.")
    

   
