import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Clasificación de Clientes con IA",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Clasificación de Clientes con Algoritmos de IA")
from PIL import Image

# Carga tu logo
logo = Image.open("logo_sume_blanco.png")

# Opción A: Mostrarlo en la cabecera
#st.image(logo, width=200)

# Opción B: Mostrarlo en la barra lateral
st.sidebar.image(logo, width=250)
# Función para cargar datos
@st.cache_data
def cargar_datos(archivo):
    return pd.read_csv(archivo)

# Función para preprocesar datos
@st.cache_data
def preprocesar_datos(df):
    df_proc = df.copy()
    
    # Codificar variables categóricas
    categorical_columns = ['categoria_productos', 'canal_preferido', 'programa_lealtad', 'region']
    
    for col in categorical_columns:
        if col in df_proc.columns:
            le = LabelEncoder()
            df_proc[f'{col}_encoded'] = le.fit_transform(df_proc[col].astype(str))
    
    return df_proc

# Función para aplicar algoritmos de clustering/clasificación
def aplicar_algoritmo(df, columnas, algoritmo, parametros):
    X = df[columnas]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    try:
        if algoritmo == "K-Means":
            modelo = KMeans(n_clusters=parametros['n_clusters'], random_state=42)
            clusters = modelo.fit_predict(X_scaled)
            
        elif algoritmo == "DBSCAN":
            modelo = DBSCAN(eps=parametros['eps'], min_samples=parametros['min_samples'])
            clusters = modelo.fit_predict(X_scaled)
            
        elif algoritmo == "Gaussian Mixture":
            modelo = GaussianMixture(n_components=parametros['n_components'], random_state=42)
            clusters = modelo.fit_predict(X_scaled)
            
        elif algoritmo == "Agglomerative Clustering":
            modelo = AgglomerativeClustering(n_clusters=parametros['n_clusters'])
            clusters = modelo.fit_predict(X_scaled)
        
        # Calcular métricas
        metricas = {}
        if len(set(clusters)) > 1:
            metricas['silhouette'] = silhouette_score(X_scaled, clusters)
            metricas['calinski_harabasz'] = calinski_harabasz_score(X_scaled, clusters)
            metricas['n_clusters'] = len(set(clusters))
            
            if algoritmo == "DBSCAN":
                metricas['ruido'] = np.sum(clusters == -1)
        
        return clusters, metricas
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None

# Cargar datos
st.header("📁 Cargar Dataset")
archivo_cargado = st.file_uploader("Selecciona archivo CSV", type=['csv'])

if archivo_cargado is not None:
    df = cargar_datos(archivo_cargado)
    st.success(f"Dataset cargado: {len(df)} filas, {len(df.columns)} columnas")
    
    # Mostrar vista previa
    st.subheader("Vista previa de datos")
    st.dataframe(df.head())
    
    # Preprocesar datos
    df_procesado = preprocesar_datos(df)
    
    # Selección de variables
    st.header("🔧 Configuración")
    
    columnas_numericas = df_procesado.select_dtypes(include=[np.number]).columns.tolist()
    columnas_numericas = [col for col in columnas_numericas if 'id' not in col.lower()]
    
    columnas_seleccionadas = st.multiselect(
        "Variables para clasificación:",
        columnas_numericas,
        default=columnas_numericas[:4] if len(columnas_numericas) >= 4 else columnas_numericas
    )
    
    # Selección de algoritmo
    algoritmo = st.selectbox(
        "Algoritmo de IA:",
        ["K-Means", "DBSCAN", "Gaussian Mixture", "Agglomerative Clustering"]
    )
    
    # Parámetros específicos por algoritmo
    parametros = {}
    
    if algoritmo == "K-Means":
        parametros['n_clusters'] = st.slider("Número de clusters:", 2, 10, 4)
        
    elif algoritmo == "DBSCAN":
        parametros['eps'] = st.slider("Epsilon (eps):", 0.1, 2.0, 0.5, 0.1)
        parametros['min_samples'] = st.slider("Mínimo de muestras:", 2, 20, 5)
        
    elif algoritmo == "Gaussian Mixture":
        parametros['n_components'] = st.slider("Número de componentes:", 2, 10, 4)
        
    elif algoritmo == "Agglomerative Clustering":
        parametros['n_clusters'] = st.slider("Número de clusters:", 2, 10, 4)
    
    # Ejecutar clasificación
    if st.button("🚀 Ejecutar Clasificación") and columnas_seleccionadas:
        with st.spinner("Procesando..."):
            clusters, metricas = aplicar_algoritmo(
                df_procesado, columnas_seleccionadas, algoritmo, parametros
            )
            
            if clusters is not None:
                # Agregar resultados al dataframe
                df_resultado = df.copy()
                df_resultado['Cluster'] = clusters
                df_resultado['Segmento'] = df_resultado['Cluster'].apply(
                    lambda x: f'Segmento {x+1}' if x >= 0 else 'Ruido'
                )
                
                # Mostrar resultados
                st.header("📊 Resultados")
                
                # Métricas
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Clusters encontrados", metricas.get('n_clusters', 0))
                with col2:
                    st.metric("Silhouette Score", f"{metricas.get('silhouette', 0):.3f}")
                with col3:
                    if 'ruido' in metricas:
                        st.metric("Puntos de ruido", metricas['ruido'])
                    else:
                        st.metric("Calinski-Harabasz", f"{metricas.get('calinski_harabasz', 0):.1f}")
                
                # Distribución de clusters
                st.subheader("Distribución por Segmento")
                distribucion = df_resultado['Segmento'].value_counts()
                st.dataframe(distribucion)
                
                # Análisis por cluster
                st.subheader("Análisis por Segmento")
                variables_analisis = ['edad', 'ingresos_anuales', 'valor_total_compras', 
                                    'num_compras_ultimo_ano', 'ticket_promedio']
                variables_disponibles = [var for var in variables_analisis if var in df_resultado.columns]
                
                if variables_disponibles:
                    analisis_clusters = df_resultado.groupby('Segmento')[variables_disponibles].mean().round(2)
                    st.dataframe(analisis_clusters)
                
                # Visualización simple
                if len(columnas_seleccionadas) >= 2:
                    st.subheader("Visualización")
                    var_x = columnas_seleccionadas[0]
                    var_y = columnas_seleccionadas[1]
                    
                    chart_data = df_resultado[[var_x, var_y, 'Cluster']].copy()
                    chart_data = chart_data.rename(columns={var_x: 'x', var_y: 'y'})
                    
                    st.scatter_chart(
                        data=chart_data,
                        x='x',
                        y='y',
                        color='Cluster'
                    )
                
                # Descargar resultados
                st.subheader("💾 Descargar")
                csv_resultado = df_resultado.to_csv(index=False)
                st.download_button(
                    label="Descargar dataset clasificado",
                    data=csv_resultado,
                    file_name="clientes_clasificados.csv",
                    mime="text/csv"
                )
                
else:
    st.info("Carga un archivo CSV para comenzar la clasificación.")
