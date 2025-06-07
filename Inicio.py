import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(
    page_title="Segmentación de Clientes",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🎯 Segmentación de Clientes con Machine Learning")
st.markdown("### Análisis y clustering de clientes para estrategias de ventas personalizadas")

# Sidebar para configuraciones
st.sidebar.header("⚙️ Configuraciones")

# Función para cargar datos
@st.cache_data
def cargar_datos(archivo):
    return pd.read_csv(archivo)

# Función para preprocesar datos
@st.cache_data
def preprocesar_datos(df):
    # Crear una copia para no modificar el original
    df_proc = df.copy()
    
    # Codificar variables categóricas
    le_categoria = LabelEncoder()
    le_canal = LabelEncoder()
    le_programa = LabelEncoder()
    le_region = LabelEncoder()
    
    df_proc['categoria_productos_encoded'] = le_categoria.fit_transform(df_proc['categoria_productos'])
    df_proc['canal_preferido_encoded'] = le_canal.fit_transform(df_proc['canal_preferido'])
    df_proc['programa_lealtad_encoded'] = le_programa.fit_transform(df_proc['programa_lealtad'])
    df_proc['region_encoded'] = le_region.fit_transform(df_proc['region'])
    
    return df_proc, le_categoria, le_canal, le_programa, le_region

# Función para realizar clustering
@st.cache_data
def realizar_clustering(df, columnas_seleccionadas, n_clusters):
    # Seleccionar solo las columnas numéricas para clustering
    X = df[columnas_seleccionadas]
    
    # Escalar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Aplicar K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Calcular score de silueta
    silhouette_avg = silhouette_score(X_scaled, clusters)
    
    return clusters, silhouette_avg, kmeans, scaler, X_scaled

# Sección de carga de archivo
st.header("📁 Cargar Dataset")

# Opción para usar dataset ejemplo o cargar archivo
opcion_datos = st.radio(
    "Selecciona el origen de los datos:",
    ["Usar dataset de ejemplo", "Cargar archivo CSV"]
)

df = None

if opcion_datos == "Usar dataset de ejemplo":
    # Generar dataset de ejemplo
    if st.button("Generar Dataset de Ejemplo"):
        with st.spinner("Generando dataset..."):
            # Aquí iría el código del generador anterior (simplificado para el ejemplo)
            # Por simplicidad, crearemos un dataset básico
            np.random.seed(42)
            n_samples = 500
            
            data = {
                'cliente_id': [f'C{i:04d}' for i in range(1, n_samples+1)],
                'edad': np.random.randint(18, 70, n_samples),
                'ingresos_anuales': np.random.randint(20000, 120000, n_samples),
                'antiguedad_meses': np.random.randint(1, 60, n_samples),
                'num_compras_ultimo_ano': np.random.randint(0, 30, n_samples),
                'valor_total_compras': np.random.randint(50, 15000, n_samples),
                'ticket_promedio': np.random.randint(20, 800, n_samples),
                'dias_ultima_compra': np.random.randint(1, 365, n_samples),
                'categoria_productos': np.random.choice(['Básico', 'Estándar', 'Premium'], n_samples),
                'canal_preferido': np.random.choice(['Online', 'Tienda', 'Móvil'], n_samples),
                'programa_lealtad': np.random.choice(['Si', 'No'], n_samples),
                'descuentos_utilizados': np.random.randint(0, 15, n_samples),
                'calificacion_servicio': np.random.randint(1, 5, n_samples),
                'region': np.random.choice(['Norte', 'Sur', 'Centro'], n_samples)
            }
            
            df = pd.DataFrame(data)
            # Calcular variables derivadas
            df['frecuencia_compra'] = df['num_compras_ultimo_ano'] / 12
            df['valor_por_mes'] = df['valor_total_compras'] / df['antiguedad_meses']
            df['recencia_score'] = np.where(df['dias_ultima_compra'] <= 30, 5,
                                  np.where(df['dias_ultima_compra'] <= 60, 4,
                                  np.where(df['dias_ultima_compra'] <= 90, 3,
                                  np.where(df['dias_ultima_compra'] <= 180, 2, 1))))
            
            st.success("Dataset generado exitosamente!")
            
else:
    archivo_cargado = st.file_uploader(
        "Carga tu archivo CSV",
        type=['csv'],
        help="El archivo debe contener las columnas necesarias para el análisis"
    )
    
    if archivo_cargado is not None:
        df = cargar_datos(archivo_cargado)
        st.success("Archivo cargado exitosamente!")

# Si tenemos datos, proceder con el análisis
if df is not None:
    st.header("📊 Exploración de Datos")
    
    # Mostrar información básica
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Clientes", len(df))
    with col2:
        st.metric("Variables", len(df.columns))
    with col3:
        st.metric("Valor Total Ventas", f"${df['valor_total_compras'].sum():,.0f}")
    with col4:
        st.metric("Ticket Promedio", f"${df['ticket_promedio'].mean():.2f}")
    
    # Mostrar muestra de datos
    st.subheader("Vista previa de los datos")
    st.dataframe(df.head(10))
    
    # Estadísticas descriptivas
    if st.expander("Ver estadísticas descriptivas"):
        st.write(df.describe())
    
    # Preprocesamiento
    df_procesado, le_cat, le_canal, le_prog, le_region = preprocesar_datos(df)
    
    # Configuración del clustering
    st.header("🔧 Configuración del Clustering")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Selección de variables para clustering
        columnas_numericas = [
            'edad', 'ingresos_anuales', 'antiguedad_meses', 'num_compras_ultimo_ano',
            'valor_total_compras', 'ticket_promedio', 'dias_ultima_compra',
            'frecuencia_compra', 'valor_por_mes', 'recencia_score',
            'descuentos_utilizados', 'calificacion_servicio',
            'categoria_productos_encoded', 'canal_preferido_encoded',
            'programa_lealtad_encoded', 'region_encoded'
        ]
        
        # Filtrar solo las columnas que existen en el dataframe
        columnas_disponibles = [col for col in columnas_numericas if col in df_procesado.columns]
        
        columnas_seleccionadas = st.multiselect(
            "Selecciona las variables para el clustering:",
            columnas_disponibles,
            default=['valor_total_compras', 'num_compras_ultimo_ano', 'ticket_promedio', 'dias_ultima_compra']
        )
    
    with col2:
        # Número de clusters
        n_clusters = st.slider(
            "Número de clusters:",
            min_value=2,
            max_value=10,
            value=4,
            help="Número de segmentos de clientes a crear"
        )
    
    if columnas_seleccionadas:
        # Realizar clustering
        clusters, silhouette_score_val, modelo_kmeans, scaler, X_scaled = realizar_clustering(
            df_procesado, columnas_seleccionadas, n_clusters
        )
        
        # Agregar clusters al dataframe
        df_con_clusters = df.copy()
        df_con_clusters['Cluster'] = clusters
        df_con_clusters['Cluster_Label'] = df_con_clusters['Cluster'].map({
            0: 'Segmento A', 1: 'Segmento B', 2: 'Segmento C', 
            3: 'Segmento D', 4: 'Segmento E'
        })
        
        # Mostrar resultados
        st.header("📈 Resultados del Clustering")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Score de Silueta", f"{silhouette_score_val:.3f}")
        with col2:
            st.metric("Clusters Creados", n_clusters)
        
        # Distribución de clusters
        st.subheader("Distribución de Clientes por Segmento")
        cluster_counts = df_con_clusters['Cluster_Label'].value_counts()
        
        fig_pie = px.pie(
            values=cluster_counts.values,
            names=cluster_counts.index,
            title="Distribución de Clientes por Segmento"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Análisis de clusters
        st.subheader("Análisis Detallado por Segmento")
        
        # Características promedio por cluster
        cluster_analysis = df_con_clusters.groupby('Cluster_Label').agg({
            'edad': 'mean',
            'ingresos_anuales': 'mean',
            'valor_total_compras': 'mean',
            'num_compras_ultimo_ano': 'mean',
            'ticket_promedio': 'mean',
            'dias_ultima_compra': 'mean',
            'calificacion_servicio': 'mean'
        }).round(2)
        
        st.dataframe(cluster_analysis)
        
        # Visualizaciones
        st.subheader("Visualizaciones de Segmentos")
        
        # PCA para visualización en 2D
        if len(columnas_seleccionadas) > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            fig_pca = px.scatter(
                x=X_pca[:, 0], y=X_pca[:, 1],
                color=df_con_clusters['Cluster_Label'],
                title="Visualización de Clusters (PCA)",
                labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)',
                       'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)'}
            )
            st.plotly_chart(fig_pca, use_container_width=True)
        
        # Gráfico de dispersión por variables seleccionadas
        if len(columnas_seleccionadas) >= 2:
            var1 = st.selectbox("Variable X:", columnas_seleccionadas)
            var2 = st.selectbox("Variable Y:", [col for col in columnas_seleccionadas if col != var1])
            
            fig_scatter = px.scatter(
                df_con_clusters, x=var1, y=var2,
                color='Cluster_Label',
                title=f"Distribución de Clusters: {var1} vs {var2}",
                hover_data=['cliente_id']
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Análisis de características por cluster
        st.subheader("Perfiles de Segmentos")
        
        for i in range(n_clusters):
            cluster_name = f"Segmento {chr(65+i)}"  # A, B, C, etc.
            cluster_data = df_con_clusters[df_con_clusters['Cluster'] == i]
            
            with st.expander(f"{cluster_name} ({len(cluster_data)} clientes)"):
                st.write(f"**Características principales del {cluster_name}:**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Edad promedio", f"{cluster_data['edad'].mean():.1f} años")
                    st.metric("Ingresos promedio", f"${cluster_data['ingresos_anuales'].mean():,.0f}")
                
                with col2:
                    st.metric("Compras/año", f"{cluster_data['num_compras_ultimo_ano'].mean():.1f}")
                    st.metric("Valor total", f"${cluster_data['valor_total_compras'].mean():,.0f}")
                
                with col3:
                    st.metric("Ticket promedio", f"${cluster_data['ticket_promedio'].mean():.0f}")
                    st.metric("Días última compra", f"{cluster_data['dias_ultima_compra'].mean():.0f}")
        
        # Descargar resultados
        st.header("💾 Descargar Resultados")
        
        csv_resultado = df_con_clusters.to_csv(index=False)
        st.download_button(
            label="Descargar dataset con clusters",
            data=csv_resultado,
            file_name="clientes_segmentados.csv",
            mime="text/csv"
        )
        
        # Recomendaciones estratégicas
        st.header("💡 Recomendaciones Estratégicas")
        
        st.markdown("""
        **Basado en la segmentación realizada, considera estas estrategias:**
        
        🎯 **Segmento de Alto Valor**: Programas VIP, productos premium, atención personalizada
        
        📈 **Segmento Frecuente**: Programas de lealtad, ofertas por volumen, comunicación regular
        
        🔔 **Segmento Ocasional**: Campañas de reactivación, ofertas especiales, recordatorios
        
        🆕 **Segmento Nuevo**: Onboarding, productos de entrada, seguimiento cercano
        
        ⚠️ **Segmento en Riesgo**: Campañas de retención, encuestas de satisfacción, ofertas personalizadas
        """)

else:
    st.info("Por favor, carga un dataset o genera uno de ejemplo para comenzar el análisis.")

# Footer
st.markdown("---")
st.markdown("Desarrollado con ❤️ usando Streamlit y Scikit-learn")
