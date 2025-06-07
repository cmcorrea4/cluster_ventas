# 🤖 Clasificación de Clientes con Algoritmos de IA

Una aplicación web desarrollada en Streamlit que permite segmentar clientes usando múltiples algoritmos de Machine Learning para estrategias de marketing y ventas personalizadas.

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Algoritmos Disponibles](#-algoritmos-disponibles)
- [Cómo Funciona K-Means](#-cómo-funciona-k-means)
- [Interpretación de Resultados](#-interpretación-de-resultados)
- [Estructura del Dataset](#-estructura-del-dataset)
- [Ejemplos de Uso](#-ejemplos-de-uso)

## 🚀 Características

- **Carga de datos CSV** con validación automática
- **4 algoritmos de IA** para clustering/clasificación
- **Preprocesamiento automático** de variables categóricas
- **Métricas de evaluación** para medir la calidad del clustering
- **Visualización interactiva** de resultados
- **Exportación de datos** clasificados
- **Interfaz simple e intuitiva**

## 🛠 Instalación

### Requisitos Previos
- Python 3.7 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clona o descarga el proyecto**
```bash
git clone [URL-del-repositorio]
cd clasificacion-clientes-ia
```

2. **Instala las dependencias**
```bash
pip install -r requirements.txt
```

3. **Ejecuta la aplicación**
```bash
streamlit run app.py
```

4. **Abre tu navegador** en la URL mostrada (generalmente `http://localhost:8501`)

## 📊 Uso

### 1. Carga de Datos
- Sube un archivo CSV con datos de clientes
- La aplicación mostrará una vista previa automáticamente
- Las variables categóricas se codificarán automáticamente

### 2. Configuración
- **Selecciona variables**: Elige las características numéricas para el análisis
- **Elige algoritmo**: Selecciona entre K-Means, DBSCAN, Gaussian Mixture o Agglomerative
- **Ajusta parámetros**: Configura los parámetros específicos del algoritmo

### 3. Ejecución
- Haz clic en "🚀 Ejecutar Clasificación"
- Revisa los resultados y métricas
- Descarga el dataset clasificado

## 🧠 Algoritmos Disponibles

### 1. **K-Means** 
- **Tipo**: Clustering por centroides
- **Mejor para**: Grupos de tamaño similar y forma esférica
- **Parámetro**: Número de clusters (2-10)

### 2. **DBSCAN**
- **Tipo**: Clustering basado en densidad
- **Mejor para**: Grupos de forma irregular y detección de outliers
- **Parámetros**: Epsilon (radio) y mínimo de muestras

### 3. **Gaussian Mixture**
- **Tipo**: Modelo probabilístico
- **Mejor para**: Grupos con distribución gaussiana
- **Parámetro**: Número de componentes (2-10)

### 4. **Agglomerative Clustering**
- **Tipo**: Clustering jerárquico
- **Mejor para**: Datos con estructura jerárquica
- **Parámetro**: Número de clusters (2-10)

## 🎯 Cómo Funciona K-Means

### Proceso Step-by-Step

1. **Inicialización**
   - El algoritmo coloca aleatoriamente K puntos centrales (centroides) en el espacio de datos
   - K = número de clusters que quieres crear

2. **Asignación**
   - Cada cliente se asigna al centroide más cercano
   - Se calculan las distancias entre cada cliente y todos los centroides

3. **Actualización**
   - Se recalcula la posición de cada centroide como el promedio de todos los clientes asignados a él
   - Los centroides se "mueven" hacia el centro de sus grupos

4. **Iteración**
   - Se repiten los pasos 2 y 3 hasta que los centroides no se muevan significativamente
   - Generalmente converge en pocas iteraciones

### Ejemplo Visual

```
Iteración 1:    Iteración 2:    Resultado Final:
   x  x           x  x              x  x
 ⭐ x  x         x ⭐ x            x ⭐ x
   x  x           x  x              x  x

 ⭐ = Centroide    x = Cliente      Grupo formado
```

### ¿Por qué funciona?
- **Minimiza la distancia**: Busca que los clientes estén lo más cerca posible de su centroide
- **Maximiza la separación**: Los grupos resultantes están bien diferenciados
- **Encuentra patrones**: Identifica automáticamente características comunes entre clientes

## 📈 Interpretación de Resultados

### Métricas de Evaluación

#### **Silhouette Score (-1 a 1)**
- **0.7 - 1.0**: Excelente separación entre clusters
- **0.5 - 0.7**: Buena estructura de clusters
- **0.2 - 0.5**: Estructura débil, clusters solapados
- **< 0.2**: No hay estructura clara de clusters

#### **Calinski-Harabasz Score (mayor es mejor)**
- **> 100**: Clusters bien definidos
- **50-100**: Separación moderada
- **< 50**: Clusters poco definidos

### Análisis por Segmento

La tabla de análisis muestra el **perfil promedio** de cada segmento:

| Métrica | Significado | Interpretación |
|---------|-------------|----------------|
| **Edad promedio** | Edad típica del segmento | Ayuda a definir estrategias por generación |
| **Ingresos promedio** | Poder adquisitivo | Determina el pricing y productos |
| **Valor total compras** | Gasto histórico | Identifica clientes de alto valor |
| **Compras/año** | Frecuencia de compra | Mide la lealtad y engagement |
| **Ticket promedio** | Gasto por transacción | Indica comportamiento de compra |

### Ejemplos de Interpretación

#### Segmento 1: "Clientes VIP"
```
Edad: 45 años | Ingresos: $85,000 | Valor total: $8,500 | Compras/año: 12
```
**Interpretación**: Clientes maduros, alta frecuencia de compra, alto valor
**Estrategia**: Programas VIP, productos premium, atención personalizada

#### Segmento 2: "Clientes Ocasionales"
```
Edad: 28 años | Ingresos: $35,000 | Valor total: $1,200 | Compras/año: 3
```
**Interpretación**: Clientes jóvenes, compras esporádicas, precio-sensibles
**Estrategia**: Ofertas especiales, programas de descuentos, marketing digital

#### Segmento 3: "Clientes Inactivos"
```
Edad: 52 años | Ingresos: $60,000 | Valor total: $2,800 | Compras/año: 1
```
**Interpretación**: Ex-clientes frecuentes, posible insatisfacción o cambio de hábitos
**Estrategia**: Campañas de reactivación, encuestas de satisfacción, ofertas de retorno

## 📂 Estructura del Dataset

### Columnas Requeridas (mínimo)
- **cliente_id**: Identificador único del cliente
- **Variables numéricas**: edad, ingresos_anuales, valor_total_compras, etc.
- **Variables categóricas** (opcionales): categoria_productos, canal_preferido, region

### Ejemplo de Dataset
```csv
cliente_id,edad,ingresos_anuales,num_compras_ultimo_ano,valor_total_compras,ticket_promedio
C0001,45,85000,12,8500,708.33
C0002,28,35000,3,1200,400.00
C0003,52,60000,1,2800,2800.00
```

## 💡 Ejemplos de Uso

### Caso 1: E-commerce
**Variables**: valor_total_compras, frecuencia_compra, ticket_promedio
**Objetivo**: Identificar compradores frecuentes vs. ocasionales
**Resultado**: 3 segmentos (VIP, Regulares, Ocasionales)

### Caso 2: Servicios Financieros
**Variables**: ingresos_anuales, antiguedad_meses, productos_contratados
**Objetivo**: Segmentar para productos financieros
**Resultado**: 4 segmentos por nivel de ingresos y madurez

### Caso 3: Retail Físico
**Variables**: edad, region, categoria_preferida, calificacion_servicio
**Objetivo**: Personalizar experiencia en tienda
**Resultado**: Segmentos geográficos y demográficos

## 🔍 Consejos para Mejores Resultados

### Selección de Variables
- **Incluye variables relevantes** para tu objetivo de negocio
- **Evita variables redundantes** (muy correlacionadas)
- **Usa 3-6 variables** para resultados óptimos
- **Combina variables** de comportamiento y demográficas

### Interpretación de Clusters
- **Analiza el contexto de negocio** de cada segmento
- **Valida los resultados** con conocimiento del dominio
- **Considera la accionabilidad** de cada segmento
- **Monitorea la estabilidad** de los clusters en el tiempo

### Troubleshooting
- **Silhouette Score muy bajo**: Prueba diferentes números de clusters o variables
- **Clusters muy desbalanceados**: Considera DBSCAN en lugar de K-Means
- **Resultados no interpretables**: Revisa la calidad y relevancia de los datos

## 📞 Soporte

¿Tienes preguntas o problemas?
- Revisa que tu CSV tenga el formato correcto
- Verifica que las variables seleccionadas sean numéricas
- Asegúrate de tener suficientes datos (mínimo 50 registros)

---

**Desarrollado con ❤️ usando Streamlit y Scikit-learn**
