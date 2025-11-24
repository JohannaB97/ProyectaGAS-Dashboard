"""
Dashboard ProyectaGAS - Predicción de Demanda de Gas Natural y Precios Internacionales
Modelo XGBoost - Datos Reales
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np

# ===========================================================================
# CONFIGURACIÓN DE PÁGINA
# ===========================================================================

st.set_page_config(
    page_title="ProyectaGAS Dashboard",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================================================================
# FUNCIÓN PARA CARGAR DATOS
# ===========================================================================

@st.cache_data
def cargar_datos():
    """
    Carga todos los CSVs necesarios
    """
    try:
        # Métricas
        metricas_agregado = pd.read_csv('data/xgboost_metricas.csv')
        metricas_desagregado = pd.read_csv('data/xgboost_metricas_desagregadas.csv')
        
        # Predicciones
        pred_modelo1 = pd.read_csv('data/predicciones_modelo1_xgboost.csv', parse_dates=['Fecha'])
        pred_modelo2 = pd.read_csv('data/predicciones_modelo2_desagregado.csv', parse_dates=['Fecha'])
        
        return metricas_agregado, metricas_desagregado, pred_modelo1, pred_modelo2
    
    except FileNotFoundError as e:
        st.error(f"""
        ❌ Error al cargar archivos: {e}
        
        **Asegúrate de tener estos archivos en la carpeta `data/`:**
        - xgboost_metricas.csv
        - xgboost_metricas_desagregadas.csv
        - predicciones_modelo1_xgboost.csv
        - predicciones_modelo2_desagregado.csv
        """)
        st.stop()

# ===========================================================================
# CARGAR DATOS
# ===========================================================================

metricas_agregado, metricas_desagregado, pred_modelo1, pred_modelo2 = cargar_datos()

# ===========================================================================
# SIDEBAR
# ===========================================================================

st.sidebar.title("⛽ ProyectaGAS")
st.sidebar.markdown("### Dashboard de Predicción")
st.sidebar.markdown("---")

st.sidebar.markdown("""
**Variables Proyectadas (13):**

**Demanda (11):**
- Total Nacional
- Costa / Interior
- 8 Sectores de Consumo

**Precios (2):**
- Henry Hub (USD/MMBtu)
- TTF (USD/MMBtu)

**Modelo:** XGBoost  
**Horizonte:** 590 días
""")

st.sidebar.markdown("---")
st.sidebar.markdown("*Universidad del Norte*")
st.sidebar.markdown("*Johanna Blanquicet*")

# ===========================================================================
# TÍTULO PRINCIPAL
# ===========================================================================

st.title("⛽ ProyectaGAS - Predicción de Demanda y Precios")
st.markdown("### Proyección con XGBoost - Análisis Desagregado")

# ===========================================================================
# TABS PRINCIPALES
# ===========================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Resumen Ejecutivo",
    "📈 Demanda Total",
    "🗺️ Costa vs Interior",
    "🏭 Análisis por Sector",
    "💰 Precios Internacionales"
])

# ===========================================================================
# TAB 1: RESUMEN EJECUTIVO
# ===========================================================================

with tab1:
    st.header("Resumen Ejecutivo")
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    # Mejor demanda sectorial
    mejor_demanda = metricas_desagregado.loc[
        metricas_desagregado['MAPE_Test'].idxmin()
    ]
    
    with col1:
        st.metric(
            "Mejor Demanda Sectorial",
            f"{mejor_demanda['Variable'].replace('Demanda_', '').replace('_Total_MBTUD', '')}",
            f"{mejor_demanda['MAPE_Test']:.2f}% MAPE"
        )
    
    # Mejor precio
    mejor_precio = metricas_agregado[metricas_agregado['Variable'].isin(['Henry Hub', 'TTF'])].loc[
        metricas_agregado[metricas_agregado['Variable'].isin(['Henry Hub', 'TTF'])]['MAPE_Test'].idxmin()
    ]
    
    with col2:
        st.metric(
            "Mejor Precio",
            f"{mejor_precio['Variable']}",
            f"{mejor_precio['MAPE_Test']:.2f}% MAPE"
        )
    
    # Demanda total agregada
    demanda_total_agg = metricas_agregado[metricas_agregado['Variable'] == 'Demanda']
    
    with col3:
        st.metric(
            "Demanda Total (Agregado)",
            f"{demanda_total_agg['MAPE_Test'].values[0]:.2f}% MAPE",
            f"R² {demanda_total_agg['R2_Test'].values[0]:.3f}"
        )
    
    # Variables con MAPE <10%
    variables_buenas = (metricas_desagregado['MAPE_Test'] < 10).sum()
    
    with col4:
        st.metric(
            "Variables <10% MAPE",
            f"{variables_buenas} de 11",
            "Desagregado"
        )
    
    st.markdown("---")
    
    # Gráfico de barras con MAPE por variable
    st.subheader("MAPE por Variable")
    
    # Combinar métricas
    df_plot = pd.concat([
        metricas_agregado[['Variable', 'MAPE_Test']].assign(Tipo='Precio'),
        metricas_desagregado[['Variable', 'MAPE_Test']].assign(Tipo='Demanda')
    ])
    
    # Limpiar nombres
    df_plot['Variable'] = df_plot['Variable'].str.replace('Demanda_', '').str.replace('_Total_MBTUD', '').str.replace('_', ' ')
    
    # Ordenar por MAPE
    df_plot = df_plot.sort_values('MAPE_Test')
    
    # Colores por tipo
    color_map = {'Precio': '#FFD700', 'Demanda': '#4169E1'}
    
    fig = px.bar(
        df_plot,
        x='Variable',
        y='MAPE_Test',
        color='Tipo',
        color_discrete_map=color_map,
        title='MAPE por Variable (XGBoost)',
        labels={'MAPE_Test': 'MAPE (%)', 'Variable': ''}
    )
    
    fig.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Tabla completa
    st.subheader("Métricas Detalladas - Demanda Desagregada")
    
    df_tabla = metricas_desagregado.copy()
    df_tabla['Variable'] = df_tabla['Variable'].str.replace('Demanda_', '').str.replace('_Total_MBTUD', '')
    
    # Clasificación
    def clasificar_mape(mape):
        if mape < 5:
            return "🟢 Excelente"
        elif mape < 10:
            return "🟡 Bueno"
        elif mape < 20:
            return "🟠 Aceptable"
        else:
            return "🔴 Desafiante"
    
    df_tabla['Clasificación'] = df_tabla['MAPE_Test'].apply(clasificar_mape)
    
    st.dataframe(
        df_tabla[['Variable', 'MAPE_Test', 'R2_Test', 'Clasificación']].style.format({
            'MAPE_Test': '{:.2f}%',
            'R2_Test': '{:.3f}'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    
    # Hallazgos clave
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Hallazgos - Demanda")
        st.markdown(f"""
        - **Mejor sector:** Residencial (3.07% MAPE, R² 0.734)
        - **Costa vs Interior:** Interior 1.8× más predecible (9.04% vs 16.32%)
        - **Sectores <10% MAPE:** 6 de 11 variables
        - **Más desafiante:** Generación Térmica (33.55%) requiere variables exógenas
        """)
    
    with col2:
        st.subheader("💰 Hallazgos - Precios")
        st.markdown(f"""
        - **Mejor precio:** TTF (6.67% MAPE, R² 0.555)
        - **Henry Hub:** 8.20% MAPE, R² 0.570
        - **Modelo agregado:** Demanda Total 4.77% MAPE con features de precios
        - **Ventaja integración:** Precios como features mejoran proyección demanda
        """)

# ===========================================================================
# TAB 2: DEMANDA TOTAL
# ===========================================================================

with tab2:
    st.header("Demanda Total Colombia")
    
    st.info("""
    **Nota:** Este tab muestra el **Modelo Agregado** que incluye precios (Henry Hub, TTF) 
    como features exógenas. Ver Tab "Análisis por Sector" para el modelo desagregado.
    """)
    
    # Métricas
    demanda_metrics = metricas_agregado[metricas_agregado['Variable'] == 'Demanda'].iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("MAPE", f"{demanda_metrics['MAPE_Test']:.2f}%")
    
    with col2:
        st.metric("R²", f"{demanda_metrics['R2_Test']:.3f}")
    
    with col3:
        media = pred_modelo1['Demanda_Total_real'].mean()
        st.metric("Media", f"{media:,.0f} MBTUD")
    
    with col4:
        st.metric("Días Proyectados", len(pred_modelo1))
    
    st.markdown("---")
    
    # Gráfico principal
    st.subheader("Predicciones: Real vs XGBoost")
    
    # Submuestreo para visualización (cada 3 días)
    df_plot = pred_modelo1.iloc[::3].copy()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_plot['Fecha'],
        y=df_plot['Demanda_Total_real'],
        name='Real',
        line=dict(color='blue', width=2),
        mode='lines'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_plot['Fecha'],
        y=df_plot['Demanda_Total_pred'],
        name='XGBoost',
        line=dict(color='green', width=2, dash='dash'),
        mode='lines'
    ))
    
    fig.update_layout(
        title='Demanda Total MBTUD - Test Set',
        xaxis_title='Fecha',
        yaxis_title='MBTUD',
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Análisis de errores
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribución de Errores")
        
        errores = ((pred_modelo1['Demanda_Total_pred'] - pred_modelo1['Demanda_Total_real']) / 
                   pred_modelo1['Demanda_Total_real']) * 100
        
        fig = go.Figure(data=[go.Histogram(x=errores, nbinsx=50)])
        fig.update_layout(
            title='Distribución de Errores Porcentuales',
            xaxis_title='Error (%)',
            yaxis_title='Frecuencia',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Estadísticas de Error")
        
        st.metric("Error Medio", f"{errores.mean():.2f}%")
        st.metric("Desv. Estándar", f"{errores.std():.2f}%")
        st.metric("Error Máximo", f"{errores.abs().max():.2f}%")
        
        dentro_10 = (errores.abs() < 10).sum() / len(errores) * 100
        st.metric("% dentro de ±10%", f"{dentro_10:.1f}%")
    
    st.markdown("---")
    
    st.subheader("💡 Recomendaciones")
    st.markdown("""
    - **Planificación de capacidad:** Usar este modelo para proyecciones agregadas nacionales
    - **Contratos de suministro:** Complementar con proyecciones sectoriales específicas
    - **Gestión de riesgo:** Considerar bandas de confianza ±10% para contingencias
    - **Mejora del modelo:** Integrar variables climáticas e hidrológicas para mayor precisión
    """)

# ===========================================================================
# TAB 3: COSTA VS INTERIOR
# ===========================================================================

with tab3:
    st.header("Demanda por Zona Geográfica")
    
    # Métricas comparativas
    costa_metrics = metricas_desagregado[metricas_desagregado['Variable'] == 'Demanda_Costa_Total_MBTUD'].iloc[0]
    interior_metrics = metricas_desagregado[metricas_desagregado['Variable'] == 'Demanda_Interior_Total_MBTUD'].iloc[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌊 Costa Atlántica")
        st.metric("MAPE", f"{costa_metrics['MAPE_Test']:.2f}%")
        st.metric("R²", f"{costa_metrics['R2_Test']:.3f}")
        st.metric("Participación", "51.2%")
    
    with col2:
        st.subheader("🏔️ Interior")
        st.metric("MAPE", f"{interior_metrics['MAPE_Test']:.2f}%")
        st.metric("R²", f"{interior_metrics['R2_Test']:.3f}")
        st.metric("Participación", "48.8%")
    
    st.markdown("---")
    
    # Gráficos comparativos
    col1, col2 = st.columns(2)
    
    df_plot = pred_modelo2.iloc[::5].copy()
    
    with col1:
        st.subheader("Costa Atlántica")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_plot['Fecha'],
            y=df_plot['Demanda_Costa_Total_MBTUD_real'],
            name='Real',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df_plot['Fecha'],
            y=df_plot['Demanda_Costa_Total_MBTUD_pred'],
            name='XGBoost',
            line=dict(color='green', width=2, dash='dash')
        ))
        fig.update_layout(height=400, yaxis_title='MBTUD')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Interior")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_plot['Fecha'],
            y=df_plot['Demanda_Interior_Total_MBTUD_real'],
            name='Real',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df_plot['Fecha'],
            y=df_plot['Demanda_Interior_Total_MBTUD_pred'],
            name='XGBoost',
            line=dict(color='green', width=2, dash='dash')
        ))
        fig.update_layout(height=400, yaxis_title='MBTUD')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("🔍 Análisis Diferencial")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Costa Atlántica (MAPE 16.32%)**
        
        **Características:**
        - Mayor heterogeneidad sectorial
        - Mix industrial complejo (petroquímica, zona franca)
        - Refinería de Cartagena con alta variabilidad
        - Zonas residenciales dispersas
        
        **Por qué es menos predecible:**
        - Demanda industrial sujeta a ciclos económicos
        - Paradas de mantenimiento no programadas
        - Operación de refinería con patrones irregulares
        """)
    
    with col2:
        st.markdown("""
        **Interior (MAPE 9.04%)**
        
        **Características:**
        - Patrones más homogéneos
        - Dominado por Residencial y Generación Térmica
        - Estacionalidades predecibles
        - Menor concentración industrial
        
        **Por qué es más predecible:**
        - Consumo residencial muy regular
        - Estacionalidad climática clara
        - Generación térmica complementa hidráulica
        - Menor exposición a ciclos industriales
        """)
    
    st.markdown("---")
    
    st.subheader("💡 Implicaciones Operacionales")
    st.markdown("""
    - **Costa:** Requiere mayor flexibilidad en contratos, inventarios de seguridad más altos
    - **Interior:** Posible contratos de largo plazo con menor riesgo, gestión basada en estacionalidad
    - **Infraestructura:** Priorizar expansión de capacidad de almacenamiento en Costa
    - **Comercial:** Segmentar estrategias de pricing por zona geográfica
    """)

# ===========================================================================
# TAB 4: ANÁLISIS POR SECTOR
# ===========================================================================

with tab4:
    st.header("Análisis por Sector de Consumo")
    
    # Selector de sector
    sectores_disponibles = [
        'Residencial', 'Petrolero', 'GNVC', 'Refineria', 
        'Industrial', 'Comercial', 'GeneracionTermica', 'Compresora'
    ]
    
    sector_sel = st.selectbox(
        "Selecciona un sector:",
        sectores_disponibles,
        format_func=lambda x: x.replace('GeneracionTermica', 'Generación Térmica')
    )
    
    # Obtener métricas del sector
    var_name = f'Demanda_{sector_sel}_Total_MBTUD'
    sector_metrics = metricas_desagregado[metricas_desagregado['Variable'] == var_name].iloc[0]
    
    # Ranking del sector
    ranking = metricas_desagregado.sort_values('MAPE_Test').reset_index(drop=True)
    pos = ranking[ranking['Variable'] == var_name].index[0] + 1
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("MAPE", f"{sector_metrics['MAPE_Test']:.2f}%")
    
    with col2:
        st.metric("R²", f"{sector_metrics['R2_Test']:.3f}")
    
    with col3:
        st.metric("Ranking", f"{pos} de 8")
    
    with col4:
        clasif = ""
        if sector_metrics['MAPE_Test'] < 5:
            clasif = "🟢 Excelente"
        elif sector_metrics['MAPE_Test'] < 10:
            clasif = "🟡 Bueno"
        elif sector_metrics['MAPE_Test'] < 20:
            clasif = "🟠 Aceptable"
        else:
            clasif = "🔴 Desafiante"
        st.metric("Clasificación", clasif)
    
    st.markdown("---")
    
    # Estadísticas de consumo
    st.subheader("📊 Estadísticas de Consumo (MBTUD)")
    
    real_col = f'{var_name}_real'
    pred_col = f'{var_name}_pred'
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Promedio", f"{pred_modelo2[real_col].mean():,.0f}")
    
    with col2:
        st.metric("Mediana", f"{pred_modelo2[real_col].median():,.0f}")
    
    with col3:
        st.metric("Máximo", f"{pred_modelo2[real_col].max():,.0f}")
    
    with col4:
        st.metric("Mínimo", f"{pred_modelo2[real_col].min():,.0f}")
    
    st.markdown("---")
    
    # Gráfico de predicciones
    st.subheader("Predicciones: Real vs XGBoost")
    
    df_plot = pred_modelo2.iloc[::4].copy()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_plot['Fecha'],
        y=df_plot[real_col],
        name='Real',
        line=dict(color='blue', width=2),
        mode='lines'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_plot['Fecha'],
        y=df_plot[pred_col],
        name='XGBoost',
        line=dict(color='green', width=2, dash='dash'),
        mode='lines'
    ))
    
    fig.update_layout(
        title=f'{sector_sel} - MAPE: {sector_metrics["MAPE_Test"]:.2f}% | R²: {sector_metrics["R2_Test"]:.3f}',
        xaxis_title='Fecha',
        yaxis_title='MBTUD',
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Box plot de distribución
    st.subheader("Distribución y Volatilidad")
    
    fig = go.Figure()
    fig.add_trace(go.Box(y=pred_modelo2[real_col], name='Real'))
    fig.add_trace(go.Box(y=pred_modelo2[pred_col], name='Predicción'))
    fig.update_layout(height=400, yaxis_title='MBTUD')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Información detallada por sector
    st.subheader(f"📋 Análisis Específico: {sector_sel.replace('GeneracionTermica', 'Generación Térmica')}")
    
    analisis = {
        'Residencial': {
            'consumo': '165,000-178,000 MBTUD',
            'participacion': '16.7%',
            'caracteristicas': [
                'Estacionalidad mensual fuerte (22% amplitud)',
                'Patrones horarios muy regulares',
                'Mayor consumo en meses fríos',
                'Correlación con temperatura'
            ],
            'recomendaciones': [
                'Contratos estacionales con tarifas diferenciadas',
                'Gestión de demanda en picos invernales',
                'Programas de eficiencia energética',
                'Proyección más confiable de todos los sectores'
            ]
        },
        'Petrolero': {
            'consumo': '17,000-20,000 MBTUD',
            'participacion': '1.8%',
            'caracteristicas': [
                'Consumo muy estable',
                'Baja volatilidad',
                'Poco sensible a estacionalidad',
                'Operación continua de campos'
            ],
            'recomendaciones': [
                'Contratos de largo plazo fijos',
                'Bajo riesgo operacional',
                'Inventarios mínimos de seguridad',
                'Monitoreo de producción petrolera'
            ]
        },
        'GNVC': {
            'consumo': '58,000-67,000 MBTUD',
            'participacion': '6.1%',
            'caracteristicas': [
                'Crecimiento sostenido (+8% anual)',
                'Expansión de red de transporte',
                'Sustitución de combustibles',
                'Urbano principalmente'
            ],
            'recomendaciones': [
                'Proyectar crecimiento en contratos',
                'Expansión coordinada infraestructura',
                'Incentivos para conversión vehicular',
                'Monitoreo de tendencias de movilidad'
            ]
        },
        'Refineria': {
            'consumo': '100,000-115,000 MBTUD',
            'participacion': '10.5%',
            'caracteristicas': [
                'Alta volatilidad por paradas',
                'Mantenimientos programados',
                'Refinería Cartagena dominante',
                'Correlación con producción refinados'
            ],
            'recomendaciones': [
                'Coordinación estrecha mantenimientos',
                'Contratos con cláusulas de flexibilidad',
                'Inventarios de seguridad ampliados',
                'Integrar calendario de paradas'
            ]
        },
        'Industrial': {
            'consumo': '115,000-130,000 MBTUD',
            'participacion': '12.0%',
            'caracteristicas': [
                'Correlación con PMI manufacturero',
                'Sensible a ciclos económicos',
                'Mix heterogéneo de industrias',
                'Mayor demanda en recuperación'
            ],
            'recomendaciones': [
                'Segmentar por subsector industrial',
                'Contratos vinculados a indicadores económicos',
                'Flexibilidad en volúmenes',
                'Integrar variables macroeconómicas'
            ]
        },
        'Comercial': {
            'consumo': '54,000-67,000 MBTUD',
            'participacion': '5.9%',
            'caracteristicas': [
                'Pico diciembre (+35%)',
                'Estacionalidad comercial',
                'Sensible a días festivos',
                'Horarios laborales marcados'
            ],
            'recomendaciones': [
                'Contratos trimestrales',
                'Provisión picos fin de año',
                'Gestión de demanda en temporadas altas',
                'Tarifas incentivadas fuera de pico'
            ]
        },
        'GeneracionTermica': {
            'consumo': '200,000-380,000 MBTUD',
            'participacion': '28.5%',
            'caracteristicas': [
                'Inversamente correlacionado con hidrología',
                'Picos extremos en El Niño',
                'Mayor volatilidad de todos',
                'Complementa generación hidráulica'
            ],
            'recomendaciones': [
                'CRÍTICO: Integrar pronósticos hidrológicos',
                'Monitoreo índice ENSO',
                'Contratos de respaldo flexible',
                'No proyectar independiente - usar hidrología'
            ]
        },
        'Compresora': {
            'consumo': '25,000-70,000 MBTUD',
            'participacion': '4.8%',
            'caracteristicas': [
                'Alta volatilidad (53% MAPE)',
                'Depende de flujos de transporte',
                'Consumo de estaciones compresoras',
                'No independiente de demanda total'
            ],
            'recomendaciones': [
                'NO proyectar de forma independiente',
                'Modelar como f(Demanda Total)',
                'Derivar de flujos de gasoductos',
                'Coordinar con infraestructura de transporte'
            ]
        }
    }
    
    if sector_sel in analisis:
        info = analisis[sector_sel]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Consumo Típico:**")
            st.info(info['consumo'])
            st.markdown("**Participación:**")
            st.info(info['participacion'])
            
            st.markdown("**Características:**")
            for car in info['caracteristicas']:
                st.markdown(f"- {car}")
        
        with col2:
            st.markdown("**Recomendaciones Operacionales:**")
            for rec in info['recomendaciones']:
                st.markdown(f"- {rec}")

# ===========================================================================
# TAB 5: PRECIOS INTERNACIONALES
# ===========================================================================

with tab5:
    st.header("Precios Internacionales de Gas Natural")
    
    st.markdown("""
    Proyección de precios de referencia internacional para importaciones de GNL, 
    exportaciones, benchmarking y decisiones de inversión.
    """)
    
    # Selector de precio
    precio_sel = st.radio(
        "Selecciona mercado:",
        ["Henry Hub (EE.UU.)", "TTF (Europa)"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if precio_sel == "Henry Hub (EE.UU.)":
        # Métricas Henry Hub
        hh_metrics = metricas_agregado[metricas_agregado['Variable'] == 'Henry Hub'].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("MAPE", f"{hh_metrics['MAPE_Test']:.2f}%")
        
        with col2:
            st.metric("R²", f"{hh_metrics['R2_Test']:.3f}")
        
        with col3:
            st.metric("Mercado", "Estados Unidos")
        
        with col4:
            precio_prom = pred_modelo1['Henry_Hub_real'].mean()
            st.metric("Precio Promedio", f"${precio_prom:.2f}/MMBtu")
        
        st.markdown("---")
        
        # Descripción
        st.subheader("📍 Henry Hub Natural Gas Spot Price")
        st.markdown("""
        **Mercado:** Louisiana, Estados Unidos  
        **Referencia:** NYMEX natural gas futures  
        **Fuente:** Federal Reserve Economic Data (FRED)
        """)
        
        st.markdown("---")
        
        # Estadísticas
        st.subheader("📊 Estadísticas de Precio (USD/MMBtu)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Promedio", f"${pred_modelo1['Henry_Hub_real'].mean():.2f}")
        
        with col2:
            st.metric("Mediana", f"${pred_modelo1['Henry_Hub_real'].median():.2f}")
        
        with col3:
            st.metric("Máximo", f"${pred_modelo1['Henry_Hub_real'].max():.2f}")
        
        with col4:
            st.metric("Mínimo", f"${pred_modelo1['Henry_Hub_real'].min():.2f}")
        
        st.markdown("---")
        
        # Gráfico principal
        st.subheader("Predicciones: Real vs XGBoost")
        
        df_plot = pred_modelo1.iloc[::4].copy()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_plot['Fecha'],
            y=df_plot['Henry_Hub_real'],
            name='Real',
            line=dict(color='blue', width=2),
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            x=df_plot['Fecha'],
            y=df_plot['Henry_Hub_pred'],
            name='XGBoost',
            line=dict(color='green', width=2, dash='dash'),
            mode='lines'
        ))
        
        fig.update_layout(
            title=f'Henry Hub (USD/MMBtu) - MAPE: {hh_metrics["MAPE_Test"]:.2f}%',
            xaxis_title='Fecha',
            yaxis_title='USD/MMBtu',
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Box plot
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribución de Precios")
            
            fig = go.Figure()
            fig.add_trace(go.Box(y=pred_modelo1['Henry_Hub_real'], name='Real'))
            fig.add_trace(go.Box(y=pred_modelo1['Henry_Hub_pred'], name='Predicción'))
            fig.update_layout(height=400, yaxis_title='USD/MMBtu')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Estadísticas Adicionales")
            
            desv = pred_modelo1['Henry_Hub_real'].std()
            cv = (desv / pred_modelo1['Henry_Hub_real'].mean()) * 100
            
            st.metric("Desv. Estándar", f"${desv:.2f}")
            st.metric("Coef. Variación", f"{cv:.1f}%")
            st.metric("Percentil 95", f"${pred_modelo1['Henry_Hub_real'].quantile(0.95):.2f}")
            st.metric("Percentil 5", f"${pred_modelo1['Henry_Hub_real'].quantile(0.05):.2f}")
        
        st.markdown("---")
        
        # Características del mercado
        st.subheader("🔍 Características del Mercado Henry Hub")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Drivers de Precio:**
            1. Producción shale gas (estable, abundante)
            2. Demanda estacional (calefacción invierno, cooling verano)
            3. Almacenamiento subterráneo (inventarios semanales)
            4. Exportaciones GNL (Asia, Europa)
            5. Clima extremo (olas de frío/calor)
            
            **Rango Típico:** $2-4 USD/MMBtu  
            **Picos:** $6-8 en inviernos extremos
            """)
        
        with col2:
            st.markdown("""
            **Por qué es predecible (8.20% MAPE):**
            - Mercado maduro y líquido
            - Producción doméstica abundante
            - Estacionalidad marcada y repetible
            - Infrastructure de almacenamiento robusta
            - Datos de alta calidad (EIA, FRED)
            
            **Factores de Riesgo:**
            - Eventos climáticos extremos
            - Cambios en política energética
            - Decisiones OPEC+ (precio petróleo)
            - Demanda asiática de GNL
            """)
        
        st.markdown("---")
        
        st.subheader("💡 Aplicaciones para Colombia")
        st.markdown("""
        - **Importación GNL:** Referencia para contratos de importación (spread vs HH)
        - **Contratos de suministro:** Indexación a Henry Hub + flete
        - **Hedge financiero:** Coberturas en NYMEX futures
        - **Análisis competitividad:** Comparación precios domésticos vs internacional
        - **Decisiones de inversión:** Evaluación proyectos de regasificación
        """)
    
    else:  # TTF
        # Métricas TTF
        ttf_metrics = metricas_agregado[metricas_agregado['Variable'] == 'TTF'].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("MAPE", f"{ttf_metrics['MAPE_Test']:.2f}%")
        
        with col2:
            st.metric("R²", f"{ttf_metrics['R2_Test']:.3f}")
        
        with col3:
            st.metric("Mercado", "Europa")
        
        with col4:
            precio_prom = pred_modelo1['TTF_real'].mean()
            st.metric("Precio Promedio", f"${precio_prom:.2f}/MMBtu")
        
        st.markdown("---")
        
        # Descripción
        st.subheader("📍 Dutch TTF Natural Gas Futures")
        st.markdown("""
        **Mercado:** Title Transfer Facility, Holanda  
        **Referencia:** Precio de referencia para Europa  
        **Fuente:** Investing.com
        """)
        
        st.markdown("---")
        
        # Estadísticas
        st.subheader("📊 Estadísticas de Precio (USD/MMBtu)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Promedio", f"${pred_modelo1['TTF_real'].mean():.2f}")
        
        with col2:
            st.metric("Mediana", f"${pred_modelo1['TTF_real'].median():.2f}")
        
        with col3:
            st.metric("Máximo", f"${pred_modelo1['TTF_real'].max():.2f}")
        
        with col4:
            st.metric("Mínimo", f"${pred_modelo1['TTF_real'].min():.2f}")
        
        st.markdown("---")
        
        # Gráfico principal
        st.subheader("Predicciones: Real vs XGBoost")
        
        df_plot = pred_modelo1.iloc[::4].copy()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_plot['Fecha'],
            y=df_plot['TTF_real'],
            name='Real',
            line=dict(color='blue', width=2),
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            x=df_plot['Fecha'],
            y=df_plot['TTF_pred'],
            name='XGBoost',
            line=dict(color='green', width=2, dash='dash'),
            mode='lines'
        ))
        
        fig.update_layout(
            title=f'TTF (USD/MMBtu) - MAPE: {ttf_metrics["MAPE_Test"]:.2f}%',
            xaxis_title='Fecha',
            yaxis_title='USD/MMBtu',
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Box plot
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribución de Precios")
            
            fig = go.Figure()
            fig.add_trace(go.Box(y=pred_modelo1['TTF_real'], name='Real'))
            fig.add_trace(go.Box(y=pred_modelo1['TTF_pred'], name='Predicción'))
            fig.update_layout(height=400, yaxis_title='USD/MMBtu')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Estadísticas Adicionales")
            
            desv = pred_modelo1['TTF_real'].std()
            cv = (desv / pred_modelo1['TTF_real'].mean()) * 100
            
            st.metric("Desv. Estándar", f"${desv:.2f}")
            st.metric("Coef. Variación", f"{cv:.1f}%")
            st.metric("Percentil 95", f"${pred_modelo1['TTF_real'].quantile(0.95):.2f}")
            st.metric("Percentil 5", f"${pred_modelo1['TTF_real'].quantile(0.05):.2f}")
        
        st.markdown("---")
        
        # Características del mercado
        st.subheader("🔍 Características del Mercado TTF")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Drivers de Precio:**
            1. Suministro ruso (reducido post-2022)
            2. Importaciones GNL (competencia con Asia)
            3. Almacenamiento europeo (niveles de inventario)
            4. Producción renovable (eólica, solar)
            5. Demanda industrial (economía europea)
            
            **Rango Típico:** $8-15 USD/MMBtu  
            **Crisis 2022:** Picos de $40-70 USD/MMBtu
            """)
        
        with col2:
            st.markdown("""
            **Por qué es MÁS predecible que HH (6.67% MAPE):**
            - Bandas de volatilidad muy informativas
            - Rolling min/max capturan rango reciente
            - Estructura de mercado post-crisis estabilizada
            - Features de rolling stats dominan (>70%)
            
            **Factores de Riesgo:**
            - Geopolítica (Rusia-Ucrania)
            - Niveles de almacenamiento
            - Clima invernal extremo
            - Competencia GNL con Asia
            - Política energética UE
            """)
        
        st.markdown("---")
        
        st.subheader("💡 Aplicaciones para Colombia")
        st.markdown("""
        - **Competencia GNL:** Europa compite por mismos cargamentos que Colombia
        - **Arbitraje internacional:** Decisiones de exportación vs consumo doméstico
        - **Diversificación portafolio:** TTF como referencia alternativa a Henry Hub
        - **Análisis de mercado:** Entender dinámicas globales de GNL
        - **Gestión de riesgo:** Correlación con otros mercados energéticos
        """)
    
    st.markdown("---")
    
    # Comparación HH vs TTF
    st.subheader("🌍 Comparación Henry Hub vs TTF")
    
    # Gráfico comparativo
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=pred_modelo1['Fecha'],
        y=pred_modelo1['Henry_Hub_real'],
        name='Henry Hub',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=pred_modelo1['Fecha'],
        y=pred_modelo1['TTF_real'],
        name='TTF',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title='Comparación de Precios (USD/MMBtu)',
        xaxis_title='Fecha',
        yaxis_title='USD/MMBtu',
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla comparativa
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Métricas de Proyección**")
        
        comp_data = {
            'Métrica': ['MAPE', 'R²', 'MAE', 'RMSE'],
            'Henry Hub': [
                f"{hh_metrics['MAPE_Test']:.2f}%",
                f"{hh_metrics['R2_Test']:.3f}",
                f"${hh_metrics['MAE_Test']:.3f}",
                f"${hh_metrics['RMSE_Test']:.3f}"
            ],
            'TTF': [
                f"{ttf_metrics['MAPE_Test']:.2f}%",
                f"{ttf_metrics['R2_Test']:.3f}",
                f"${ttf_metrics['MAE_Test']:.3f}",
                f"${ttf_metrics['RMSE_Test']:.3f}"
            ]
        }
        
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("**Hallazgos Clave**")
        st.markdown("""
        - **TTF tiene mejor MAPE** (6.67% vs 8.20%)
        - **Ambos con R² similar** (~0.55-0.57)
        - **TTF más volátil** pero bandas más predictivas
        - **Henry Hub más estable** en valor absoluto
        - **Spread HH-TTF** crucial para arbitraje GNL
        """)

# ===========================================================================
# FOOTER
# ===========================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><b>ProyectaGAS Dashboard</b> | Universidad del Norte | Johanna Blanquicet</p>
    <p>13 Modelos XGBoost | Mejor Demanda: Residencial (3.07%) | Mejor Precio: TTF (6.67%)</p>
</div>
""", unsafe_allow_html=True)
