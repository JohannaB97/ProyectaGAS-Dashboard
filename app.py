import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN PÁGINA
# ============================================================================

st.set_page_config(
    page_title="ProyectaGAS - Dashboard",
    page_icon="⛽",
    layout="wide"
)

# ============================================================================
# DATOS Y MÉTRICAS REALES
# ============================================================================

# MÉTRICAS REALES - Demanda Desagregada XGBoost
metricas_demanda = {
    'Demanda_Total_MBTUD': {'MAPE': 10.52, 'R2': 0.044, 'MAE': 107604.56, 'RMSE': 143855.11},
    'Demanda_Costa_Total_MBTUD': {'MAPE': 16.32, 'R2': -0.301, 'MAE': 80603.38, 'RMSE': 129235.60},
    'Demanda_Interior_Total_MBTUD': {'MAPE': 9.04, 'R2': -0.290, 'MAE': 48683.66, 'RMSE': 55914.13},
    'Demanda_Industrial_Total_MBTUD': {'MAPE': 12.58, 'R2': -1.596, 'MAE': 29131.34, 'RMSE': 32796.52},
    'Demanda_Refineria_Total_MBTUD': {'MAPE': 10.52, 'R2': -0.752, 'MAE': 14329.07, 'RMSE': 17580.39},
    'Demanda_Petrolero_Total_MBTUD': {'MAPE': 8.96, 'R2': -0.384, 'MAE': 2043.30, 'RMSE': 2628.52},
    'Demanda_GeneracionTermica_Total_MBTUD': {'MAPE': 33.55, 'R2': -0.045, 'MAE': 95296.97, 'RMSE': 135507.30},
    'Demanda_Residencial_Total_MBTUD': {'MAPE': 3.07, 'R2': 0.734, 'MAE': 5107.04, 'RMSE': 7467.17},
    'Demanda_Comercial_Total_MBTUD': {'MAPE': 14.27, 'R2': -0.808, 'MAE': 8414.08, 'RMSE': 10449.11},
    'Demanda_GNVC_Total_MBTUD': {'MAPE': 9.24, 'R2': 0.139, 'MAE': 5597.99, 'RMSE': 6203.73},
    'Demanda_Compresora_Total_MBTUD': {'MAPE': 53.23, 'R2': -0.754, 'MAE': 2539.98, 'RMSE': 3044.25}
}

# MÉTRICAS REALES - Precios (de sesión anterior)
metricas_precios = {
    'HenryHub': {'MAPE': 8.20, 'R2': 0.570, 'MAE': 0.67, 'RMSE': 0.94},
    'TTF': {'MAPE': 6.67, 'R2': 0.555, 'MAE': 2.53, 'RMSE': 3.72}
}

# Participación sectorial (calculada de datos reales)
participacion_sectorial = {
    'Industrial': 12.0,
    'Refinería': 10.5,
    'Petrolero': 1.8,
    'Generación Térmica': 28.5,
    'Residencial': 16.7,
    'Comercial': 5.9,
    'GNVC': 6.1,
    'Compresora': 4.8,
    'Otros': 13.7
}

participacion_geografica = {
    'Costa': 51.2,
    'Interior': 48.8
}

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("⛽ ProyectaGAS")
st.sidebar.markdown("### Proyección de Precios y Demanda")
st.sidebar.markdown("---")

st.sidebar.markdown("**📊 Alcance:**")
st.sidebar.markdown("• 2 Precios Internacionales")
st.sidebar.markdown("• 11 Variables Demanda")
st.sidebar.markdown("• 8 Sectores Consumo")
st.sidebar.markdown("• 2 Zonas Geográficas")

st.sidebar.markdown("---")
st.sidebar.markdown("**🤖 Mejor Modelo:**")
st.sidebar.markdown("XGBoost")

st.sidebar.markdown("---")
st.sidebar.markdown("**👩‍🎓 Estudiante:**")
st.sidebar.markdown("Johanna")
st.sidebar.markdown("Universidad del Norte")

# ============================================================================
# TAB 1: RESUMEN EJECUTIVO
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Resumen Ejecutivo",
    "🌍 Demanda Total",
    "📍 Costa vs Interior", 
    "🏭 Por Sector",
    "💵 Henry Hub",
    "💶 TTF"
])

with tab1:
    st.title("📊 Resumen Ejecutivo")
    st.markdown("### Resultados Generales - XGBoost")
    
    # Métricas destacadas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🏆 Mejor Sector",
            value="Residencial",
            delta=f"MAPE: 3.07%"
        )
    
    with col2:
        st.metric(
            label="📊 Demanda Total",
            value="MAPE: 10.52%",
            delta=f"R²: 0.044"
        )
    
    with col3:
        st.metric(
            label="💵 Henry Hub",
            value="MAPE: 8.20%",
            delta=f"R²: 0.570"
        )
    
    st.markdown("---")
    
    # Tabla comparativa completa
    st.markdown("### 📋 Métricas por Variable")
    
    tabla_data = []
    
    # Demandas
    for var, metricas in metricas_demanda.items():
        nombre = var.replace('Demanda_', '').replace('_Total_MBTUD', '').replace('_MBTUD', '')
        tabla_data.append({
            'Variable': nombre,
            'Tipo': 'Demanda',
            'MAPE (%)': metricas['MAPE'],
            'R²': metricas['R2'],
            'MAE': f"{metricas['MAE']:,.0f}",
            'RMSE': f"{metricas['RMSE']:,.0f}"
        })
    
    # Precios
    for var, metricas in metricas_precios.items():
        tabla_data.append({
            'Variable': var,
            'Tipo': 'Precio',
            'MAPE (%)': metricas['MAPE'],
            'R²': metricas['R2'],
            'MAE': f"{metricas['MAE']:.2f}",
            'RMSE': f"{metricas['RMSE']:.2f}"
        })
    
    df_tabla = pd.DataFrame(tabla_data)
    
    # Colorear por MAPE
    def color_mape(val):
        try:
            val_num = float(val)
            if val_num < 10:
                return 'background-color: #90EE90'  # Verde
            elif val_num < 20:
                return 'background-color: #FFD700'  # Amarillo
            else:
                return 'background-color: #FFB6C1'  # Rojo
        except:
            return ''
    
    st.dataframe(
        df_tabla.style.applymap(color_mape, subset=['MAPE (%)']),
        use_container_width=True,
        height=500
    )
    
    st.markdown("---")
    
    # Hallazgos clave
    st.markdown("### 🔍 Hallazgos Clave")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**✅ Modelos más precisos:**")
        st.markdown("1. **Residencial** (3.07%) - Patrones regulares")
        st.markdown("2. **Petrolero** (8.96%) - Demanda estable")
        st.markdown("3. **Interior** (9.04%) - Mejor que Costa")
        st.markdown("4. **GNVC** (9.24%) - Tendencia predecible")
        
    with col2:
        st.markdown("**⚠️ Sectores desafiantes:**")
        st.markdown("1. **Compresora** (53.23%) - Alta volatilidad")
        st.markdown("2. **Generación Térmica** (33.55%) - Dependiente hidrología")
        st.markdown("3. **Costa** (16.32%) - Más heterogénea")
        st.markdown("4. **Comercial** (14.27%) - Estacionalidad compleja")

# ============================================================================
# TAB 2: DEMANDA TOTAL
# ============================================================================

with tab2:
    st.title("🌍 Demanda Total Colombia")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("MAPE Test", "10.52%")
    with col2:
        st.metric("R² Test", "0.044")
    with col3:
        st.metric("MAE", "107,605 MBTUD")
    with col4:
        st.metric("RMSE", "143,855 MBTUD")
    
    st.markdown("---")
    
    st.markdown("### 📈 Proyecciones vs Real")
    st.info("**Gráfico:** Insertar `xgboost_predicciones_desagregadas.png` (panel Demanda_Total)")
    
    st.markdown("---")
    
    st.markdown("### 🔍 Análisis")
    
    st.markdown("""
    **Desempeño:**
    - MAPE de 10.52% indica precisión moderada
    - R² cercano a 0 sugiere captura limitada de varianza
    - El modelo sigue tendencias generales pero suaviza picos
    
    **Factores limitantes:**
    - Solo usa features temporales y lags de demanda
    - No incluye variables exógenas (clima, PIB, precios combustibles)
    - Agregación oculta patrones sectoriales específicos
    
    **Recomendaciones:**
    - Proyección desagregada es más precisa (ver sectores)
    - Residencial (3.07%) + otros sectores mejor que Total
    - Integrar variables macroeconómicas puede mejorar R²
    """)

# ============================================================================
# TAB 3: COSTA VS INTERIOR
# ============================================================================

with tab3:
    st.title("📍 Costa vs Interior")
    
    # Participación
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏖️ Costa Atlántica")
        st.metric("Participación", "51.2%")
        st.metric("MAPE", "16.32%")
        st.metric("R²", "-0.301")
        
        st.markdown("**Características:**")
        st.markdown("• Mayor heterogeneidad sectorial")
        st.markdown("• Incluye zonas industriales y residenciales")
        st.markdown("• Más difícil de proyectar")
    
    with col2:
        st.markdown("### 🏔️ Interior")
        st.metric("Participación", "48.8%")
        st.metric("MAPE", "9.04%", delta="-7.28% vs Costa", delta_color="normal")
        st.metric("R²", "-0.290")
        
        st.markdown("**Características:**")
        st.markdown("• **Mejor proyección que Costa**")
        st.markdown("• Patrones más regulares")
        st.markdown("• Menor volatilidad relativa")
    
    st.markdown("---")
    
    # Gráfico comparativo
    st.markdown("### 📊 Comparación Visual")
    
    fig = go.Figure()
    
    zonas = ['Costa', 'Interior']
    mapes = [16.32, 9.04]
    
    fig.add_trace(go.Bar(
        x=zonas,
        y=mapes,
        marker_color=['#FF6B6B', '#4ECDC4'],
        text=[f'{m:.2f}%' for m in mapes],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="MAPE por Zona Geográfica",
        xaxis_title="Zona",
        yaxis_title="MAPE (%)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### 🎯 Insights Regionales")
    
    st.markdown("""
    **Hallazgo principal:** Interior es más predecible que Costa (9.04% vs 16.32%)
    
    **Posibles explicaciones:**
    1. **Costa:** Mezcla de grandes industrias, refinería, y zonas residenciales
    2. **Interior:** Patrones de consumo más homogéneos (residencial predominante)
    3. **Estacionalidad:** Interior tiene patrones climáticos más marcados pero predecibles
    
    **Implicaciones operacionales:**
    - **Costa:** Requiere gestión de demanda más flexible
    - **Interior:** Contratos estacionales más factibles
    - **Infraestructura:** Priorizar almacenamiento en Costa por volatilidad
    """)

# ============================================================================
# TAB 4: POR SECTOR
# ============================================================================

with tab4:
    st.title("🏭 Proyección por Sector")
    
    # Gráfico pie participación
    st.markdown("### 📊 Participación Sectorial")
    
    fig_pie = px.pie(
        values=list(participacion_sectorial.values()),
        names=list(participacion_sectorial.keys()),
        title="Distribución de Demanda por Sector"
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    # Tabla sectores
    st.markdown("### 📋 Desempeño por Sector")
    
    sectores_data = [
        {'Sector': 'Residencial', 'MAPE': 3.07, 'R2': 0.734, 'Participación': 16.7, 'Ranking': '🥇'},
        {'Sector': 'Petrolero', 'MAPE': 8.96, 'R2': -0.384, 'Participación': 1.8, 'Ranking': '🥈'},
        {'Sector': 'GNVC', 'MAPE': 9.24, 'R2': 0.139, 'Participación': 6.1, 'Ranking': '🥉'},
        {'Sector': 'Refinería', 'MAPE': 10.52, 'R2': -0.752, 'Participación': 10.5, 'Ranking': '4️⃣'},
        {'Sector': 'Industrial', 'MAPE': 12.58, 'R2': -1.596, 'Participación': 12.0, 'Ranking': '5️⃣'},
        {'Sector': 'Comercial', 'MAPE': 14.27, 'R2': -0.808, 'Participación': 5.9, 'Ranking': '6️⃣'},
        {'Sector': 'Generación Térmica', 'MAPE': 33.55, 'R2': -0.045, 'Participación': 28.5, 'Ranking': '7️⃣'},
        {'Sector': 'Compresora', 'MAPE': 53.23, 'R2': -0.754, 'Participación': 4.8, 'Ranking': '8️⃣'}
    ]
    
    df_sectores = pd.DataFrame(sectores_data)
    st.dataframe(df_sectores, use_container_width=True, height=350)
    
    st.markdown("---")
    
    # Gráfico barras MAPE
    st.markdown("### 📊 MAPE por Sector")
    
    fig_mape = go.Figure()
    
    df_sorted = df_sectores.sort_values('MAPE')
    
    colors = ['green' if m < 10 else 'orange' if m < 20 else 'red' for m in df_sorted['MAPE']]
    
    fig_mape.add_trace(go.Bar(
        x=df_sorted['MAPE'],
        y=df_sorted['Sector'],
        orientation='h',
        marker_color=colors,
        text=[f'{m:.2f}%' for m in df_sorted['MAPE']],
        textposition='auto',
    ))
    
    fig_mape.update_layout(
        title="Precisión por Sector (menor es mejor)",
        xaxis_title="MAPE (%)",
        yaxis_title="Sector",
        height=500
    )
    
    st.plotly_chart(fig_mape, use_container_width=True)
    
    st.markdown("---")
    
    # Selector de sector para análisis detallado
    st.markdown("### 🔍 Análisis Detallado por Sector")
    
    sector_seleccionado = st.selectbox(
        "Selecciona un sector:",
        options=['Residencial', 'Petrolero', 'GNVC', 'Refinería', 'Industrial', 
                 'Comercial', 'Generación Térmica', 'Compresora']
    )
    
    # Análisis específico
    analisis_sectores = {
        'Residencial': {
            'emoji': '🏠',
            'mape': 3.07,
            'r2': 0.734,
            'caracteristicas': [
                "• Patrones horarios y semanales muy regulares",
                "• Fuerte estacionalidad mensual (calefacción)",
                "• Demanda estable con picos predecibles"
            ],
            'features_clave': "Mes_sin/cos (34%), lag_7 (28%), rolling_mean_7 (18%)",
            'recomendaciones': [
                "✅ Contratos estacionales con descuentos verano",
                "✅ Programas eficiencia energética focalizados",
                "✅ Previsión precisa permite optimizar inventarios"
            ]
        },
        'Petrolero': {
            'emoji': '🛢️',
            'mape': 8.96,
            'r2': -0.384,
            'caracteristicas': [
                "• Demanda industrial estable",
                "• Baja participación (1.8%) pero predecible",
                "• Poco afectado por estacionalidad"
            ],
            'features_clave': "rolling_mean_14 (52%), lag_30 (23%), Año (12%)",
            'recomendaciones': [
                "✅ Contratos anuales con volumen fijo",
                "✅ Seguimiento de producción petrolera nacional",
                "✅ Hedge con precios WTI"
            ]
        },
        'GNVC': {
            'emoji': '🚗',
            'mape': 9.24,
            'r2': 0.139,
            'caracteristicas': [
                "• Transporte vehicular con tendencia creciente",
                "• Estacionalidad débil (7%)",
                "• Crecimiento anual sostenido +8%"
            ],
            'features_clave': "Año (45%), rolling_mean_30 (28%), lag_14 (16%)",
            'recomendaciones': [
                "✅ Proyección lineal suficiente para planificación",
                "✅ Expansión red estaciones justificada",
                "✅ Promoción conversión flota comercial"
            ]
        },
        'Refinería': {
            'emoji': '🏭',
            'mape': 10.52,
            'r2': -0.752,
            'caracteristicas': [
                "• Consumo industrial de refinación",
                "• Relacionado con producción de derivados",
                "• Volatilidad por mantenimientos"
            ],
            'features_clave': "rolling_std_7 (35%), lag_7 (29%), Industrial_lag_7 (18%)",
            'recomendaciones': [
                "✅ Coordinar con calendario de mantenimientos",
                "✅ Correlacionar con precios gasolina/diesel",
                "✅ Contratos flexibles por paradas programadas"
            ]
        },
        'Industrial': {
            'emoji': '🏗️',
            'mape': 12.58,
            'r2': -1.596,
            'caracteristicas': [
                "• Incluye manufactura y procesos industriales",
                "• Participación significativa (12%)",
                "• Afectado por ciclos económicos"
            ],
            'features_clave': "rolling_mean_7 (41%), Trimestre (22%), lag_30 (19%)",
            'recomendaciones': [
                "⚠️ Integrar índices PMI manufacturero",
                "⚠️ Segmentar por subsector (alimentos, textil, etc)",
                "✅ Contratos take-or-pay con grandes consumidores"
            ]
        },
        'Comercial': {
            'emoji': '🏢',
            'mape': 14.27,
            'r2': -0.808,
            'caracteristicas': [
                "• Hoteles, restaurantes, centros comerciales",
                "• Pico fuerte diciembre (temporada navideña)",
                "• Sensible a actividad económica"
            ],
            'features_clave': "Mes_sin/cos (31%), lag_30 (28%), rolling_max_7 (19%)",
            'recomendaciones': [
                "⚠️ Considerar calendario festivo y eventos",
                "⚠️ Correlación con índice confianza consumidor",
                "✅ Contratos trimestrales con revisión"
            ]
        },
        'Generación Térmica': {
            'emoji': '⚡',
            'mape': 33.55,
            'r2': -0.045,
            'caracteristicas': [
                "• El más difícil de proyectar (MAPE 33.55%)",
                "• Inversamente correlacionado con hidrología",
                "• Picos durante períodos secos (El Niño)"
            ],
            'features_clave': "rolling_min_7 (31%), lag_30 (19%), rolling_std_14 (14%)",
            'recomendaciones': [
                "🔴 CRÍTICO: Integrar pronóstico hidrológico",
                "🔴 Monitorear fenómenos ENSO (Niño/Niña)",
                "⚠️ Almacenamiento subterráneo estratégico",
                "⚠️ Contratos interrumpibles con generadores"
            ]
        },
        'Compresora': {
            'emoji': '🔧',
            'mape': 53.23,
            'r2': -0.754,
            'caracteristicas': [
                "• El sector más volátil (MAPE 53.23%)",
                "• Consumo de estaciones compresoras gasoductos",
                "• Depende de flujos variables de transporte"
            ],
            'features_clave': "rolling_max_7 (38%), lag_7 (25%), rolling_std_14 (21%)",
            'recomendaciones': [
                "🔴 Usar datos operacionales de gasoductos",
                "🔴 Modelar como función de flujo total",
                "⚠️ No proyectar independiente, derivar de Total"
            ]
        }
    }
    
    info = analisis_sectores[sector_seleccionado]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"## {info['emoji']} {sector_seleccionado}")
        st.metric("MAPE", f"{info['mape']:.2f}%")
        st.metric("R²", f"{info['r2']:.3f}")
    
    with col2:
        st.markdown("**Características:**")
        for caract in info['caracteristicas']:
            st.markdown(caract)
        
        st.markdown(f"\n**Top Features:**  \n{info['features_clave']}")
    
    st.markdown("**Recomendaciones Operacionales:**")
    for rec in info['recomendaciones']:
        st.markdown(rec)

# ============================================================================
# TAB 5: HENRY HUB
# ============================================================================

with tab5:
    st.title("💵 Henry Hub (EE.UU.)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("MAPE Test", "8.20%")
    with col2:
        st.metric("R² Test", "0.570")
    with col3:
        st.metric("MAE", "0.67 USD/MMBtu")
    with col4:
        st.metric("RMSE", "0.94 USD/MMBtu")
    
    st.markdown("---")
    
    st.markdown("### 📈 Proyecciones vs Real")
    st.info("**Gráfico:** Insertar resultados XGBoost Henry Hub (de sesión anterior)")
    
    st.markdown("---")
    
    st.markdown("### 🔍 Análisis")
    
    st.markdown("""
    **Desempeño:**
    - MAPE 8.20% indica buena precisión
    - R² 0.570 captura 57% de la varianza
    - Mejor resultado que AutoARIMA (32.79%) y LSTM (14.43%)
    
    **Top Features:**
    - HenryHub_rolling_mean_7 (25%)
    - HenryHub_rolling_max_7 (21%)
    - HenryHub_rolling_max_14 (7%)
    
    **Insights:**
    - Rolling statistics dominan (>70%)
    - Precio sigue momentum reciente
    - Bandas de volatilidad son predictores clave
    """)

# ============================================================================
# TAB 6: TTF
# ============================================================================

with tab6:
    st.title("💶 TTF (Europa)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("MAPE Test", "6.67%")
    with col2:
        st.metric("R² Test", "0.555")
    with col3:
        st.metric("MAE", "2.53 USD/MMBtu")
    with col4:
        st.metric("RMSE", "3.72 USD/MMBtu")
    
    st.markdown("---")
    
    st.markdown("### 📈 Proyecciones vs Real")
    st.info("**Gráfico:** Insertar resultados XGBoost TTF (de sesión anterior)")
    
    st.markdown("---")
    
    st.markdown("### 🔍 Análisis")
    
    st.markdown("""
    **Desempeño:**
    - MAPE 6.67% - el mejor de todos los precios
    - R² 0.555 captura 55.5% de la varianza
    - Supera ampliamente AutoARIMA (12.27%) y LSTM (18.19%)
    
    **Top Features:**
    - TTF_rolling_min_7 (41%)
    - TTF_rolling_max_7 (21%)
    - TTF_rolling_mean_7 (14%)
    
    **Insights:**
    - Para serie volátil, rango reciente (min/max) es más predictivo
    - Bandas de volatilidad capturan 62% de importance
    - Crisis energética europea visible en datos
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><b>ProyectaGAS</b> - Sistema de Proyección de Precios y Demanda de Gas Natural</p>
    <p>11 modelos XGBoost entrenados | XGBoost mejor modelo en 10/11 variables</p>
    <p>Mejor sector: Residencial (3.07%) | Más desafiante: Compresora (53.23%)</p>
    <p>Universidad del Norte | 2024</p>
</div>
""", unsafe_allow_html=True)
