"""
ProyectaGAS - Dashboard Empresarial
Proyecciones de Demanda de Gas Natural y Precios Internacionales
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

# ===========================================================================
# CONFIGURACIÓN
# ===========================================================================

st.set_page_config(
    page_title="ProyectaGAS | Dashboard Ejecutivo",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================================================================
# CARGAR DATOS
# ===========================================================================

@st.cache_data
def cargar_datos():
    try:
        metricas_agregado = pd.read_csv('data/xgboost_metricas.csv')
        metricas_desagregado = pd.read_csv('data/xgboost_metricas_desagregadas.csv')
        pred_modelo1 = pd.read_csv('data/predicciones_modelo1_xgboost.csv', parse_dates=['Fecha'])
        pred_modelo2 = pd.read_csv('data/predicciones_modelo2_desagregado.csv', parse_dates=['Fecha'])
        return metricas_agregado, metricas_desagregado, pred_modelo1, pred_modelo2
    except FileNotFoundError as e:
        st.error(f"❌ Error: {e}\n\nAsegúrate de tener los archivos en data/")
        st.stop()

metricas_agregado, metricas_desagregado, pred_modelo1, pred_modelo2 = cargar_datos()

# Limpiar nombres de variables
metricas_agregado['Variable'] = metricas_agregado['Variable'].str.strip()
metricas_desagregado['Variable'] = metricas_desagregado['Variable'].str.strip()

# ===========================================================================
# SIDEBAR
# ===========================================================================

st.sidebar.title("⛽ ProyectaGAS")
st.sidebar.markdown("### Dashboard Ejecutivo")
st.sidebar.markdown("---")

# Selector de período
st.sidebar.markdown("**📅 Período de Análisis**")
fecha_min = pred_modelo1['Fecha'].min()
fecha_max = pred_modelo1['Fecha'].max()

fecha_inicio = st.sidebar.date_input(
    "Desde:",
    value=fecha_min,
    min_value=fecha_min,
    max_value=fecha_max
)

fecha_fin = st.sidebar.date_input(
    "Hasta:",
    value=fecha_max,
    min_value=fecha_min,
    max_value=fecha_max
)

st.sidebar.markdown("---")

# Filtrar datos por fecha
pred_modelo1_filtrado = pred_modelo1[
    (pred_modelo1['Fecha'] >= pd.to_datetime(fecha_inicio)) &
    (pred_modelo1['Fecha'] <= pd.to_datetime(fecha_fin))
]

pred_modelo2_filtrado = pred_modelo2[
    (pred_modelo2['Fecha'] >= pd.to_datetime(fecha_inicio)) &
    (pred_modelo2['Fecha'] <= pd.to_datetime(fecha_fin))
]

dias_proyeccion = len(pred_modelo1_filtrado)

st.sidebar.markdown(f"""
**Proyección:** {dias_proyeccion} días  
**Desde:** {fecha_inicio.strftime('%Y-%m-%d')}  
**Hasta:** {fecha_fin.strftime('%Y-%m-%d')}
""")

st.sidebar.markdown("---")
st.sidebar.info("**Modelo:** XGBoost  \n**Variables:** 13 (11 Demanda + 2 Precios)")

# ===========================================================================
# HEADER
# ===========================================================================

st.title("⛽ ProyectaGAS - Dashboard Ejecutivo")
st.markdown(f"### Proyección de Demanda y Precios | {dias_proyeccion} días")

# ===========================================================================
# TABS
# ===========================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Resumen Ejecutivo",
    "📈 Proyección Nacional",
    "🗺️ Proyección por Zona",
    "🏭 Proyección por Sector",
    "💰 Precios Internacionales",
    "📉 Desempeño del Modelo"
])

# ===========================================================================
# TAB 1: RESUMEN EJECUTIVO
# ===========================================================================

with tab1:
    st.header("Resumen Ejecutivo - Proyecciones Clave")
    
    # KPIs Principales
    col1, col2, col3, col4 = st.columns(4)
    
    # Demanda Total Proyectada
    demanda_total_prom = pred_modelo1_filtrado['Demanda_Total_pred'].mean()
    demanda_total_max = pred_modelo1_filtrado['Demanda_Total_pred'].max()
    
    with col1:
        st.metric(
            "Demanda Promedio Proyectada",
            f"{demanda_total_prom:,.0f} MBTUD",
            help="Demanda promedio nacional en el período"
        )
        st.caption(f"Pico: {demanda_total_max:,.0f} MBTUD")
    
    # Precio Henry Hub Proyectado
    hh_prom = pred_modelo1_filtrado['Henry_Hub_pred'].mean()
    hh_max = pred_modelo1_filtrado['Henry_Hub_pred'].max()
    
    with col2:
        st.metric(
            "Henry Hub Proyectado",
            f"${hh_prom:.2f}/MMBtu",
            help="Precio promedio proyectado"
        )
        st.caption(f"Pico: ${hh_max:.2f}/MMBtu")
    
    # Precio TTF Proyectado
    ttf_prom = pred_modelo1_filtrado['TTF_pred'].mean()
    ttf_max = pred_modelo1_filtrado['TTF_pred'].max()
    
    with col3:
        st.metric(
            "TTF Proyectado",
            f"${ttf_prom:.2f}/MMBtu",
            help="Precio promedio proyectado Europa"
        )
        st.caption(f"Pico: ${ttf_max:.2f}/MMBtu")
    
    # Spread HH-TTF
    spread = ttf_prom - hh_prom
    
    with col4:
        st.metric(
            "Spread TTF - HH",
            f"${spread:.2f}/MMBtu",
            delta=f"{(spread/hh_prom)*100:.1f}%",
            help="Diferencia de precio entre mercados"
        )
    
    st.markdown("---")
    
    # Proyección Demanda Nacional
    st.subheader("📈 Proyección Demanda Nacional")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig = go.Figure()
        
        # Submuestrear para mejor visualización
        df_plot = pred_modelo1_filtrado.iloc[::max(1, len(pred_modelo1_filtrado)//100)]
        
        fig.add_trace(go.Scatter(
            x=df_plot['Fecha'],
            y=df_plot['Demanda_Total_pred'],
            name='Proyección XGBoost',
            line=dict(color='#1f77b4', width=3),
            fill='tonexty',
            mode='lines'
        ))
        
        fig.update_layout(
            height=350,
            xaxis_title='Fecha',
            yaxis_title='MBTUD',
            hovermode='x unified',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Estadísticas**")
        st.metric("Promedio", f"{demanda_total_prom:,.0f}")
        st.metric("Mediana", f"{pred_modelo1_filtrado['Demanda_Total_pred'].median():,.0f}")
        st.metric("Máximo", f"{demanda_total_max:,.0f}")
        st.metric("Mínimo", f"{pred_modelo1_filtrado['Demanda_Total_pred'].min():,.0f}")
        
        rango = demanda_total_max - pred_modelo1_filtrado['Demanda_Total_pred'].min()
        st.caption(f"Rango: {rango:,.0f} MBTUD")
    
    st.markdown("---")
    
    # Proyección por Sector - Top 5
    st.subheader("🏭 Proyección por Sector - Top 5 Consumidores")
    
    sectores_cols = [col for col in pred_modelo2_filtrado.columns if '_pred' in col and 'Demanda_' in col]
    
    # Calcular promedios
    promedios = {}
    for col in sectores_cols:
        nombre = col.replace('Demanda_', '').replace('_Total_MBTUD_pred', '').replace('_', ' ')
        if nombre not in ['Total', 'Costa', 'Interior']:
            promedios[nombre] = pred_modelo2_filtrado[col].mean()
    
    # Top 5
    top5 = sorted(promedios.items(), key=lambda x: x[1], reverse=True)[:5]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure(data=[
            go.Bar(
                x=[x[1] for x in top5],
                y=[x[0] for x in top5],
                orientation='h',
                marker=dict(color='#2ca02c'),
                text=[f"{x[1]:,.0f}" for x in top5],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            height=300,
            xaxis_title='MBTUD Promedio',
            yaxis_title='',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Distribución %**")
        total_top5 = sum([x[1] for x in top5])
        for nombre, valor in top5:
            pct = (valor / demanda_total_prom) * 100
            st.metric(
                nombre.replace('GeneracionTermica', 'Gen. Térmica'),
                f"{pct:.1f}%",
                f"{valor:,.0f} MBTUD"
            )
    
    st.markdown("---")
    
    # Alertas y Recomendaciones
    st.subheader("⚠️ Alertas e Insights Operacionales")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **📊 Demanda Nacional**
        
        Proyección estable en rango {min:,.0f} - {max:,.0f} MBTUD.
        
        **Acción:** Mantener capacidad de suministro base.
        """.format(
            min=pred_modelo1_filtrado['Demanda_Total_pred'].min(),
            max=demanda_total_max
        ))
    
    with col2:
        if spread > 5:
            st.warning(f"""
            **💰 Spread HH-TTF Elevado**
            
            Diferencia de ${spread:.2f}/MMBtu favorece importación desde EE.UU.
            
            **Acción:** Evaluar contratos GNL indexados a Henry Hub.
            """)
        else:
            st.success("""
            **💰 Spread HH-TTF Normal**
            
            Mercados en equilibrio.
            
            **Acción:** Mantener estrategia actual.
            """)
    
    with col3:
        # Calcular volatilidad
        volatilidad = pred_modelo1_filtrado['Demanda_Total_pred'].std() / demanda_total_prom * 100
        
        if volatilidad > 15:
            st.warning(f"""
            **📈 Alta Variabilidad**
            
            Volatilidad: {volatilidad:.1f}%
            
            **Acción:** Aumentar inventarios de seguridad.
            """)
        else:
            st.success(f"""
            **📈 Demanda Estable**
            
            Volatilidad: {volatilidad:.1f}%
            
            **Acción:** Optimización normal.
            """)

# ===========================================================================
# TAB 2: PROYECCIÓN NACIONAL
# ===========================================================================

with tab2:
    st.header("Proyección Demanda Nacional")
    
    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Promedio", f"{demanda_total_prom:,.0f} MBTUD")
    
    with col2:
        st.metric("Máximo", f"{demanda_total_max:,.0f} MBTUD")
    
    with col3:
        st.metric("Mínimo", f"{pred_modelo1_filtrado['Demanda_Total_pred'].min():,.0f} MBTUD")
    
    with col4:
        desv = pred_modelo1_filtrado['Demanda_Total_pred'].std()
        st.metric("Desv. Std", f"{desv:,.0f} MBTUD")
    
    with col5:
        cv = (desv / demanda_total_prom) * 100
        st.metric("Coef. Variación", f"{cv:.1f}%")
    
    st.markdown("---")
    
    # Gráfico principal
    st.subheader("📊 Proyección Temporal")
    
    fig = go.Figure()
    
    df_plot = pred_modelo1_filtrado.iloc[::max(1, len(pred_modelo1_filtrado)//200)]
    
    # Banda de confianza (±10%)
    fig.add_trace(go.Scatter(
        x=df_plot['Fecha'],
        y=df_plot['Demanda_Total_pred'] * 1.1,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_plot['Fecha'],
        y=df_plot['Demanda_Total_pred'] * 0.9,
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(31, 119, 180, 0.2)',
        fill='tonexty',
        showlegend=True,
        name='Banda ±10%',
        hoverinfo='skip'
    ))
    
    # Proyección
    fig.add_trace(go.Scatter(
        x=df_plot['Fecha'],
        y=df_plot['Demanda_Total_pred'],
        name='Proyección',
        line=dict(color='#1f77b4', width=3),
        mode='lines'
    ))
    
    fig.update_layout(
        height=500,
        xaxis_title='Fecha',
        yaxis_title='MBTUD',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Distribución mensual
    st.subheader("📅 Distribución por Mes")
    
    pred_modelo1_filtrado['Mes'] = pred_modelo1_filtrado['Fecha'].dt.month
    mensual = pred_modelo1_filtrado.groupby('Mes')['Demanda_Total_pred'].agg(['mean', 'min', 'max'])
    
    meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    mensual.index = [meses[i-1] for i in mensual.index]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=mensual.index,
        y=mensual['mean'],
        name='Promedio',
        marker_color='#1f77b4',
        error_y=dict(
            type='data',
            symmetric=False,
            array=mensual['max'] - mensual['mean'],
            arrayminus=mensual['mean'] - mensual['min']
        )
    ))
    
    fig.update_layout(
        height=400,
        xaxis_title='Mes',
        yaxis_title='MBTUD',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Recomendaciones
    st.subheader("💡 Recomendaciones Operacionales")
    
    mes_mayor = mensual['mean'].idxmax()
    mes_menor = mensual['mean'].idxmin()
    amplitud = mensual['mean'].max() - mensual['mean'].min()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **📈 Planificación de Capacidad**
        
        - **Mes de mayor demanda:** {mes_mayor} ({mensual.loc[mes_mayor, 'mean']:,.0f} MBTUD)
        - **Mes de menor demanda:** {mes_menor} ({mensual.loc[mes_menor, 'mean']:,.0f} MBTUD)
        - **Amplitud estacional:** {amplitud:,.0f} MBTUD ({(amplitud/demanda_total_prom)*100:.1f}%)
        
        **Acciones:**
        - Asegurar capacidad de {mensual['max'].max():,.0f} MBTUD en picos
        - Optimizar inventarios para variación estacional
        """)
    
    with col2:
        st.markdown(f"""
        **🔧 Gestión de Contratos**
        
        - **Demanda base:** {mensual['min'].min():,.0f} MBTUD (contratos firmes)
        - **Demanda variable:** {mensual['max'].max() - mensual['min'].min():,.0f} MBTUD (contratos flexibles)
        - **Coeficiente variación:** {cv:.1f}%
        
        **Estrategia:**
        - 70% contratos largo plazo (base)
        - 30% contratos flexibles (picos)
        """)

# ===========================================================================
# TAB 3: PROYECCIÓN POR ZONA
# ===========================================================================

with tab3:
    st.header("Proyección por Zona Geográfica")
    
    # KPIs por zona
    costa_prom = pred_modelo2_filtrado['Demanda_Costa_Total_MBTUD_pred'].mean()
    interior_prom = pred_modelo2_filtrado['Demanda_Interior_Total_MBTUD_pred'].mean()
    total_zonas = costa_prom + interior_prom
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "🌊 Costa Atlántica",
            f"{costa_prom:,.0f} MBTUD",
            f"{(costa_prom/total_zonas)*100:.1f}%"
        )
    
    with col2:
        st.metric(
            "🏔️ Interior",
            f"{interior_prom:,.0f} MBTUD",
            f"{(interior_prom/total_zonas)*100:.1f}%"
        )
    
    with col3:
        diferencia = abs(costa_prom - interior_prom)
        st.metric(
            "Diferencia",
            f"{diferencia:,.0f} MBTUD",
            f"{(diferencia/total_zonas)*100:.1f}%"
        )
    
    st.markdown("---")
    
    # Gráficos comparativos
    col1, col2 = st.columns(2)
    
    df_plot = pred_modelo2_filtrado.iloc[::max(1, len(pred_modelo2_filtrado)//100)]
    
    with col1:
        st.subheader("🌊 Costa Atlántica")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_plot['Fecha'],
            y=df_plot['Demanda_Costa_Total_MBTUD_pred'],
            name='Proyección',
            line=dict(color='#ff7f0e', width=2),
            fill='tonexty'
        ))
        
        fig.update_layout(
            height=350,
            xaxis_title='Fecha',
            yaxis_title='MBTUD',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        **Características:**
        - Participación: {(costa_prom/total_zonas)*100:.1f}%
        - Promedio: {costa_prom:,.0f} MBTUD
        - Rango: {pred_modelo2_filtrado['Demanda_Costa_Total_MBTUD_pred'].min():,.0f} - {pred_modelo2_filtrado['Demanda_Costa_Total_MBTUD_pred'].max():,.0f} MBTUD
        
        **Sectores principales:**
        - Industrial (petroquímica, zona franca)
        - Refinería de Cartagena
        - Residencial urbano
        """)
    
    with col2:
        st.subheader("🏔️ Interior")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_plot['Fecha'],
            y=df_plot['Demanda_Interior_Total_MBTUD_pred'],
            name='Proyección',
            line=dict(color='#2ca02c', width=2),
            fill='tonexty'
        ))
        
        fig.update_layout(
            height=350,
            xaxis_title='Fecha',
            yaxis_title='MBTUD',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        **Características:**
        - Participación: {(interior_prom/total_zonas)*100:.1f}%
        - Promedio: {interior_prom:,.0f} MBTUD
        - Rango: {pred_modelo2_filtrado['Demanda_Interior_Total_MBTUD_pred'].min():,.0f} - {pred_modelo2_filtrado['Demanda_Interior_Total_MBTUD_pred'].max():,.0f} MBTUD
        
        **Sectores principales:**
        - Residencial (Bogotá, Medellín)
        - Generación térmica
        - Industrial manufacturero
        """)
    
    st.markdown("---")
    
    # Comparación directa
    st.subheader("📊 Comparación Temporal")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_plot['Fecha'],
        y=df_plot['Demanda_Costa_Total_MBTUD_pred'],
        name='Costa',
        line=dict(color='#ff7f0e', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_plot['Fecha'],
        y=df_plot['Demanda_Interior_Total_MBTUD_pred'],
        name='Interior',
        line=dict(color='#2ca02c', width=2)
    ))
    
    fig.update_layout(
        height=400,
        xaxis_title='Fecha',
        yaxis_title='MBTUD',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Estrategia por zona
    st.subheader("🎯 Estrategia Operacional por Zona")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Costa Atlántica**
        
        📦 **Infraestructura:**
        - Mayor capacidad de almacenamiento
        - Flexibilidad en contratos industriales
        - Acceso a terminales de GNL
        
        ⚡ **Gestión:**
        - Coordinación con grandes consumidores
        - Contratos interrumpibles
        - Provisión para paradas de refinería
        """)
    
    with col2:
        st.markdown("""
        **Interior**
        
        📦 **Infraestructura:**
        - Red de distribución residencial densa
        - Interconexión con hidrogeneración
        - Gasoductos principales
        
        ⚡ **Gestión:**
        - Estacionalidad predecible
        - Contratos de largo plazo
        - Coordinación con generación eléctrica
        """)

# ===========================================================================
# TAB 4: PROYECCIÓN POR SECTOR
# ===========================================================================

with tab4:
    st.header("Proyección por Sector de Consumo")
    
    # Selector de sector
    sectores_map = {
        'Residencial': 'Demanda_Residencial_Total_MBTUD_pred',
        'Industrial': 'Demanda_Industrial_Total_MBTUD_pred',
        'Comercial': 'Demanda_Comercial_Total_MBTUD_pred',
        'Generación Térmica': 'Demanda_GeneracionTermica_Total_MBTUD_pred',
        'Refinería': 'Demanda_Refineria_Total_MBTUD_pred',
        'Petrolero': 'Demanda_Petrolero_Total_MBTUD_pred',
        'GNVC (Transporte)': 'Demanda_GNVC_Total_MBTUD_pred',
        'Compresora': 'Demanda_Compresora_Total_MBTUD_pred'
    }
    
    sector_sel = st.selectbox("Selecciona un sector:", list(sectores_map.keys()))
    col_name = sectores_map[sector_sel]
    
    # KPIs del sector
    sector_prom = pred_modelo2_filtrado[col_name].mean()
    sector_max = pred_modelo2_filtrado[col_name].max()
    sector_min = pred_modelo2_filtrado[col_name].min()
    sector_pct = (sector_prom / demanda_total_prom) * 100
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Promedio Proyectado", f"{sector_prom:,.0f} MBTUD")
    
    with col2:
        st.metric("Máximo", f"{sector_max:,.0f} MBTUD")
    
    with col3:
        st.metric("Mínimo", f"{sector_min:,.0f} MBTUD")
    
    with col4:
        st.metric("Participación", f"{sector_pct:.1f}%")
    
    with col5:
        rango = sector_max - sector_min
        st.metric("Rango", f"{rango:,.0f} MBTUD")
    
    st.markdown("---")
    
    # Gráfico principal
    st.subheader(f"📈 Proyección: {sector_sel}")
    
    fig = go.Figure()
    
    df_plot = pred_modelo2_filtrado.iloc[::max(1, len(pred_modelo2_filtrado)//150)]
    
    fig.add_trace(go.Scatter(
        x=df_plot['Fecha'],
        y=df_plot[col_name],
        name='Proyección',
        line=dict(color='#9467bd', width=3),
        fill='tonexty',
        mode='lines'
    ))
    
    fig.update_layout(
        height=450,
        xaxis_title='Fecha',
        yaxis_title='MBTUD',
        hovermode='x unified',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Análisis específico por sector
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📊 Análisis: {sector_sel}")
        
        # Distribución mensual
        pred_modelo2_filtrado['Mes'] = pred_modelo2_filtrado['Fecha'].dt.month
        mensual_sector = pred_modelo2_filtrado.groupby('Mes')[col_name].mean()
        
        meses_abr = ['E', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        mensual_sector.index = [meses_abr[i-1] for i in mensual_sector.index]
        
        fig = go.Figure(data=[
            go.Bar(x=mensual_sector.index, y=mensual_sector.values, marker_color='#9467bd')
        ])
        
        fig.update_layout(
            height=300,
            xaxis_title='Mes',
            yaxis_title='MBTUD Promedio',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📋 Estadísticas")
        
        st.metric("Media", f"{sector_prom:,.0f}")
        st.metric("Mediana", f"{pred_modelo2_filtrado[col_name].median():,.0f}")
        st.metric("Desv. Std", f"{pred_modelo2_filtrado[col_name].std():,.0f}")
        
        cv_sector = (pred_modelo2_filtrado[col_name].std() / sector_prom) * 100
        st.metric("Coef. Var.", f"{cv_sector:.1f}%")
        
        st.metric("P95", f"{pred_modelo2_filtrado[col_name].quantile(0.95):,.0f}")
        st.metric("P5", f"{pred_modelo2_filtrado[col_name].quantile(0.05):,.0f}")
    
    st.markdown("---")
    
    # Recomendaciones por sector
    st.subheader("💡 Recomendaciones Operacionales")
    
    recomendaciones = {
        'Residencial': {
            'caracteristicas': '• Patrón estacional fuerte\n• Picos en meses fríos\n• Alta predictibilidad',
            'estrategia': '• Contratos estacionales diferenciados\n• Gestión de picos invernales\n• Programas de eficiencia energética'
        },
        'Industrial': {
            'caracteristicas': '• Correlación con actividad económica\n• Sensible a ciclos\n• Mix heterogéneo',
            'estrategia': '• Contratos indexados a PMI\n• Flexibilidad en volúmenes\n• Segmentar por subsector'
        },
        'Comercial': {
            'caracteristicas': '• Pico diciembre (+35%)\n• Horarios laborales\n• Estacionalidad comercial',
            'estrategia': '• Provisión fin de año\n• Tarifas incentivadas fuera de pico\n• Contratos trimestrales'
        },
        'Generación Térmica': {
            'caracteristicas': '• Alta volatilidad\n• Complementa hidráulica\n• Picos en El Niño',
            'estrategia': '• CRÍTICO: Integrar hidrología\n• Monitoreo ENSO\n• Contratos de respaldo flexibles'
        },
        'Refinería': {
            'caracteristicas': '• Volatilidad por paradas\n• Mantenimientos programados\n• Cartagena dominante',
            'estrategia': '• Coordinación mantenimientos\n• Cláusulas de flexibilidad\n• Inventarios ampliados'
        },
        'Petrolero': {
            'caracteristicas': '• Muy estable\n• Baja volatilidad\n• Operación continua',
            'estrategia': '• Contratos largo plazo fijos\n• Bajo riesgo\n• Inventarios mínimos'
        },
        'GNVC (Transporte)': {
            'caracteristicas': '• Crecimiento +8% anual\n• Expansión red\n• Urbano principalmente',
            'estrategia': '• Proyectar crecimiento\n• Expansión infraestructura\n• Incentivos conversión'
        },
        'Compresora': {
            'caracteristicas': '• Alta volatilidad\n• Depende de flujos\n• No independiente',
            'estrategia': '• NO proyectar independiente\n• Modelar como f(Total)\n• Coordinación transporte'
        }
    }
    
    if sector_sel in recomendaciones:
        rec = recomendaciones[sector_sel]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Características del Sector:**")
            st.info(rec['caracteristicas'])
        
        with col2:
            st.markdown("**Estrategia Recomendada:**")
            st.success(rec['estrategia'])

# ===========================================================================
# TAB 5: PRECIOS INTERNACIONALES
# ===========================================================================

with tab5:
    st.header("Precios Internacionales de Gas Natural")
    
    # KPIs comparativos
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Henry Hub Promedio",
            f"${hh_prom:.2f}/MMBtu",
            help="Precio promedio proyectado EE.UU."
        )
    
    with col2:
        st.metric(
            "TTF Promedio",
            f"${ttf_prom:.2f}/MMBtu",
            help="Precio promedio proyectado Europa"
        )
    
    with col3:
        st.metric(
            "Spread TTF - HH",
            f"${spread:.2f}/MMBtu",
            delta=f"{(spread/hh_prom)*100:.1f}%"
        )
    
    st.markdown("---")
    
    # Comparación precios
    st.subheader("📊 Comparación de Mercados")
    
    fig = go.Figure()
    
    df_plot = pred_modelo1_filtrado.iloc[::max(1, len(pred_modelo1_filtrado)//100)]
    
    fig.add_trace(go.Scatter(
        x=df_plot['Fecha'],
        y=df_plot['Henry_Hub_pred'],
        name='Henry Hub (EE.UU.)',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_plot['Fecha'],
        y=df_plot['TTF_pred'],
        name='TTF (Europa)',
        line=dict(color='#ff7f0e', width=2)
    ))
    
    fig.update_layout(
        height=450,
        xaxis_title='Fecha',
        yaxis_title='USD/MMBtu',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Detalle por mercado
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🇺🇸 Henry Hub (EE.UU.)")
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Promedio", f"${hh_prom:.2f}")
        with col_b:
            st.metric("Máximo", f"${hh_max:.2f}")
        with col_c:
            st.metric("Mínimo", f"${pred_modelo1_filtrado['Henry_Hub_pred'].min():.2f}")
        
        st.markdown("""
        **Características:**
        - Mercado líquido y maduro
        - Producción shale abundante
        - Estacionalidad marcada
        
        **Rango típico:** $2-4/MMBtu  
        **Drivers:** Almacenamiento, clima, exportaciones GNL
        
        **Aplicaciones para Colombia:**
        - Referencia contratos importación GNL
        - Indexación con spread
        - Hedge en NYMEX futures
        """)
    
    with col2:
        st.subheader("🇪🇺 TTF (Europa)")
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Promedio", f"${ttf_prom:.2f}")
        with col_b:
            st.metric("Máximo", f"${ttf_max:.2f}")
        with col_c:
            st.metric("Mínimo", f"${pred_modelo1_filtrado['TTF_pred'].min():.2f}")
        
        st.markdown("""
        **Características:**
        - Mayor volatilidad
        - Suministro ruso reducido
        - Competencia GNL con Asia
        
        **Rango típico:** $8-15/MMBtu  
        **Drivers:** Geopolítica, almacenamiento, clima europeo
        
        **Aplicaciones para Colombia:**
        - Competencia GNL global
        - Arbitraje internacional
        - Diversificación portafolio
        """)
    
    st.markdown("---")
    
    # Análisis de spread
    st.subheader("💰 Análisis de Spread y Oportunidades")
    
    spread_serie = pred_modelo1_filtrado['TTF_pred'] - pred_modelo1_filtrado['Henry_Hub_pred']
    spread_prom = spread_serie.mean()
    spread_max = spread_serie.max()
    spread_min = spread_serie.min()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure()
        
        df_plot = pred_modelo1_filtrado.iloc[::max(1, len(pred_modelo1_filtrado)//100)]
        spread_plot = df_plot['TTF_pred'] - df_plot['Henry_Hub_pred']
        
        fig.add_trace(go.Scatter(
            x=df_plot['Fecha'],
            y=spread_plot,
            name='Spread TTF - HH',
            line=dict(color='#2ca02c', width=2),
            fill='tozeroy'
        ))
        
        fig.add_hline(y=spread_prom, line_dash="dash", line_color="red", 
                      annotation_text=f"Promedio: ${spread_prom:.2f}")
        
        fig.update_layout(
            height=350,
            xaxis_title='Fecha',
            yaxis_title='Spread (USD/MMBtu)',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Estadísticas Spread**")
        st.metric("Promedio", f"${spread_prom:.2f}")
        st.metric("Máximo", f"${spread_max:.2f}")
        st.metric("Mínimo", f"${spread_min:.2f}")
        
        st.markdown("---")
        
        if spread_prom > 5:
            st.success("""
            **🔥 Oportunidad**
            
            Spread elevado favorece:
            - Importación desde EE.UU.
            - Contratos indexados HH
            - Arbitraje GNL
            """)
        elif spread_prom > 3:
            st.info("""
            **✓ Normal**
            
            Spread en rango normal.
            Mantener estrategia.
            """)
        else:
            st.warning("""
            **⚠️ Spread Bajo**
            
            Evaluar competitividad
            contratos actuales.
            """)

# ===========================================================================
# TAB 6: DESEMPEÑO DEL MODELO
# ===========================================================================

with tab6:
    st.header("Desempeño del Modelo XGBoost")
    
    st.info("""
    Esta sección presenta métricas de precisión del modelo. Las proyecciones mostradas 
    en los demás tabs se basan en el desempeño aquí documentado.
    """)
    
    # Comparación de modelos
    st.subheader("📊 Comparación de Modelos - Precios")
    
    # Tabla comparativa precios
    comp_precios = []
    
    for var in ['Demanda', 'Henry Hub', 'TTF']:
        var_clean = var.strip()
        if var_clean in metricas_agregado['Variable'].values:
            row = metricas_agregado[metricas_agregado['Variable'] == var_clean].iloc[0]
            comp_precios.append({
                'Variable': var,
                'MAPE (%)': row['MAPE_Test'],
                'R²': row['R2_Test'],
                'MAE': row['MAE_Test'],
                'RMSE': row['RMSE_Test']
            })
    
    if comp_precios:
        df_comp = pd.DataFrame(comp_precios)
        st.dataframe(
            df_comp.style.format({
                'MAPE (%)': '{:.2f}',
                'R²': '{:.3f}',
                'MAE': '{:.2f}',
                'RMSE': '{:.2f}'
            }).background_gradient(subset=['MAPE (%)'], cmap='RdYlGn_r'),
            use_container_width=True,
            hide_index=True
        )
    
    st.markdown("---")
    
    # Métricas por sector
    st.subheader("📊 Desempeño por Sector")
    
    df_sectores = metricas_desagregado.copy()
    df_sectores['Variable'] = df_sectores['Variable'].str.replace('Demanda_', '').str.replace('_Total_MBTUD', '').str.replace('_', ' ')
    df_sectores = df_sectores.sort_values('MAPE_Test')
    
    # Clasificación
    def clasificar(mape):
        if mape < 5:
            return "🟢 Excelente"
        elif mape < 10:
            return "🟡 Bueno"
        elif mape < 20:
            return "🟠 Aceptable"
        else:
            return "🔴 Requiere mejora"
    
    df_sectores['Clasificación'] = df_sectores['MAPE_Test'].apply(clasificar)
    
    st.dataframe(
        df_sectores[['Variable', 'MAPE_Test', 'R2_Test', 'Clasificación']].style.format({
            'MAPE_Test': '{:.2f}%',
            'R2_Test': '{:.3f}'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    
    # Gráfico de MAPE
    st.subheader("📈 MAPE por Variable")
    
    fig = px.bar(
        df_sectores,
        x='Variable',
        y='MAPE_Test',
        color='MAPE_Test',
        color_continuous_scale='RdYlGn_r',
        labels={'MAPE_Test': 'MAPE (%)'}
    )
    
    fig.update_layout(
        height=400,
        xaxis_tickangle=-45,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Interpretación
    st.subheader("💡 Interpretación de Métricas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **MAPE (Mean Absolute Percentage Error):**
        - Mide el error promedio en porcentaje
        - Valores menores son mejores
        - <5%: Excelente predicción
        - 5-10%: Buena predicción
        - 10-20%: Aceptable
        - >20%: Requiere mejoras
        
        **R² (Coeficiente de Determinación):**
        - Mide qué % de varianza captura el modelo
        - Rango: -∞ a 1
        - >0.7: Excelente
        - 0.4-0.7: Bueno
        - 0-0.4: Moderado
        - <0: Modelo peor que promedio simple
        """)
    
    with col2:
        st.markdown("""
        **Hallazgos Clave:**
        
        ✅ **Fortalezas:**
        - Residencial: 3.07% MAPE (excelente)
        - TTF: 6.67% MAPE (mejor precio)
        - 6 de 11 sectores con MAPE <10%
        
        ⚠️ **Áreas de Mejora:**
        - Generación Térmica: 33.55% MAPE
          → Requiere variables hidrológicas
        - Compresora: 53.23% MAPE
          → Mejor modelar como f(Total)
        
        📊 **Conclusión:**
        El modelo XGBoost proporciona proyecciones 
        confiables para planificación operacional y 
        estratégica, con alta precisión en sectores 
        clave y precios internacionales.
        """)

# ===========================================================================
# FOOTER
# ===========================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><b>ProyectaGAS Dashboard Ejecutivo</b> | Universidad del Norte</p>
    <p>Modelo XGBoost | 13 Variables | Horizonte {dias} días</p>
</div>
""".format(dias=dias_proyeccion), unsafe_allow_html=True)
