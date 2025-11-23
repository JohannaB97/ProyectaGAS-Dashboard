# ProyectaGAS Dashboard 🌎⛽

Dashboard interactivo para proyección de precios internacionales y demanda desagregada de gas natural en Colombia mediante Machine Learning.

## 📊 Características

- **2 Precios Internacionales:** Henry Hub (EE.UU.) y TTF (Europa)
- **11 Variables de Demanda:** Total, Costa, Interior + 8 sectores
- **8 Sectores de Consumo:** Industrial, Refinería, Petrolero, Generación Térmica, Residencial, Comercial, GNVC, Compresora
- **Análisis Geográfico:** Costa Atlántica vs Interior
- **Modelo:** XGBoost (mejor desempeño)

## 🎯 Resultados Destacados

- **Mejor Sector:** Residencial (MAPE 3.07%, R² 0.734)
- **Mejor Precio:** TTF (MAPE 6.67%, R² 0.555)
- **Hallazgo Regional:** Interior más predecible que Costa (9.04% vs 16.32%)

## 🚀 Ejecución Local

```bash
# Clonar repositorio
git clone https://github.com/JohannaB97/ProyectaGAS-Dashboard.git
cd proyectagas-dashboard

# Instalar dependencias
pip install -r requirements.txt

# Correr app
streamlit run app.py
```

La aplicación se abrirá en `http://localhost:8501`

## 🏗️ Estructura del Proyecto

```
proyectagas-dashboard/
├── app.py                  # Aplicación principal Streamlit
├── requirements.txt        # Dependencias Python
├── README.md              # Este archivo
├── .gitignore             # Archivos a ignorar
└── assets/                # Imágenes y recursos (opcional)
```

## 🎓 Contexto Académico

**Proyecto de Grado:** Proyección de Precios y Demanda de Gas Natural mediante Machine Learning

**Estudiante:** Johanna  
**Universidad:** Universidad del Norte  
**Año:** 2024  

## 📖 Metodología

### Datos
- **Período:** 2015-2025 (~3,800 días)
- **Fuentes:** 
  - Precios: EIA (Henry Hub), ICE (TTF)
  - Demanda: CREG-SIGNE (Sistema de Información Gas Natural Colombia)

### Modelos Comparados
1. AutoARIMA (baseline estadístico)
2. LSTM (redes neuronales recurrentes)
3. **XGBoost** (ganador - gradient boosting)

### Feature Engineering
- Features temporales: año, mes, día, encodings cíclicos
- Lags: 7, 14, 30 días
- Rolling statistics: media, std (ventanas 7, 14, 30)
- Total: ~150 features por modelo

## 📈 Resultados Principales

### Precios Internacionales

| Variable | Modelo | MAPE | R² |
|----------|--------|------|-----|
| TTF | XGBoost | 6.67% | 0.555 |
| Henry Hub | XGBoost | 8.20% | 0.570 |

### Demanda por Sector (Top 3)

| Sector | MAPE | R² | Ranking |
|--------|------|-----|---------|
| Residencial | 3.07% | 0.734 | 🥇 |
| Petrolero | 8.96% | -0.384 | 🥈 |
| GNVC | 9.24% | 0.139 | 🥉 |

### Demanda por Zona

| Zona | MAPE | Participación |
|------|------|---------------|
| Interior | 9.04% | 48.8% |
| Costa | 16.32% | 51.2% |

## 🔍 Insights Clave

1. **Desagregación mejora precisión:** Residencial (3.07%) supera significativamente proyección agregada (10.52%)

2. **Heterogeneidad regional:** Interior 1.8× más predecible que Costa

3. **Rolling statistics dominan:** >70% de feature importance en mayoría de variables

4. **Generación Térmica es desafiante:** MAPE 33.55% - requiere integración con pronóstico hidrológico

## 🛠️ Tecnologías Utilizadas

- **Python 3.11**
- **Streamlit** - Dashboard interactivo
- **Plotly** - Visualizaciones
- **Pandas/NumPy** - Procesamiento de datos
- **XGBoost** - Modelo ML (entrenamiento offline)

## 📧 Contacto

Para más información sobre el proyecto:
- **Email:** [tu-email]@uninorte.edu.co
- **LinkedIn:** [tu-perfil]
- **GitHub:** [tu-usuario]

## 📄 Licencia

Este proyecto fue desarrollado como parte de un trabajo de grado académico en Universidad del Norte.

## 🙏 Agradecimientos

- Universidad del Norte - Infraestructura computacional
- CREG - Acceso a datos SIGNE
- XM - Información contextual del sistema

---

**⚠️ Nota:** Este dashboard presenta resultados de modelos entrenados. No incluye capacidad de reentrenamiento en tiempo real.
