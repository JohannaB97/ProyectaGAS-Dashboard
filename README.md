# ProyectaGAS Dashboard 
Dashboard interactivo para proyección de precios internacionales y demanda desagregada de gas natural en Colombia mediante Machine Learning.

##  Características

- **2 Precios Internacionales:** Henry Hub (EE.UU.) y TTF (Europa)
- **11 Variables de Demanda:** Total, Costa, Interior + 8 sectores
- **8 Sectores de Consumo:** Industrial, Refinería, Petrolero, Generación Térmica, Residencial, Comercial, GNVC, Compresora
- **Análisis Geográfico:** Costa Atlántica vs Interior
- **Modelo:** XGBoost (mejor desempeño)

##  Resultados Destacados

- **Mejor Sector:** Residencial (MAPE 3.07%, R² 0.734)
- **Mejor Precio:** TTF (MAPE 6.67%, R² 0.555)
- **Hallazgo Regional:** Interior más predecible que Costa (9.04% vs 16.32%)

##  Ejecución Local

```bash
# Clonar repositorio
git clone https://github.com/JohannaB97/ProyectaGAS-Dashboard.git
cd proyectagas-dashboard

---

**⚠️ Nota:** Este dashboard presenta resultados de modelos entrenados. No incluye capacidad de reentrenamiento en tiempo real.
