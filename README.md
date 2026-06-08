🧠 eFRAUD — a Machine Learning Project
Fraud Detection in Financial Transactions using Machine Learning
📋 Índice / Table of Contents
Resumen Ejecutivo / Executive Summary

Descripción del Proyecto / Project Description

Requisitos / Requirements

Instalación / Installation

Uso / Usage

Datos / Data

Metodología / Methodology

Resultados / Results

Aplicación Web / Web App

Aprendizajes y Conclusiones / Learnings & Conclusions

Autora / Author

Licencia / License

📝 1. Resumen Ejecutivo / Executive Summary
Español
Este proyecto desarrolla un sistema de detección de fraude en transacciones financieras utilizando técnicas de machine learning.
Se trabaja con datos reales/anonimizados, altamente desbalanceados (<10% fraude), aplicando:

Limpieza y exploración de datos

Ingeniería de variables

Modelos supervisados y no supervisados

Optimización de hiperparámetros

Balanceo de clases (SMOTE)

Aplicación web interactiva (Streamlit)

El modelo final (XGBoost) obtiene:

Precision: 93%

Recall: 80%

F1-score: 86%

ROC-AUC: 0.6739

El análisis no supervisado (KMeans) respalda los patrones detectados.

English
This project builds an automated fraud detection system using machine learning.
We work with real/anonymized, highly imbalanced data (<10% fraud), applying:

Data cleaning and EDA

Feature engineering

Supervised and unsupervised models

Hyperparameter optimization

Class balancing (SMOTE)

Interactive Streamlit web app

Final model (XGBoost) achieves:

Precision: 93%

Recall: 80%

F1-score: 86%

ROC-AUC: 0.6739

KMeans clustering supports the findings.

🎯 2. Descripción del Proyecto / Project Description
Objetivo
Desarrollar un sistema automatizado capaz de identificar transacciones fraudulentas mediante patrones detectados con machine learning.

Objetivos específicos
Entrenar múltiples modelos (≥5 supervisados + 1 no supervisado)

Optimizar hiperparámetros (GridSearch, pipelines)

Minimizar falsos positivos manteniendo alto recall

Crear una aplicación web interactiva

Documentación profesional y reproducible

Hipótesis
Existen patrones detectables en las transacciones que permiten identificar fraude de forma más eficaz que reglas estáticas.

🔧 3. Requisitos / Requirements
Tecnologías principales
Python 3.9+

Pandas, NumPy

Scikit-learn

XGBoost / LightGBM

Imbalanced-learn (SMOTE)

Streamlit

Plotly / Matplotlib / Seaborn

Hardware recomendado
RAM: 8–16GB

CPU multi-core

GPU opcional (acelera boosting)

🚀 4. Instalación / Installation
1. Clonar el repositorio
Código
git clone https://github.com/tu-usuario/proyecto-deteccion-fraude.git
cd proyecto-deteccion-fraude
2. Crear entorno virtual
Código
python -m venv venv
source venv/bin/activate
3. Instalar dependencias
Código
pip install -r app_streamlit/requirements.txt
4. (Opcional) Instalar Jupyter
Código
pip install jupyter notebook
💻 5. Uso / Usage
Opción 1: Notebooks Jupyter
Paso 1 — Adquisición de datos
Código
notebooks/01_Fuentes.ipynb
Paso 2 — Limpieza y EDA
Código
notebooks/02_LimpiezaEDA.ipynb
Paso 3 — Entrenamiento y evaluación
Código
notebooks/03_Entrenamiento_Evaluacion.ipynb
Opción 2: Scripts Python
Procesar datos:

Código
python src/data_processing.py
Entrenar modelos:

Código
python src/training.py
Evaluar modelos:

Código
python src/evaluation.py
Opción 3: Aplicación Streamlit
Código
cd app_streamlit
streamlit run app.py
📊 6. Datos / Data
Dataset original:
https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets (kaggle.com in Bing)

Variables principales
(Tablas mantenidas tal cual las tenías)

Distribución:

No Fraude: 91%

Fraude: 9%

Ratio: 10:1

Estrategia: SMOTE para balanceo.

🔬 7. Metodología / Methodology
1. Exploración y Limpieza
Valores nulos y duplicados

Outliers

Distribuciones

Correlaciones

Análisis del desbalance

2. Feature Engineering
Incluye:

Variables temporales (hora, día, mes, fin de semana)

Edad de la cuenta

Días hasta expiración de tarjeta

Flags de expiración

Codificación categórica

Normalización

3. Modelado
Modelos supervisados
Logistic Regression

Decision Tree

Random Forest

Gradient Boosting

XGBoost ⭐

LightGBM

SVM

KNN

Modelo no supervisado
KMeans (detección de anomalías)

Técnicas aplicadas
Pipelines

GridSearchCV

Cross-validation

SMOTE

Train-test estratificado

🏆 8. Resultados / Results
Mejor modelo: XGBoost Classifier
Hiperparámetros óptimos
Código
{
 'n_estimators': 200,
 'max_depth': 7,
 'learning_rate': 0.1,
 'subsample': 0.8
}
Métricas en test set
Métrica	Valor
Precision	93%
Recall	80%
F1-Score	86%
ROC-AUC	0.6739


Interpretación
Precision alta (93%) → pocos falsos positivos

Recall sólido (80%) → detecta la mayoría de fraudes

F1-score equilibrado (86%)

ROC-AUC moderado (0.67) → margen de mejora con features temporales o modelos más complejos

Feature Importance
Top 3 features:

mcc_encoded

merchant_state_encoded

zip_encoded

🌐 9. Aplicación Web / Web App
Incluye:

🏠 Home
Descripción del proyecto

Métricas principales

🔮 Predicción individual
Formulario

Probabilidad de fraude

Recomendaciones

📊 Análisis por lotes
Carga CSV

Visualizaciones interactivas

Descarga de resultados

📈 Métricas del modelo
Gráficos

Importancia de variables

Explicabilidad

🎓 10. Aprendizajes y Conclusiones
Hallazgos principales
El modelo identifica patrones claros de fraude

Las variables de balance y monto son críticas

SMOTE mejora significativamente el recall

Boosting supera a modelos lineales

Limitaciones
Dependencia de la calidad del dataset

Necesidad de reentrenamiento periódico

Posibles sesgos históricos

Mejoras futuras
Más features temporales

Modelos deep learning (LSTM, Autoencoders)

API REST para producción

Monitorización en tiempo real

Feedback loop

👩‍💻 11. Autora / Author
Sara Gil Martín-Serrano  
📧 saragms217@gmail.com
💼 LinkedIn: https://www.linkedin.com/in/sara-gil-martín-serrano-84742310b/ (linkedin.com in Bing)  
🐙 GitHub: https://github.com/SaraGMS

📄 12. Licencia / License
MIT License
Proyecto desarrollado en The Bridge (2025).

