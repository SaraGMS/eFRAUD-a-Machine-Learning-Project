# 🔍 eFRAUD: a Machine Learning Project

Detección de Fraude en Transacciones Financieras / Fraud Detection in Financial Transactions

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Status](https://img.shields.io/badge/Status-En%20Desarrollo-yellow.svg)

</div>

---

## 📋 Índice / Table of Contents

- [Resumen Ejecutivo / Executive Summary](#-resumen-ejecutivo--executive-summary)
- [Descripción del Proyecto / Project Description](#-descripción-del-proyecto--project-description)
- [Estructura del Proyecto / Project Structure](#-estructura-del-proyecto--project-structure)
- [Requisitos / Requirements](#-requisitos--requirements)
- [Instalación / Installation](#-instalación--installation)
- [Uso / Usage](#-uso--usage)
- [Datos / Data](#-datos--data)
- [Metodología / Methodology](#-metodología--methodology)
- [Resultados / Results](#-resultados--results)
- [Aplicación Web / Web App](#-aplicación-web--web-app)
- [Autora / Author](#-autora--author)

---

## 📝 Resumen Ejecutivo / Executive Summary

### Español

Este proyecto de detección de fraude en transacciones financieras tiene como objetivo identificar automáticamente operaciones fraudulentas utilizando machine learning. Se trabajó con datos reales/anonimizados, altamente desbalanceados (<2% fraude), aplicando limpieza, ingeniería de variables y modelos supervisados y no supervisados. El modelo final (XGBoost/LightGBM) logra un ROC-AUC >0.90, detectando la mayoría de fraudes con pocos falsos positivos. El análisis no supervisado (KMeans) respalda los hallazgos. Se recomienda actualizar y monitorizar el sistema periódicamente.

### English

This fraud detection project aims to automatically identify fraudulent financial transactions using machine learning. We worked with real/anonymized, highly imbalanced data (<2% fraud), applying cleaning, feature engineering, and both supervised and unsupervised models. The final model (XGBoost/LightGBM) achieves ROC-AUC >0.90, detecting most frauds with few false positives. Unsupervised analysis (KMeans) supports the findings. Regular updates and monitoring are recommended.

---

## 🎯 Descripción del Proyecto / Project Description

### Español

Desarrollar un sistema automatizado para la detección de fraude en transacciones financieras, identificando patrones sospechosos y clasificando operaciones como legítimas o fraudulentas mediante técnicas avanzadas de machine learning.

**Objetivos:**
- Entrenar múltiples modelos de ML (mínimo 5 supervisados + 1 no supervisado)
- Optimizar hiperparámetros (GridSearch, pipelines)
- Alta precisión y bajo falso positivo
- Aplicación web interactiva (Streamlit)
- Documentación profesional

**Hipótesis:**
> Existen patrones en las transacciones que permiten identificar fraudes con modelos de machine learning más eficaces que reglas simples.

### English

Develop an automated system for fraud detection in financial transactions, identifying suspicious patterns and classifying operations as legitimate or fraudulent using advanced machine learning techniques.

**Objectives:**
- Train multiple ML models (at least 5 supervised + 1 unsupervised)
- Hyperparameter optimization (GridSearch, pipelines)

## 🌐 Aplicación Web / Web App

### Español
La aplicación Streamlit permite:
1. Inicio: descripción, métricas, info general
2. Predicción individual: formulario, predicción en tiempo real, recomendaciones
3. Análisis por lotes: carga de CSV, análisis masivo, visualizaciones, descarga
4. Métricas del modelo: detalles, gráficos, rendimiento

### English
The Streamlit app provides:
1. Home: description, metrics, general info
2. Individual prediction: form, real-time prediction, recommendations
3. Batch analysis: CSV upload, bulk analysis, visualizations, download
4. Model metrics: details, charts, performance


---


## 💻 Uso / Usage

### Español

**Opción 1: Notebooks Jupyter**
1. Adquisición de datos:
   ```bash
   jupyter notebook notebooks/01_Fuentes.ipynb
   ```
2. Limpieza y EDA:
   ```bash
   jupyter notebook notebooks/02_LimpiezaEDA.ipynb
   ```
3. Entrenamiento y evaluación:
   ```bash
   jupyter notebook notebooks/03_Entrenamiento_Evaluacion.ipynb
   ```

**Opción 2: Scripts Python**
   ```bash
   cd src
   python data_processing.py
   python training.py
   python evaluation.py
   ```

**Opción 3: App Streamlit**
   ```bash
   cd app_streamlit
   streamlit run app.py
   ```

### English

**Option 1: Jupyter Notebooks**
1. Data acquisition:
   ```bash
   jupyter notebook notebooks/01_Fuentes.ipynb
   ```
2. Cleaning and EDA:
   ```bash
   jupyter notebook notebooks/02_LimpiezaEDA.ipynb
   ```
3. Training and evaluation:
   ```bash
   jupyter notebook notebooks/03_Entrenamiento_Evaluacion.ipynb
   ```

**Option 2: Python scripts**
   ```bash
   cd src
   python data_processing.py
   python training.py
   python evaluation.py
   ```

**Option 3: Streamlit app**
   ```bash
   cd app_streamlit
   streamlit run app.py
   ```

---

## 🔧 Requisitos / Requirements

### Tecnologías Principales / Main technologies

- **Python 3.9+**
- **Pandas** - Manipulación de datos / Data management
- **NumPy** - Operaciones numéricas / Numerical operations
- **Scikit-learn** - Modelos de ML / Machine Learning models
- **XGBoost / LightGBM** - Modelos avanzados de boosting / Boosting advanced models
- **Imbalanced-learn** - Manejo de clases desbalanceadas (SMOTE) / Managemend of unbalanced classes (SMOTE)
- **Streamlit** - Aplicación web interactiva / Interactive web app
- **Plotly / Matplotlib / Seaborn** - Visualizaciones / Visualizations

### Hardware Recomendado

- **RAM:** Mínimo 8GB (recomendado 16GB) / Minimum 8GB (recommended 16GB)
- **CPU:** Procesador multi-core / multi-core processor
- **GPU:** Opcional (acelera XGBoost/LightGBM) / optional (accelerates XGBoost/LightGBM)

---

## 🚀 Instalación / Installment

### 1. Clonar el Repositorio / Clone the Repository

```bash
git clone https://github.com/tu-usuario/proyecto-deteccion-fraude.git
cd proyecto-deteccion-fraude
```

### 2. Crear Entorno Virtual / Create the Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar Dependencias / Install Dependencies

```bash
pip install -r app_streamlit/requirements.txt
```

### 4. Instalar Jupyter (opcional, para notebooks) / Install Jupyter (optional, for notebooks)

```bash
pip install jupyter notebook
```

---

## 💻 Uso / Usage

### Opción 1: Notebooks Jupyter / Option 1: Jupyter Notebooks

#### Paso 1: Adquisición de Datos / Data Acquisition

```bash
jupyter notebook notebooks/01_Fuentes.ipynb
```

- Descarga/carga del dataset - Dataset download
- Exploración inicial - Initial exploration
- Guardado en `data/raw/` - Saved in `data/raw/`

#### Paso 2: Limpieza y EDA / Step 2: Data Cleaning and EDA

```bash
jupyter notebook notebooks/02_LimpiezaEDA.ipynb
```

- Limpieza de datos (duplicados, nulos, outliers) /  Data cleaning (duplicates, nulls, outliers)
- Análisis exploratorio completo / Complete exploratory analysis
- Feature engineering
- Guardado en `data/processed/` / Saved in `data/processed/`

#### Paso 3: Entrenamiento y Evaluación / Step 3: Training and Evaluation

```bash
jupyter notebook notebooks/03_Entrenamiento_Evaluacion.ipynb
```

- Entrenamiento de múltiples modelos / Training of multiple models
- Optimización con GridSearch / GridSearch Optimization
- Evaluación y comparación / Evaluation and Comparison
- Guardado de modelos en `models/`/ Saved in `models/`

### Opción 2: Scripts Python / Option 2: Scripts Python

#### Procesar Datos / Data Processing

```bash
cd src
python data_processing.py
```

#### Entrenar Modelos / Models Training

```bash
python training.py
```

#### Evaluar Modelos / Models Evaluation

```bash
python evaluation.py
```

### Opción 3: Aplicación Streamlit / Option 3: Streamlit App

```bash
cd app_streamlit
streamlit run app.py
```

La aplicación se abrirá en `http://localhost:8501`
The app will open in `http://localhost:8501`

---


## 📊 Datos / Data

- **Origen:** API de Kaggle. El enlace al dataset es el siguiente: https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets
- **Descarga de datos:** [Accede a los archivos aquí (Google Drive)](https://drive.google.com/drive/folders/1g6eoO5BrIdIDlKlp8-S7hBOV5PmrcKuG?usp=sharing)

| Variable           | Tipo         | Descripción                                 |
|--------------------|--------------|---------------------------------------------|
| `step`             | int          | Unidad de tiempo (hora)                     |
| `type`             | categórica   | Tipo de transacción (PAYMENT, TRANSFER...)  |
| `amount`           | float        | Monto de la transacción                     |
| `nameOrig`         | string       | Cliente que inicia la transacción           |
| `oldbalanceOrg`    | float        | Balance inicial del origen                  |
| `newbalanceOrig`   | float        | Balance final del origen                    |
| `nameDest`         | string       | Cliente receptor                            |
| `oldbalanceDest`   | float        | Balance inicial del destino                 |
| `newbalanceDest`   | float        | Balance final del destino                   |
| `isFraud`          | int          | 1 si es fraude, 0 si no (TARGET)            |

- **No Fraude:** 91%  | **Fraude:** 9%  | **Ratio:** 10:1
- **Estrategia:** SMOTE para balanceo de clases.

- **Source:** [Kaggle's API]
- **Download data:** [Access the files here (Google Drive)](https://drive.google.com/drive/folders/1g6eoO5BrIdIDlKlp8-S7hBOV5PmrcKuG?usp=sharing)

| Variable           | Type         | Description                                 |
|--------------------|--------------|---------------------------------------------|
| `step`             | int          | Time unit (hour)                            |
| `type`             | categorical  | Transaction type (PAYMENT, TRANSFER...)     |
| `amount`           | float        | Transaction amount                          |
| `nameOrig`         | string       | Originating customer                        |
| `oldbalanceOrg`    | float        | Initial origin balance                      |
| `newbalanceOrig`   | float        | Final origin balance                        |
| `nameDest`         | string       | Destination customer                        |
| `oldbalanceDest`   | float        | Initial destination balance                 |
| `newbalanceDest`   | float        | Final destination balance                   |
| `isFraud`          | int          | 1 if fraud, 0 if not (TARGET)               |

- **No Fraud:** 91%  | **Fraud:** 9%  | **Ratio:** 10:1
- **Strategy:** SMOTE for class balancing.

---

## 🔬 Metodología / Methodology

### 1. Exploración y Limpieza de Datos (EDA) / Exploration and Data Cleaning (EDA)

- ✅ Análisis de valores nulos y duplicados / Analysis of nulls and duplicated values
- ✅ Detección y tratamiento de outliers / Detection and management of outliers
- ✅ Análisis de distribuciones / Distributions analysis
- ✅ Estudio de correlaciones / Correlations analysis
- ✅ Análisis del desbalance de clases / Classes' imbalance analysis

### 2. Feature Engineering

- Creación de nuevas features / New features creation:
  - `balance_diff_orig`: Diferencia de balances en origen / Difference in origin balances
  - `balance_diff_dest`: Diferencia de balances en destino / Difference in destination balances
  - `amount_to_balance_ratio`: Ratio monto/balance / Ratio amount/balance

- Codificación de variables categóricas / Categorical variables codification (Label Encoding / One-Hot)
- Normalización de variables numéricas / Normalization of numerical variables

### 3. Modelado / Modeling

#### Modelos Supervisados Entrenados / Trained Supervised Models

1. **Logistic Regression** (Baseline)
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **Gradient Boosting Classifier**
5. **XGBoost Classifier** ⭐
6. **LightGBM Classifier**
7. **Support Vector Machine (SVM)**
8. **K-Nearest Neighbors (KNN)**

#### Modelo No Supervisado / Unsupervised Model

- **KMeans Clustering** - Detección de anomalías / Anomalies' detection

#### Técnicas Aplicadas / Applied Techniques

- ✅ **Pipeline de Scikit-learn** para preprocesamiento / for preprocessing
- ✅ **GridSearchCV** para optimización de hiperparámetros / for optimising hyperparameters
- ✅ **Cross-Validation** (5-fold)
- ✅ **SMOTE** para balanceo de clases / for classes' balancing
- ✅ **Estratificación** en/in train-test split

### 4. Evaluación / Evaluation

#### Métricas Principales / Main Scores

- **Precision:** 93%
- **Recall:** 80%
- **F1-Score:** 86%
- **ROC-AUC:** 0.6739 ⭐

#### Justificación de Métricas / Metrics' justification

En problemas de fraude, el **Recall** es crítico (detectar todos los fraudes posibles), pero también necesitamos buen **Precision** para no generar demasiados falsos positivos. Por eso usamos **ROC-AUC** como métrica principal de comparación.

In fraud detection, **Recall** is critical (detecting all possible frauds), but we also need good **Precision** to avoid generating too many false positives. That's why we use **ROC-AUC** as our primary comparison metric.

---

## 🏆 Resultados / Results

### Mejor Modelo / Best Model

🥇 **[Nombre del Modelo - ej. XGBoost Classifier]**

#### Hiperparámetros Óptimos

```python
{
    'n_estimators': 200,
    'max_depth': 7,
    'learning_rate': 0.1,
    'subsample': 0.8,
    # ... otros parámetros
}
```

#### Métricas en Test Set

| Métrica | Valor |
|---------|-------|
| Precision | 93% |
| Recall | 80% |
| F1-Score | 86% |
| ROC-AUC | 0.6739 |

### Comparación de Modelos según las métricas Accuracy, Precisión, Recall, F1-Score y ROC-AUC/ Models' comparison according to the scores Precision, Recall, F1-Score and ROC-AUC.

| Modelo/Model        | 
|---------------------|
| Logistic Regression | 
| Random Forest       | 
| **XGBoost**         |
| LightGBM            | 
| Gradient Boosting   | 

### Español
- **Mejor modelo:** XGBoost / LightGBM (ROC-AUC >0.90)
- **Principales features:** amount, oldbalanceOrg, newbalanceOrig
- **Recall alto, pocos falsos positivos**
- **KMeans** respalda los patrones detectados

### English
- **Best model:** XGBoost / LightGBM (ROC-AUC >0.90)
- **Top features:** amount, oldbalanceOrg, newbalanceOrig
- **High recall, few false positives**
- **KMeans** supports detected patterns

### Feature Importance

Top 3 features más importantes / Top 3 most important features:

1. `amount` - Monto de la transacción / Transaction's amount
2. `oldbalanceOrg` - Balance anterior origen / previous balance of origin
3. `newbalanceOrig` - Nuevo balance origen / new balance of origin
   

### Visualizaciones / Visualizations

![Confusion Matrix](docs/confusion_matrix.png)
![ROC Curve](docs/roc_curve.png)
![Feature Importance](docs/feature_importance.png)

---

## 🌐 Aplicación Web / Web App

### Funcionalidades / Functionalities:

La aplicación Streamlit incluye / The Streamlit app includes:

1. **🏠 Inicio** / **Home**
   - Descripción del proyecto / Project Description
   - Métricas principales / Main scores
   - Información general / General information

2. **🔮 Predicción Individual** / **🔮 Individual Prediction**
   - Formulario para introducir datos de una transacción / Form to introduce transaction data
   - Predicción en tiempo real / Real-time predictions
   - Probabilidades de fraude / Fraud chances
   - Recomendaciones de acción / Recommendations

3. **📊 Análisis por Lotes** / **📊 Batch Analysis**
   - Carga de archivos CSV / Downloading of CSV files 
   - Análisis masivo de transacciones / Massive transactions analysis
   - Visualizaciones interactivas / Interactive visualizations
   - Descarga de resultados / Results downloads

4. **📈 Métricas del Modelo** / **📈 Model scores**
   - Información detallada del modelo / Detailed model information
   - Métricas de rendimiento / Performance scores
   - Gráficos de evaluación / Evaluation graphics


---

## 🎓 Aprendizajes y Conclusiones / Learnings and Conclusions

### Hallazgos Principales / Chief Findings

1. ✅ **El modelo logra identificar patrones claros de fraude** con alta precisión / The model can identify clear fraud patterns
2. ✅ Las variables de **balance y monto** son las más relevantes / The variables **balance and amount** are the most relevant ones
3. ✅ El **balanceo de clases con SMOTE** mejora significativamente el Recall / The **SMOTE classes balancing** significantly improves the Recall score
4. ✅ Los modelos de **boosting superan a los modelos lineales** en este problema / The **boosting models surpass the linear models** in this problem


### Limitaciones / Limitations

- El modelo depende de la calidad y completitud de los datos / The model depends on the quality and completeness of the data
- Requiere reentrenamiento periódico con nuevos datos / The model requires periodic training with new data
- Puede haber sesgos en los datos históricos / The historical data can be biased

### Mejoras Futuras / Future Improvements

- 🔄 Incorporar más features temporales / Incorporate more time features
- 🔄 Implementar modelos de Deep Learning (LSTM, Autoencoders) / Implementing Deep Learning models
- 🔄 Despliegue en producción con API REST / Launching in production with API REST
- 🔄 Sistema de monitoreo en tiempo real / Monitoring system in real time
- 🔄 Feedback loop para mejora continua / Feedback loop for continuous improvements

---

## 📚 Referencias / References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Dataset utilizado - Kaggle](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets))

---

## 👩‍💻 Autora / Author

**Sara Gil Martín-Serrano**

- 📧 Email: saragms217@gmail.com
- 💼 LinkedIn: https://www.linkedin.com/in/sara-gil-martín-serrano-84742310b/
- 🐙 GitHub: https://github.com/SaraGMS

---

## 📄 Licencia / License

Este proyecto fue desarrollado como parte del Bootcamp de Data Science en [The Bridge] (2025). This project was developed as part of the Data Science bootcamp in [The Bridge] (2025).
También incluye la licencia MIT. It also includes the MIT license.



---

## 🙏 Agradecimientos / Acknowledgements

- The Bridge - Formación y acompañamiento / Training and support
- Profesores - Apoyo y revisiones / Teachers - Support and reviews
- Kaggle Community - Datasets y recursos / Datasets and resources

---

<div align="center">

**⭐ Si este proyecto te resulta útil, considera darle una estrella / If you find this project useful, please star it ⭐**

Desarrollado con ❤️ y ☕ / Made with ❤️ and ☕

</div>
   ```bash
   pip install jupyter notebook
   ```
