# Clasificador de Frutas con Deep Learning y Transfer Learning 🍎🍌🍒

Este proyecto implementa un sistema de clasificación de frutas usando redes neuronales convolucionales (CNN) y técnicas de aprendizaje por transferencia (transfer learning) sobre el dataset Fruit360. Incluye herramientas para descarga, preprocesamiento, entrenamiento, visualización de resultados y recuperación de históricos.

## Estructura del Proyecto

- `data_raw/` — Datasets originales y variantes (Fruit360, versiones multi, meta, etc.)
- `transferLearning/` — Código principal para transferencia de aprendizaje:
  - `transferLearning.py` — Entrenamiento y fine-tuning de modelos preentrenados (EfficientNet, MobileNetV2, ResNet50)
  - `preprocess_data.py` — Preprocesamiento seguro de datos (sin data leakage)
  - `descarga_cifar.py` — Descarga automatizada del dataset desde Kaggle
  - `visualize_training.py` — Visualización y reporte de métricas de entrenamiento
  - `recuperar_historial.py` — Recupera y visualiza históricos de entrenamiento
  - `check_dataset_structure.py` — Verifica la estructura de los datos
  - `class_names.json` — Lista de clases del dataset
  - `training_results/` — Reportes, gráficas y resultados de entrenamiento
- `red_neuronal.py` — Ejemplo de CNN optimizada desde cero
- `requirements.txt` — Dependencias del proyecto
- `README.md` — Este archivo

## Instalación

1. Clona el repositorio y entra al directorio:
   ```bash
   git clone <url-del-repo>
   cd webApp
   ```
2. (Opcional) Crea y activa un entorno virtual:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   # venv\Scripts\activate  # Windows
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Descarga el dataset Fruit360 ejecutando:
   ```bash
   cd transferLearning
   python descarga_cifar.py
   ```

## Uso rápido

- Entrenamiento con transferencia:
  ```bash
  python transferLearning/transferLearning.py
  ```
- Visualización de resultados:
  ```bash
  python transferLearning/recuperar_historial.py
  ```

## Resultados de ejemplo

- Reportes y gráficas se guardan en `transferLearning/training_results/`.
- Ejemplo de accuracy en test: **94%** usando MobileNetV2.

## Créditos y Licencia

- Dataset: [Fruit360 en Kaggle](https://www.kaggle.com/datasets/moltean/fruits)
- Autor: Elmer Alexander Martinez Guillen
- Licencia: MIT
