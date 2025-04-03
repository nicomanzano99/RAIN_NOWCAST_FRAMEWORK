# TFM - Nowcasting Hiperlocal de Lluvia

Este proyecto forma parte de mi Trabajo Fin de Máster y tiene como objetivo desarrollar un sistema de **nowcasting de lluvia a nivel hiperlocal**, basado en series temporales y redes neuronales. El código está organizado en tres fases principales: preprocesado de datos, entrenamiento del modelo y predicción.

---

## Estructura del proyecto

```plaintext
.
├── 1_Data_Processing/
│ ├── processing_class.py # Generación de secuencias y guardado de datasets
│ ├── balancing_class.py # Balanceo de clases según IDs del dataset
│ └── scale_data.py # Escalado de variables
│
├── 2_Model_Training/
│ ├── data_generator.py # Generador de datos para entrenamiento
│ ├── architectures.py # Definición de arquitecturas de redes
│ └── train.py # Entrenamiento usando W&B y configuración dinámica
│
├── 3_Test/
│ └── test.py # Carga del modelo y predicciones sobre nuevos datos
```


## Flujo del proyecto

1. **Preprocesado (`1_Data_Processing`)**  
   - Se crean secuencias temporales a partir de datos meteorológicos y se guardan en un `.xlsx` con su `dataset_id`.
   - Se realiza el balanceo de clases tomando los `ids` del dataset base y generando nuevos archivos CSV según el método elegido.
   - Se escalan las variables según el tipo de normalización definida.

2. **Entrenamiento (`2_Model_Training`)**  
   - Con un `dataset_id` y el método de balanceo, se genera el conjunto de entrenamiento.
   - Se define la arquitectura desde `architectures.py`.
   - Se entrena el modelo con seguimiento a través de [Weights & Biases](https://wandb.ai/).

3. **Predicción (`3_Test`)**  
   - Se carga el modelo entrenado y se hacen predicciones sobre nuevas secuencias.

---
