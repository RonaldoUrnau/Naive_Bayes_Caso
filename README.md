# Caso 1 · Cauce: Clasificación Automática de Mensajes con Naive Bayes

Este repositorio contiene la implementación y análisis del **Caso 1: Cauce**, enfocado en la clasificación automática de mensajes de clientes usando **Naive Bayes Multinomial** implementado desde cero y optimizado con Pipelines de Scikit-Learn.

## 📁 Estructura del Proyecto

```text
Naive_Bayes_Caso/
├── datos/
│   ├── mensajes_historicos.csv   # Dataset histórico para entrenamiento y validación
│   └── mensajes_nuevos.csv       # Mensajes a clasificar
├── src/
│   ├── __init__.py
│   ├── limpieza.py              # Funciones de preprocesamiento y limpieza de texto
│   ├── naive_bayes.py           # Implementación NaiveBayesMultinomial desde cero (NumPy)
│   └── derivacion.py            # Lógica de reglas de negocio y derivación
├── tests/
│   ├── test_limpieza.py
│   ├── test_naive_bayes.py
│   └── test_derivacion.py
├── caso1_cauce.ipynb            # Notebook principal con el desarrollo paso a paso
├── conftest.py
├── pytest.ini
├── requirements.txt
└── .gitignore
```

## ⚙️ Requisitos e Instalación

1. Crear y activar un entorno virtual:
   ```bash
   python -m venv venv
   # En Windows:
   venv\Scripts\activate
   ```

2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## 🧪 Ejecución de Pruebas

Para ejecutar las pruebas automáticas con pytest:
```bash
python -m pytest -q
```
