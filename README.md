# Caso 1 · Cauce: Clasificación Automática de Mensajes con Naive Bayes

Este repositorio contiene la solución integral, código fuente, pruebas automáticas e informe ejecutivo del **Caso 1: Cauce**, enfocado en la clasificación automatizada de mensajes de clientes bancarios mediante **Naive Bayes Multinomial** implementado desde cero en NumPy y optimizado con pipelines de Scikit-Learn.

## 📌 Diagnóstico del Negocio

Cauce recibe **6.500 mensajes diarios** derivados manualmente por un equipo de 9 personas.
* **Problema:** Tasa de error del 16.5% (~1.070 desvíos diarios) y **33.5% de fraude derivado a equipos equivocados**, retrasando la intervención preventiva.
* **Solución:** Implementación de Naive Bayes con **Regla de Derivación Asimétrica de 3 Salidas** (Prioridad de Fraude, Derivación Segura y Revisión Humana).

---

## 🏆 Resultados y los 5 Indicadores de Aceptación (Test Set)

| Indicador de Negocio | Mesa de Entrada (Línea Base) | KNN (Coseno) | Naive Bayes (Ganador) | Impacto para Cauce |
| :--- | :---: | :---: | :---: | :--- |
| **1. Cobertura Automática** | 100.00% (Manual) | 98.30% | **95.96%** | Automatiza el 96% del volumen entrante |
| **2. Exactitud Automática** | 83.54% | 84.35% | **96.12%** | **+12.6 puntos de mejora** frente al equipo humano |
| **3. Tasa de Fraude Perdido** | **33.52%** | 1.14% | **2.84%** | **Reducción del 91.5% en fraude desviado** |
| **4. Carga a Revisión Humana** | 0.00% | 1.70% | **4.04%** | Solo ~4 de cada 100 mensajes requieren revisión |
| **5. F1 Macro de la Estrategia** | 0.8237 | 0.8318 | **0.9564** | Clasificación equilibrada y consistente |

### 📈 Impacto Diario (6.500 Mensajes / Día)
* **6.237 mensajes** derivados de forma 100% automática e instantánea (~55 ms de inferencia).
* **263 mensajes** a revisión humana (gestión absorbible por **1 sola persona**, liberando a **8 colaboradores**).
* Errores de ruteo reducidos en un **77.4%** (de 1.070 a 242 diarios).
* Fraudes perdidos reducidos de **242 a solo 20 diarios** (>220 alertas salvadas por día).

---

## 📁 Estructura del Repositorio

```text
Naive_Bayes_Caso/
├── datos/
│   ├── mensajes_historicos.csv             # 8.200 mensajes de entrenamiento y validación
│   └── mensajes_nuevos.csv                 # 2.040 mensajes para predicción final
├── src/
│   ├── __init__.py
│   ├── limpieza.py                        # Preprocesamiento y normalización de texto
│   ├── naive_bayes.py                     # NaiveBayesMultinomial desde cero (NumPy)
│   └── derivacion.py                      # Reglas de negocio y evaluación
├── tests/
│   ├── test_limpieza.py                   # 15 tests unitarios
│   ├── test_naive_bayes.py                # 12 tests unitarios (paridad 100% sklearn)
│   └── test_derivacion.py                 # 4 tests unitarios de negocio
├── caso1_cauce.ipynb                       # Notebook completo con los Pasos 1 al 20 y reflexiones
├── predicciones_caso1.csv                  # Archivo entregable final (2.040 predicciones validadas)
├── INFORME_EJECUTIVO_LAURA_BENITEZ.md      # Informe ejecutivo para Dirección de Operaciones
├── informe_ejecutivo.pdf                   # Informe ejecutivo formal en PDF
├── conftest.py
├── pytest.ini
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Instalación y Ejecución

```bash
# 1. Crear y activar entorno virtual
python -m venv .venv
.venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar la suite completa de pruebas unitarias (31/31 pasando)
python -m pytest -q
```
