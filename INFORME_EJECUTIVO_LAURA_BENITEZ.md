# 📄 INFORME EJECUTIVO · CAUCE BILLETERA VIRTUAL
**Automatización Inteligente de la Mesa de Entrada y Detección Temprana de Fraude**

**Para:** Laura Benítez (Dirección de Operaciones y Atención al Cliente)  
**De:** Equipo de Ciencia de Datos & Ciberseguridad Financiera  
**Fecha:** Octubre 2026  
**Alcance:** Flujo Operativo de 6.500 Mensajes Diarios de Clientes  
**Entregables Técnicos:** `caso1_cauce.ipynb`, `predicciones_caso1.csv` (2.040 filas), `informe_ejecutivo.pdf`  

---

## 1. Resumen Ejecutivo y Diagnóstico del Problema

Cauce recibe diariamente un volumen de **6.500 mensajes de clientes** distribuidos a través de cuatro canales principales (App Chat, WhatsApp, Email y Web). Actualmente, la clasificación y asignación hacia los 5 equipos resolutores (`transferencias`, `tarjeta`, `cuenta_acceso`, `prestamos_inversiones` y `fraude`) es realizada manualmente por un equipo de **9 personas a tiempo completo**.

Este esquema operativo presenta dos fallas críticas para la sustentabilidad del negocio:
1. **Ineficiencia y Sobrecarga Operativa:** Los operadores cometen errores en 1 de cada 6 mensajes (**tasa de error del 16.46%**, equivalente a más de **1.070 mensajes diarios mal derivados**), generando cuellos de botella, fricción con el usuario y demoras en la resolución.
2. **Vulnerabilidad Crítica ante el Fraude:** La mesa humana desvía erróneamente el **33.52% de los casos reales de fraude** hacia equipos sin protocolo urgente (principalmente plásticos y cuentas). En delitos financieros, las primeras horas son determinantes; cuando la alerta llega al área correcta, los fondos ya han sido sustraídos irreversiblemente.

---

## 2. La Solución Desarrollada: Naive Bayes con Derivación Asimétrica

Diseñamos e implementamos un sistema de procesamiento de lenguaje natural y clasificación probabilística basado en **Naive Bayes Multinomial** con preprocesamiento avanzado (normalización Unicode, enmascaramiento de montos/números y extracción de unigramas y bigramas).

Para gobernar el modelo en producción, diseñamos una **Regla de Derivación de Negocio Asimétrica de Tres Salidas**:
* **Salida 1 · Prioridad Absoluta de Fraude:** Si $P(\text{fraude}) \ge 1.0\%$, el mensaje es derivado inmediatamente al equipo de **Fraude**, aun cuando otra clase sea la más probable.
* **Salida 2 · Derivación Automática Segura:** Si la probabilidad máxima $\ge 99.9\%$, el mensaje es derivado de forma instantánea al equipo resolutor predicho.
* **Salida 3 · Filtro de Seguridad a Revisión Humana:** Si el modelo detecta ambigüedad o baja confianza, el mensaje es canalizado a **Revisión Humana**.

---

## 3. Comparativa de Estrategias y los 5 Indicadores de Aceptación

Evaluación rigurosa sobre el conjunto de prueba independiente (**Test Set** de 1.586 mensajes):

| Indicador de Aceptación | 1. Mesa de Entrada (Actual) | 2. KNN Coseno | 3. Naive Bayes (Propuesto) | Impacto para Cauce |
| :--- | :---: | :---: | :---: | :--- |
| **1. Cobertura Automática** | 100.00% (Manual) | 98.30% | **95.96%** | Automatiza el 96% de la demanda entrante |
| **2. Exactitud de Derivación Automática** | 83.54% | 84.35% | **96.12%** | **+12.6 puntos de mejora** frente al equipo humano |
| **3. Tasa de Fraude Perdido** | **33.52%** | 1.14% | **2.84%** | **Reducción del 91.5% en fraude desviado** |
| **4. Carga a Revisión Humana** | 0.00% | 1.70% | **4.04%** | Solo ~4 de cada 100 mensajes requieren revisión |
| **5. F1 Macro de la Estrategia** | 0.8237 | 0.8318 | **0.9564** | Clasificación equilibrada y de alta fidelidad |

---

## 4. Impacto Operativo y Económico en un Día de 6.500 Mensajes

La proyección para una jornada laboral típica evidencia un cambio estructural en las operaciones de Cauce:

* **Derivación Inmediata:** **6.237 mensajes diarios** son canalizados en milisegundos sin intervención humana.
* **Reasignación de Personal:** El volumen para revisión manual se desploma de 6.500 a solo **263 mensajes por día**. Esta carga puede ser absorbida por **1 sola persona**, permitiendo **redistribuir a 8 colaboradores** hacia tareas de alta empatía, soporte técnico complejo o investigación proactiva de estafas.
* **Reducción de Retrabajo:** Los desvíos erróneos caen de 1.070 a 242 diarios (una **disminución del 77.4%** en fricción operativa).
* **Protección Financiera Activa:** De 242 casos de fraude desviados erróneamente por la mesa humana, con nuestro sistema la cifra baja a **solo 20 casos diarios** (rescatando a tiempo más de 220 clientes estafados por día).

---

## 5. Justificación Técnica: Naive Bayes vs. KNN en Producción

El benchmark de tiempos de respuesta demostró que KNN es inviable para operaciones en tiempo real:
* **Naive Bayes ($O(V \cdot K)$):** Tiempo de inferencia **constante de ~55 ms** para lotes de 1.000 mensajes. Huella de memoria mínima en servidores.
* **KNN ($O(N \cdot V)$):** Tiempo de inferencia **creciente de 405 a 676 ms** (más de 12 veces más lento) y necesidad de mantener en memoria todo el histórico.

---

## 6. Recomendaciones y Hoja de Ruta para Despliegue

1. **Puesta en Marcha en Modo Shadow / Asistido (Semana 1-2):** El modelo sugiere el destino al operador humano para validar la aceptación de los umbrales en vivo.
2. **Despliegue Productivo Automatizado (Semana 3):** Activación de la regla de derivación automática directa para el 96% del tráfico.
3. **Monitoreo Continuo de Deriva Léxica (Drift):** Implementar alertas tempranas de nuevas modalidades delictivas y nuevos lanzamientos comerciales (ej. seguros) para gatillar reentrenamientos quincenales.
