"""Regla de derivación de mensajes y sus indicadores de negocio.

Completá las dos funciones. Solo NumPy. Verificá tu trabajo con:  python -m pytest -q tests/test_derivacion.py
"""
import numpy as np

REVISION = "revision_humana"


def derivar(proba, clases, umbral_fraude, umbral_confianza):
    """Decide el destino de cada mensaje a partir de las probabilidades del modelo.

    proba : array (n_mensajes, n_clases) con P(clase | mensaje); columnas en el orden de `clases`.
    clases : lista o array con los nombres de las clases (incluye "fraude").

    Regla, en este orden de prioridad:
    1. Si P(fraude) >= umbral_fraude, el destino es "fraude" (aunque no sea la clase más probable).
    2. Si no, y la probabilidad máxima >= umbral_confianza, el destino es la clase más probable.
    3. En cualquier otro caso, el destino es REVISION ("revision_humana").

    Devuelve un array de n_mensajes strings.
    """
    proba = np.asarray(proba)
    clases = np.asarray(clases)
    n = len(proba)

    idx_fraude = np.where(clases == "fraude")[0][0]
    prob_fraude = proba[:, idx_fraude]
    max_prob = np.max(proba, axis=1)
    best_class = clases[np.argmax(proba, axis=1)]

    destino = np.full(n, REVISION, dtype=object)

    # 2. Si la probabilidad máxima >= umbral_confianza
    mask_confianza = max_prob >= umbral_confianza
    destino[mask_confianza] = best_class[mask_confianza]

    # 1. Si P(fraude) >= umbral_fraude (tiene prioridad y sobreescribe)
    mask_fraude = prob_fraude >= umbral_fraude
    destino[mask_fraude] = "fraude"

    return destino.astype(str)


def evaluar_derivacion(y_real, destino):
    """Calcula los indicadores del negocio. Devuelve un dict con tres claves:

    - "cobertura": proporción de mensajes derivados automáticamente (destino distinto de REVISION).
    - "exactitud_automatica": entre los derivados automáticamente, proporción con destino == y_real.
    - "fraude_perdido": entre los mensajes cuyo y_real es "fraude", proporción derivada automáticamente
      a un equipo que no es "fraude". Los enviados a revisión humana no cuentan como perdidos.

    Los tres valores son proporciones entre 0 y 1 (float).
    """
    y_real = np.asarray(y_real)
    destino = np.asarray(destino)

    mask_auto = destino != REVISION
    cobertura = float(np.mean(mask_auto))

    if np.any(mask_auto):
        exactitud_automatica = float(np.mean(destino[mask_auto] == y_real[mask_auto]))
    else:
        exactitud_automatica = 0.0

    mask_real_fraude = y_real == "fraude"
    if np.any(mask_real_fraude):
        perdido = (destino[mask_real_fraude] != "fraude") & (destino[mask_real_fraude] != REVISION)
        fraude_perdido = float(np.mean(perdido))
    else:
        fraude_perdido = 0.0

    return {
        "cobertura": cobertura,
        "exactitud_automatica": exactitud_automatica,
        "fraude_perdido": fraude_perdido
    }
