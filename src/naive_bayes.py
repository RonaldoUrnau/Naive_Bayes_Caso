"""Naive Bayes multinomial implementado desde cero.

Reglas:
- Los cálculos se hacen solo con NumPy. No se puede importar scipy ni usar sklearn.naive_bayes.
- La clase hereda de ClassifierMixin y BaseEstimator solo para poder usarse dentro de un
  Pipeline y de GridSearchCV de scikit-learn (eso le da get_params, set_params y score).
- Todo se calcula en escala logarítmica: multiplicar cientos de probabilidades chicas da 0
  en punto flotante (underflow).
- X puede ser un array de NumPy o una matriz dispersa de SciPy (la que devuelve CountVectorizer).
  Las operaciones X[filas], X.sum(axis=0) y X @ M funcionan con ambas; usá np.asarray(...) para
  convertir los resultados a arrays comunes.

Verificá tu trabajo con:  python -m pytest -q tests/test_naive_bayes.py
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def log_suma_exp(a, axis=1):
    """Calcula log(sum(exp(a))) a lo largo de `axis` sin overflow ni underflow.

    Truco: si m es el máximo de a, entonces log(sum(exp(a))) = m + log(sum(exp(a - m))).

    Ejemplo: log_suma_exp(np.array([[-1000.0, -1000.0]])) -> array([-999.3069...])
    """
    m = np.max(a, axis=axis, keepdims=True)
    m_safe = np.where(np.isneginf(m), 0.0, m)
    exp_diff = np.exp(a - m_safe)
    sum_exp = np.sum(exp_diff, axis=axis, keepdims=True)
    res = np.where(np.isneginf(m), -np.inf, m + np.log(sum_exp))
    return np.squeeze(res, axis=axis)


class NaiveBayesMultinomial(ClassifierMixin, BaseEstimator):
    """Naive Bayes multinomial con suavizado aditivo.

    Parámetros
    ----------
    alpha : float, por defecto 1.0
        Suavizado aditivo. Con alpha=1 es el suavizado de Laplace.

    Atributos que tiene que crear fit (mismos nombres que en scikit-learn)
    -------------------------------------------------------------------
    classes_ : array (n_clases,)
        Clases ordenadas, como devuelve np.unique(y).
    class_count_ : array (n_clases,)
        Cantidad de documentos de cada clase.
    feature_count_ : array (n_clases, n_terminos)
        Suma de los conteos de cada término en los documentos de cada clase.
    class_log_prior_ : array (n_clases,)
        log P(c) = log(documentos de la clase c / documentos totales).
    feature_log_prob_ : array (n_clases, n_terminos)
        log P(t | c) = log((conteo de t en c + alpha) / (total de términos en c + alpha * n_terminos)).
    """

    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)

    def fit(self, X, y):
        """Estima las probabilidades a partir de la matriz de conteos X y las etiquetas y. Devuelve self."""
        y = np.asarray(y)
        classes, class_count = np.unique(y, return_counts=True)
        self.classes_ = classes
        self.class_count_ = class_count

        n_classes = len(classes)
        n_samples = X.shape[0]
        n_features = X.shape[1]

        self.class_log_prior_ = np.log(class_count / n_samples)

        feature_count = np.empty((n_classes, n_features), dtype=np.float64)
        for i, c in enumerate(classes):
            mask = (y == c)
            feature_count[i] = np.asarray(X[mask].sum(axis=0)).ravel()
        self.feature_count_ = feature_count

        smoothed_fc = feature_count + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1, keepdims=True)
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc)

        return self

    def _log_verosimilitud_conjunta(self, X):
        """Devuelve log P(c) + sum_t x_t · log P(t | c) para cada documento y clase.

        Forma del resultado: (n_documentos, n_clases). Se resuelve con un solo producto matricial.
        """
        return np.asarray(X @ self.feature_log_prob_.T) + self.class_log_prior_

    def predict_log_proba(self, X):
        """log P(c | documento): la log-verosimilitud conjunta normalizada con log_suma_exp."""
        jll = self._log_verosimilitud_conjunta(X)
        log_norm = log_suma_exp(jll, axis=1)
        return jll - log_norm[:, np.newaxis]

    def predict_proba(self, X):
        """P(c | documento). Cada fila suma 1. Las columnas siguen el orden de classes_."""
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        """La clase con mayor probabilidad a posteriori para cada documento."""
        jll = self._log_verosimilitud_conjunta(X)
        return self.classes_[np.argmax(jll, axis=1)]


def terminos_mas_indicativos(modelo, vocabulario, n=10):
    """Devuelve {clase: [n términos]} con los términos que más empujan hacia cada clase.

    Para la clase c, el puntaje de un término t es
        log P(t | c) - promedio de log P(t | c') en las demás clases c'.
    Se devuelven los n términos de mayor puntaje, de mayor a menor.

    `vocabulario` es la lista de términos en el orden de las columnas de X
    (por ejemplo, CountVectorizer.get_feature_names_out()).
    """
    vocab = np.asarray(vocabulario)
    k_clases = len(modelo.classes_)
    feature_log_prob = modelo.feature_log_prob_

    indicativos = {}
    if k_clases > 1:
        sum_total = feature_log_prob.sum(axis=0)
        for i, c in enumerate(modelo.classes_):
            sum_otras = sum_total - feature_log_prob[i]
            promedio_otras = sum_otras / (k_clases - 1)
            puntaje = feature_log_prob[i] - promedio_otras
            mejores_idx = np.argsort(puntaje)[::-1][:n]
            indicativos[c] = vocab[mejores_idx].tolist()
    else:
        for i, c in enumerate(modelo.classes_):
            mejores_idx = np.argsort(feature_log_prob[i])[::-1][:n]
            indicativos[c] = vocab[mejores_idx].tolist()

    return indicativos
