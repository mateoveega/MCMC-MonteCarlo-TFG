import numpy as np
import copy
from scipy.stats import multivariate_normal

def box_muller(D, num_muestras, semilla=1):
    """
    Genera una matriz de muestras usando el método polar de Box-Muller.
    
    Parámetros:
    - D (int): Dimensión del espacio (número de columnas).
    - num_muestras (int): Número total de muestras a generar (número de filas).
    - semilla (int): Semilla para la replicabilidad de los resultados.
    
    Retorna:
    - matriz_muestras (np.ndarray): Matriz de tamaño (num_muestras, D) con muestras N(0, I_D).
    """
    generador = np.random.default_rng(semilla)
    total_muestras = D * num_muestras
    muestras = []

    while len(muestras) < total_muestras:
        x1 = 2 * generador.random() - 1
        x2 = 2 * generador.random() - 1

        radio = x1 ** 2 + x2 ** 2
        if 0 < radio < 1:
            factor = np.sqrt(-2 * np.log(radio) / radio)
            y1 = x1 * factor
            y2 = x2 * factor
            
            muestras.extend([y1, y2])

    muestras_planas = np.array(muestras)[:total_muestras]
    matriz_muestras = muestras_planas.reshape((num_muestras, D))
    return matriz_muestras


def rechazo(pdf_objetivo, pdf_propuesta, muestreador_propuesta, k, num_muestras, semilla=1):
    """
    Aplica el algoritmo de muestreo por rechazo.
    
    Parámetros:
    - pdf_objetivo (callable): Función que evalúa la PDF de la distribución objetivo p(x).
    - pdf_propuesta (callable): Función que evalúa la PDF de la propuesta q(x).
    - muestreador_propuesta (callable): Función que genera 'num_muestras' muestras de la propuesta.
    - k (float o array): Constante de escalado tal que p(x) <= k * q(x) en todo el dominio.
    - num_muestras (int): Número total de candidatos a generar.
    - semilla (int): Semilla para la replicabilidad de los resultados.
    
    Retorna:
    - muestras_aceptadas (np.ndarray): Las muestras que han sido aceptadas.
    - tasa_aceptacion (float): La proporción de muestras aceptadas (0 a 1).
    """
    generador = np.random.default_rng(semilla)
    
    x = muestreador_propuesta(num_muestras)
    p_x = pdf_objetivo(x)
    q_x = pdf_propuesta(x)
    u = generador.uniform(0, k * q_x)
    
    mascara_aceptacion = (q_x > 1e-10) & (u <= p_x)
    muestras_aceptadas = x[mascara_aceptacion]
    tasa_aceptacion = len(muestras_aceptadas) / num_muestras
    return muestras_aceptadas, tasa_aceptacion


def random_walk_metropolis(pdf_objetivo, num_iteraciones, posicion_inicial, sigma_propuesta, semilla=1):
    """
    Algoritmo Random Walk Metropolis para cualquier distribución objetivo.
    
    Parámetros:
    - pdf_objetivo (callable): Función que evalúa la densidad de probabilidad p(x).
    - num_iteraciones (int): Número de pasos de la cadena.
    - posicion_inicial (array-like): Punto de inicio de la cadena.
    - sigma_propuesta (float): Desviación estándar del salto de propuesta.
    - semilla (int): Semilla para la replicabilidad de los resultados.
    
    Retorna:
    - np.ndarray: Matriz con las muestras de la cadena.
    - float: Tasa de aceptación.
    """
    generador = np.random.default_rng(semilla)

    muestras_aceptadas = []
    contador_aceptadas = 0
    
    posicion_actual = np.array(posicion_inicial)
    dimension = len(posicion_actual)
    p_actual = pdf_objetivo(posicion_actual)

    for i in range(num_iteraciones):
        ruido_propuesta = generador.normal(loc=0.0, scale=sigma_propuesta, size=dimension)
        z_primo = posicion_actual + ruido_propuesta
        p_primo = pdf_objetivo(z_primo)

        if p_actual == 0 and p_primo > 0:
            alfa = 1.0
        elif p_actual == 0 and p_primo == 0:
            alfa = 0.0
        elif p_actual > 0:
            alfa = min(1.0, p_primo / p_actual)
        else:
            alfa = 0.0

        u = generador.random()
        if u <= alfa:
            posicion_actual = z_primo
            p_actual = p_primo
            contador_aceptadas += 1
        muestras_aceptadas.append(posicion_actual.copy())

    tasa_aceptacion = contador_aceptadas / num_iteraciones
    return np.array(muestras_aceptadas), tasa_aceptacion


def metropolis_hastings_log(estado_inicial, log_pdf_objetivo, muestreador_propuesta, log_pdf_propuesta, iteraciones, semilla=1):
    """
    Algoritmo Metropolis-Hastings en espacio logarítmico para cualquier distribución objetivo.
    
    Parámetros:
    - estado_inicial: Punto de inicio de la cadena.
    - log_pdf_objetivo (callable): Función que evalúa el logaritmo de la densidad objetivo.
    - muestreador_propuesta (callable): Función que propone un nuevo estado dado el actual.
    - log_pdf_propuesta (callable): Función que evalúa el logaritmo de la densidad de propuesta.
    - iteraciones (int): Número de pasos de la cadena.
    - semilla (int): Semilla para la replicabilidad de los resultados.
    
    Retorna:
    - mejor_estado: El estado con mayor log-probabilidad encontrado.
    - historial (list): Historial de log-probabilidades en cada iteración.
    - historial_estados (list): Historial de estados visitados en cada iteración.
    """
    generador = np.random.default_rng(semilla)

    mejor_estado = copy.deepcopy(estado_inicial)
    estado_actual = copy.deepcopy(estado_inicial)
    
    log_objetivo_actual = log_pdf_objetivo(estado_inicial)
    mejor_log_objetivo = log_objetivo_actual
    
    historial = []
    historial_estados = []

    for i in range(iteraciones):
        estado_propuesto = muestreador_propuesta(estado_actual, generador)
        
        log_objetivo_propuesto = log_pdf_objetivo(estado_propuesto)
        
        log_propuesta_adelante = log_pdf_propuesta(estado_propuesto, estado_actual)
        log_propuesta_atras = log_pdf_propuesta(estado_actual, estado_propuesto)
        
        log_ratio = (log_objetivo_propuesto - log_objetivo_actual) + (log_propuesta_atras - log_propuesta_adelante)
        
        if log_ratio >= 0 or np.log(generador.random()) < log_ratio:
            estado_actual = estado_propuesto
            log_objetivo_actual = log_objetivo_propuesto
            
            if log_objetivo_actual > mejor_log_objetivo:
                mejor_log_objetivo = log_objetivo_actual
                mejor_estado = copy.deepcopy(estado_actual)
                
        historial.append(log_objetivo_actual)
        historial_estados.append(copy.deepcopy(estado_actual))
        
    return mejor_estado, historial, historial_estados

def gibbs(mu, sigma, num_muestras, posicion_inicial, semilla=1):
    rng = np.random.default_rng(semilla)
    muestras = []
    posicion_actual = np.array(posicion_inicial, dtype=float)

    mu_x, mu_y = mu[0], mu[1]
    var_x, cov_xy, var_y = sigma[0, 0], sigma[0, 1], sigma[1, 1]

    for i in range(num_muestras):
        mu_x_cond = mu_x + (cov_xy / var_y) * (posicion_actual[1] - mu_y)
        sigma_x_cond = np.sqrt(var_x - (cov_xy**2 / var_y))
        posicion_actual[0] = rng.normal(loc=mu_x_cond, scale=sigma_x_cond)

        mu_y_cond = mu_y + (cov_xy / var_x) * (posicion_actual[0] - mu_x)
        sigma_y_cond = np.sqrt(var_y - (cov_xy**2 / var_x))
        posicion_actual[1] = rng.normal(loc=mu_y_cond, scale=sigma_y_cond)

        muestras.append(posicion_actual.copy())

    return np.array(muestras)

    return np.array(samples)
    
def transicion_compuesta(transiciones_base, alfas):
    """
    Fábrica de transiciones compuestas: toma un vector de transiciones base y
    un vector de probabilidades y devuelve una única transición unificada.
    
    Parámetros:
    - transiciones_base (array-like): Vector de funciones de transición base.
    - alfas (array-like): Vector de probabilidades asociadas a cada transición base.
    
    Retorna:
    - transicion_unificada (callable): Función de transición que combina las transiciones base.
    """
    assert np.isclose(sum(alfas), 1.0), "¡Error! El vector de alfas debe sumar 1."

    def transicion_unificada(estado_actual, generador):
        transicion_elegida = generador.choice(transiciones_base, p=alfas)
        return transicion_elegida(estado_actual, generador)
    return transicion_unificada
