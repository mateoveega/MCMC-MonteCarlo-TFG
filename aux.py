import re
import unicodedata
import numpy as np
import math
import networkx as nx
from scipy.stats import norm, multivariate_normal

# Capítulo 1 ==================================================================

# rechazo ----------------------------------------------------------------------
def pdf_objetivo_rechazo(x):
    return 0.4 * norm.pdf(x, loc=-2, scale=1) + 0.6 * norm.pdf(x, loc=3, scale=1)

def pdf_envolvente_rechazo(x):
    return norm.pdf(x, loc=0.5, scale=3)
    
# ARS --------------------------------------------------------------------------
def pdf_objetivo_ars(x):
    return np.exp(-((x - 1) ** 2))

def ln_pdf_objetivo_ars(x):
    return np.log(pdf_objetivo_ars(x))

def ln_pdf_objetivo_prima_ars(x):
    return -2 * (x - 1)

def ln_pdf_envolvente_ars(x, nodos_x, nodos_y, nodos_dy):
    tangentes = []
    for i in range(len(nodos_x)):
        tangentes.append(nodos_y[i] + nodos_dy[i] * (x - nodos_x[i]))
    return np.min(tangentes, axis=0)

def funcion_compresion_ars(x, nodos_x, nodos_y):
    return np.interp(x, nodos_x, nodos_y, left=-np.inf, right=-np.inf)

# comparacion_prng --------------------------------------------------------------
def generador_lcg(n, semilla=1):
    a = 65539
    c = 0
    m = 2 ** 31
    x = semilla
    resultados = []
    for i in range(n * 3):
        x = (a * x + c) % m
        resultados.append(x / m)
    return np.array(resultados).reshape(n, 3)

# Capítulo 2 ==================================================================

# lim_gibbs -------------------------------------------------------------------

def target_pdf_lim_gibbs(position, mu, sigma):
    return multivariate_normal.pdf(position, mean=mu, cov=sigma)

# Capítulo 3 ==================================================================

# mensaje_encriptado ----------------------------------------------------------

def caracter_a_entero(c):
    """
    Mapea un carácter a su índice entero correspondiente.

    Parámetros:
    - c (str): Carácter a mapear. Debe ser una letra A-Z o un espacio ' '.

    Retorna:
    - (int): Índice entero correspondiente. A-Z se mapea a 0-25, y ' ' al índice 26.
    """
    if c == ' ': return 26
    return ord(c) - 65

def entero_a_caracter(i):
    """
    Mapea un índice entero a su carácter correspondiente.

    Parámetros:
    - i (int): Índice entero a mapear. 0-25 se mapea a A-Z, y 26 al espacio ' '.

    Retorna:
    - (str): Carácter correspondiente al índice.
    """
    if i == 26: return ' '
    return chr(i + 65)

def limpiar_texto_con_espacios(texto):
    """
    Limpia el texto convirtiendo a mayúsculas, eliminando tildes
    y dejando solo letras A-Z y espacios.

    Parámetros:
    - texto (str): Texto a limpiar.

    Retorna:
    - (str): Texto limpio con solo letras A-Z y espacios.
    """
    texto = texto.upper()
    texto = unicodedata.normalize('NFD', texto)
    texto = texto.encode('ascii', 'ignore').decode('utf-8')
    texto = re.sub(r'[\n\t]+', ' ', texto)
    texto = re.sub(r'[^A-Z ]', '', texto)
    texto = re.sub(r' +', ' ', texto)
    return texto.strip()

def obtener_matriz_bigramas_27(texto, tamano):
    """
    Calcula la matriz de probabilidades de bigramas de un texto.

    Parámetros:
    - texto (str): Texto del que se extraen los bigramas.
    - tamano (int): Tamaño del alfabeto (número de filas y columnas de la matriz).

    Retorna:
    - matriz (np.ndarray): Matriz de tamaño (tamano x tamano) con las probabilidades
      de transición entre caracteres consecutivos.
    """
    matriz = np.ones((tamano, tamano))
    for i in range(len(texto) - 1):
        idx1 = caracter_a_entero(texto[i])
        idx2 = caracter_a_entero(texto[i+1])
        matriz[idx1, idx2] += 1

    matriz = matriz / matriz.sum(axis=1, keepdims=True)
    return matriz

def transicion_q1_doble(estado_actual, generador):
    """
    Q1 (Exploración Doble): Intercambia dos caracteres cualesquiera de la permutación.
    Proporciona movilidad básica en todo el espacio de búsqueda.

    Parámetros:
    - estado_actual (np.ndarray): Permutación actual de la clave.
    - generador (np.random.Generator): Generador de números aleatorios.

    Retorna:
    - propuesta (np.ndarray): Nueva permutación con dos posiciones intercambiadas.
    """
    propuesta = estado_actual.copy()
    p1, p2 = generador.choice(len(propuesta), size=2, replace=False)
    propuesta[p1], propuesta[p2] = propuesta[p2], propuesta[p1]
    return propuesta

def transicion_q2_triple(estado_actual, generador):
    """
    Q2 (Exploración Triple): Realiza una rotación cíclica entre tres posiciones de la permutación.
    Proporciona una perturbación más compleja que el intercambio simple.

    Parámetros:
    - estado_actual (np.ndarray): Permutación actual de la clave.
    - generador (np.random.Generator): Generador de números aleatorios.

    Retorna:
    - propuesta (np.ndarray): Nueva permutación con tres posiciones rotadas cíclicamente.
    """
    propuesta = np.array(estado_actual).copy()
    idx = generador.choice(len(propuesta), size=3, replace=False)
    val0, val1, val2 = propuesta[idx]
    propuesta[idx[1]], propuesta[idx[2]], propuesta[idx[0]] = val0, val1, val2
    return propuesta

def log_pdf_objetivo_mensaje_encriptado(estado_actual, O, M_log_esp):
        return np.sum(O * M_log_esp[np.ix_(estado_actual, estado_actual)])

def log_pdf_propuesta_mensaje_encriptado(estado_propuesto, estado_actual):
    return 0.0

# nodo_cero -------------------------------------------------------------------

def log_pdf_objetivo_nodo_cero(estado_actual, matriz_distancias, indices_obs, velocidad_propagacion, valores_obs, desviacion):
    distancias = matriz_distancias[estado_actual, indices_obs]
    tiempos_teoricos = distancias * velocidad_propagacion
    error_total = np.sum((valores_obs - tiempos_teoricos)**2)

    return - error_total / (2 * desviacion**2)

def generador_guiado_nodo_cero(nodo_actual, rng, p, G):
    return rng.choice(list(G.nodes()), p=p)

def generador_random_walk_nodo_cero(nodo_actual, rng, G):
    vecinos = list(G.neighbors(nodo_actual))
    return rng.choice(vecinos)

def log_pdf_propuesta_nodo_cero(estado_propuesto, estado_actual, probabilidades_salto, G, alfa_guiado=0.15):
    p_guiado = probabilidades_salto[estado_propuesto]
    vecinos_actuales = list(G.neighbors(estado_actual))

    if estado_propuesto in vecinos_actuales:
        p_ca = 1.0 / len(vecinos_actuales)
    else:
        p_ca = 0.0

    p_total = (alfa_guiado * p_guiado) + ((1.0 - alfa_guiado) * p_ca)
    eps = 1e-12

    return np.log(p_total + eps)


def nodo_bajo_grado(G, semilla=1):
    """
    Selecciona aleatoriamente un nodo del 1% de los nodos con menor grado (menos conexiones).

    Parámetros:
    - G (networkx.Graph): Grafo sobre el que se opera.
    - semilla (int): Semilla para la replicabilidad de los resultados.

    Retorna:
    - nodo (int): Nodo seleccionado aleatoriamente del grupo de menor grado.
    """
    generador = np.random.default_rng(semilla)
    nodos_ordenados = sorted(G.degree(), key=lambda x: x[1])
    num_candidatos = math.ceil(len(G) * 0.01)
    candidatos = [nodo for nodo, grado in nodos_ordenados[:num_candidatos]]
    nodo = generador.choice(candidatos)
    return nodo

def probabilidades_salto_dirigido(G, T_obs, dist):
    """
    Calcula la probabilidad para el 'Salto Dirigido por los Datos' basándose en:
    1. Proximidad a nodos infectados: Se da más peso a los nodos cercanos a los que
       se detectaron primero (bajos T_obs).
    2. Centralidad de intermediación: Identifica nodos que actúan como puentes
       estratégicos dentro de la topología de la red.

    Parámetros:
    - G (networkx.Graph): Grafo sobre el que se opera.
    - T_obs (dict): Diccionario con los tiempos de observación de los nodos infectados.
    - dist (dict): Diccionario de distancias entre pares de nodos del grafo.

    Retorna:
    - probabilidades_salto (np.ndarray): Vector de probabilidades de salto normalizado
      para cada nodo del grafo.
    """
    nodos_observados = list(T_obs.keys())
    centralidad = nx.betweenness_centrality(G, k=200)
    probabilidades_salto = np.zeros(len(G))

    for nodo in G.nodes():
        dist_a_obs = [dist[nodo][obs] for obs in nodos_observados]
        puntuacion_centralidad = centralidad[nodo]
        peso_proximidad = 1.0 / (1.0 + min(dist_a_obs))
        probabilidades_salto[nodo] = puntuacion_centralidad * peso_proximidad

    if np.sum(probabilidades_salto) > 0:
        probabilidades_salto = probabilidades_salto / np.sum(probabilidades_salto)

    return probabilidades_salto

# platano ----------------------------------------------------------------------------------

def log_platano(x, B=0.03):
    """
    Log-densidad objetivo de la distribución plátano.

    Parámetros:
    - x (array-like): Vector de dos componentes [x0, x1] en el que evaluar la densidad.
    - B (float): Parámetro de curvatura de la distribución. Por defecto 0.03.

    Retorna:
    - (float): Valor del logaritmo de la densidad evaluado en x.
    """
    return -0.5 * (x[0]**2 / 100 + (x[1] + B * x[0]**2 - 100 * B)**2)

def pdf_banana(x):
    return np.exp(log_banana(x))
