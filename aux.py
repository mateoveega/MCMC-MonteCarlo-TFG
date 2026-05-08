import re
import unicodedata
import numpy as np
import math

def char_to_int(c):
    """Mapea A-Z a 0-25, y el espacio ' ' al índice 26."""
    if c == ' ': return 26
    return ord(c) - 65

def int_to_char(i):
    """Mapea 0-25 a A-Z, y el índice 26 al espacio ' '."""
    if i == 26: return ' '
    return chr(i + 65)

def clean_text_with_spaces(text):
    """
    Limpia el texto convirtiendo a mayúsculas, eliminando tildes
    y dejando solo letras A-Z y espacios.
    """
    # 1. Convertir a mayúsculas
    text = text.upper()
    
    # 2. Normalizar tildes (NFD separa la letra del acento)
    # Ejemplo: 'Á' se convierte en 'A' + '´'
    text = unicodedata.normalize('NFD', text)
    
    # 3. Codificar en ASCII ignorando caracteres no representables (los acentos)
    # y decodificar de nuevo a string.
    text = text.encode('ascii', 'ignore').decode('utf-8')
    
    # 4. Tu lógica original de limpieza con Regex
    text = re.sub(r'[\n\t]+', ' ', text) # Saltos de línea a espacios
    text = re.sub(r'[^A-Z ]', '', text)  # Mantenemos solo A-Z y espacios
    text = re.sub(r' +', ' ', text)      # Colapsamos espacios múltiples
    
    return text.strip()

def get_bigram_matrix_27(text,size):
    """Calcula la matriz (size x size) de probabilidades de bigramas."""
    matrix = np.ones((size, size))
    for i in range(len(text) - 1):
        idx1 = char_to_int(text[i])
        idx2 = char_to_int(text[i+1])
        matrix[idx1, idx2] += 1

    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    return matrix

def transition_b1_double(current_state, rng):
    """
    B1 (Exploración Local): Intercambia dos caracteres cualesquiera de la permutación.
    Proporciona movilidad básica en todo el espacio de búsqueda.
    """
    
    proposed = current_state.copy()
    
    # Elige 2 posiciones cualquiera de toda la clave
    p1, p2 = rng.choice(len(proposed), size=2, replace=False)
    
    proposed[p1], proposed[p2] = proposed[p2], proposed[p1]
    return proposed

def transition_b2_structural(current_state, rng):
    # 1. Convertimos a lista para poder usar .index()
    proposed = list(current_state.copy())
    
    # 2. Estos son los VALORES de las letras A, E, O, S, R, N
    letras_objetivo = [0, 4, 15, 19, 18, 13]
    
    # 3. Elegimos dos letras planas para intercambiar (ej: la A y la E)
    L1, L2 = rng.choice(letras_objetivo, size=2, replace=False)
    
    # 4. BUSCAMOS en qué posición de la clave están esas letras ahora mismo
    idx1 = proposed.index(L1)
    idx2 = proposed.index(L2)
    
    # 5. Intercambiamos los símbolos cifrados que tienen asignados
    proposed[idx1], proposed[idx2] = proposed[idx2], proposed[idx1]
    
    return np.array(proposed)

def transition_b2_triple(current_state, rng):
    # Aseguramos que sea array de numpy para que funcione proposed[idx]
    proposed = np.array(current_state).copy()
    idx = rng.choice(len(proposed), size=3, replace=False)
    val0, val1, val2 = proposed[idx]
    proposed[idx[1]], proposed[idx[2]], proposed[idx[0]] = val0, val1, val2
    return proposed

def low_degree_node(G, seed = 1):
    """
    Selecciona aleatoriamente un nodo del 1% de los nodos con menor grado (menos conexiones).
    """
    rng = np.random.default_rng(seed)
    # Ordenamos los nodos del grafo por conexiones
    sorted_nodes = sorted(G.degree(), key=lambda x: x[1])
    
    # Tomamos el 1% de nodos con menos conexiones
    candidate_count = math.ceil(len(G) * 0.01)
    candidates = [node for node, degree in sorted_nodes[:candidate_count]]
    
    # Seleccionamos aleatoriamente dentro de ese grupo reducido
    node = rng.choice(candidates)
    
    return node 

def data_driven_hop_probabilities(G, T_obs, dist):
    """
    Calcula la probabilidad para el 'Salto Dirigido por los Datos' basándose en:
    1. Proximidad a nodos infectados: Se da más peso a los nodos cercanos a los que 
       se detectaron primero (bajos T_obs).
    2. Centralidad de intermediación: Identifica nodos que actúan como puentes 
       estratégicos dentro de la topología de la red.
    """
    observed_nodes = list(T_obs.keys())
    
    # Estimamos la intermediación (Betweenness)
    centrality = nx.betweenness_centrality(G, k=200)
    
    # Usamos len(G) para asegurar que el tamaño es correcto
    hop_probabilities = np.zeros(len(G))

    for node in G.nodes():
        # Calculamos distancias a los nodos observados
        # (Asegúrate de que 'dist' sea un diccionario de distancias)
        dist_to_obs = [dist[node][obs] for obs in observed_nodes]

        centrality_score = centrality[node]

        # Peso de proximidad: a menor distancia, mayor peso
        # Proximity weight: 1 / (1 + d)
        proximity_weight = 1.0 / (1.0 + min(dist_to_obs))

        # El score final combina la importancia estructural y la temporal
        hop_probabilities[node] = centrality_score * proximity_weight

    # Normalizamos: la suma de todas las probabilidades debe ser 1.0
    if np.sum(hop_probabilities) > 0:
        hop_probabilities = hop_probabilities / np.sum(hop_probabilities)

    return hop_probabilities
