import re
import numpy as np

def char_to_int(c):
    """Mapea A-Z a 0-25, y el espacio ' ' al índice 26."""
    if c == ' ': return 26
    return ord(c) - 65

def int_to_char(i):
    """Mapea 0-25 a A-Z, y el índice 26 al espacio ' '."""
    if i == 26: return ' '
    return chr(i + 65)

def clean_text_with_spaces(text):
    """Limpia el texto dejando letras mayúsculas y espacios reales."""
    text = text.upper()
    text = re.sub(r'[\n\t]+', ' ', text) # Saltos de línea a espacios
    text = re.sub(r'[^A-Z ]', '', text)  # Mantenemos A-Z y espacios
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

def transition_b1_local(current_state):
    """
    B1 (Exploración Local): Intercambia dos caracteres cualesquiera de la permutación.
    Proporciona movilidad básica en todo el espacio de búsqueda.
    """
    
    proposed = current_state.copy()
    
    # Elige 2 posiciones cualquiera de toda la clave
    p1, p2 = rng.choice(len(proposed), size=2, replace=False)
    
    proposed[p1], proposed[p2] = proposed[p2], proposed[p1]
    return proposed

def transition_b2_structural(current_state):
    
        proposed = current_state.copy()
        indices_frecuentes = [0, 4, 15, 19, 18, 13]
        # Solo intercambia entre los números correspondientes a A, E, O, S, R, N
        p1, p2 = rng.choice(indices_frecuentes, size=2, replace=False)
        proposed[p1], proposed[p2] = proposed[p2], proposed[p1]
        return proposed

