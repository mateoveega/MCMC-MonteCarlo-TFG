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

def calculate_energy(perm, O_matrix, M_log_esp):
    M_permuted = M_log_esp[np.ix_(perm, perm)]
    energy = -np.sum(O_matrix * M_permuted)
    return energy

def proposal_B1(perm):
    new_perm = perm.copy()
    i, j = random.sample(range(size), 2)
    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
    return new_perm

def proposal_B2(perm):
    new_perm = perm.copy()
    i, j, k = random.sample(range(size), 3)
    new_perm[i], new_perm[j], new_perm[k] = new_perm[k], new_perm[i], new_perm[j]
    return new_perm
