import re

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

def get_bigram_matrix_27(text):
    """Calcula la matriz empírica (27x27) de probabilidades de bigramas."""
    matrix = np.ones((ALFABETO_SIZE, ALFABETO_SIZE))
    for i in range(len(text) - 1):
        idx1 = char_to_int(text[i])
        idx2 = char_to_int(text[i+1])
        matrix[idx1, idx2] += 1

    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    return matrix
