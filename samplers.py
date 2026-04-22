import numpy as np

# Establecemos una semilla para la replicabilidad de la imagen
rng = np.random.default_rng(1)

def box_muller(D, num_samples):
    """
    Genera una matriz de tamaño (num_samples, D) usando el método polar de Box-Muller.
    Cada fila es una muestra independiente de N(0, I_D).
    """
    # Calculamos el total de números individuales que necesitamos generar
    total_samples = D * num_samples
    samples = []

    while len(samples) < total_samples: # Aplicamos el algoritmo de Box-Muller en forma polar
        # Generamos dos muestras uniformemente distribuidas en (-1,1)
        X1 = 2 * rng.random() - 1
        X2 = 2 * rng.random() - 1

        R = X1 ** 2 + X2 ** 2 # Calculamos el radio al cuadrado

        if 0 < R < 1: # Rechazamos si está fuera del disco unidad
            factor = np.sqrt(-2 * np.log(R) / R) # Cambio de variable
            Y1 = X1 * factor
            Y2 = X2 * factor
            
            # Usamos extend para añadir ambos valores
            samples.extend([Y1, Y2])

    # Recortamos exactamente al número de elementos que necesitamos (por si sobra uno)
    flat_samples = np.array(samples)[:total_samples] 
    
    # Reorganizamos la lista plana en una matriz de (num_samples filas, D columnas)
    samples_matrix = flat_samples.reshape((num_samples, D))
    
    return samples_matrix 
