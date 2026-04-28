import numpy as np

def box_muller(D, num_samples, seed=1):
    """
    Genera una matriz de tamaño (num_samples, D) usando el método polar de Box-Muller.
    Cada fila es una muestra independiente de N(0, I_D).
    """
    # Establecemos una semilla para la replicabilidad de la imagen
    rng = np.random.default_rng(seed)
    
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

def rejection(target_pdf, proposal_pdf, proposal_sampler, k, num_samples, seed=1):
    """
    Aplica el algoritmo de muestreo por rechazo.
    
    Parámetros:
    - target_pdf (callable): Función que evalúa la PDF de la distribución objetivo p(x).
    - proposal_pdf (callable): Función que evalúa la PDF de la propuesta q(x).
    - proposal_sampler (callable): Función que genera 'num_samples' muestras de la propuesta.
    - k (float o array): Constante de escalado tal que p(x) <= k * q(x) en todo el dominio.
    - num_samples (int): Número total de candidatos a generar.
    
    Retorna:
    - accepted_samples (np.ndarray): Las muestras que han sido aceptadas.
    - acceptance_rate (float): La proporción de muestras aceptadas (0 a 1).
    """
    # Establecemos una semilla para la replicabilidad de la imagen
    rng = np.random.default_rng(seed)
    
    # Extraemos todas las muestras candidatas
    x = proposal_sampler(num_samples)
    
    # Evaluamos ambas PDF en los puntos generados
    p_x = target_pdf(x)
    q_x = proposal_pdf(x)
    
    # Generamos los umbrales uniformes para decidir si aceptamos
    u = rng.uniform(0, k * q_x) # u se genera entre 0 y k * q(x)
    
    # Aplicamos el filtro de aceptación: u <= p(x)
    accepted_mask = (q_x > 1e-10) & (u <= p_x) # Mantenemos q_z > 1e-10 para evitar colas de probabilidad 0
    
    # Extraemos solo las muestras que cumplieron la condición
    accepted_samples = x[accepted_mask]
    
    # Calculamos la tasa de aceptación
    acceptance_rate = len(accepted_samples) / num_samples
    
    return accepted_samples, acceptance_rate
