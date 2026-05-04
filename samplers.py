import numpy as np
from scipy.stats import multivariate_normal

def box_muller(D, num_samples, seed=1):
    """
    Genera una matriz de muestras usando el método polar de Box-Muller.
    
    Parámetros:
    - D (int): Dimensión del espacio (número de columnas).
    - num_samples (int): Número total de muestras a generar (número de filas).
    - seed (int): Semilla para la replicabilidad de los resultados.
    
    Retorna:
    - samples_matrix (np.ndarray): Matriz de tamaño (num_samples, D) con muestras N(0, I_D).
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
    - seed (int): Semilla para la replicabilidad de los resultados.
    
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

def random_walk_metropolis(target_pdf, num_iterations, initial_position, sigma_proposal, seed=1):
    """
    Algoritmo Random Walk Metropolis para cualquier distribución objetivo.
    
    Parámetros:
    - target_pdf (callable): Función que evalúa la densidad de probabilidad p(x).
    - num_iterations (int): Número de pasos de la cadena.
    - initial_position (array-like): Punto de inicio de la cadena.
    - sigma_proposal (float): Desviación estándar del salto de propuesta.
    - seed (int): Semilla para la replicabilidad de los resultados.
    
    Retorna:
    - np.ndarray: Matriz con las muestras de la cadena.
    - float: Tasa de aceptación.
    """
    # Establecemos una semilla para la replicabilidad del experimento
    rng = np.random.default_rng(seed)

    accepted_samples = []
    accepted_count = 0
    
    current_position = np.array(initial_position) # Aseguramos que la posición inicial sea un array de NumPy
    D = len(current_position) # Inferimos la dimensión desde la posición inicial
    p_current = target_pdf(current_position) # Calculamos la densidad inicial

    for i in range(num_iterations):
        # Generamos una propuesta de nuevo estado (z_prime) usando ruido gaussiano
        proposal_noise = rng.normal(loc=0.0, scale=sigma_proposal, size=D)
        z_prime = current_position + proposal_noise

        p_prime = target_pdf(z_prime) # Evaluamos la densidad del salto propuesto

        # Calculamos la probabilidad de aceptación alpha
        if p_current == 0 and p_prime > 0:
            alpha = 1.0
        elif p_current == 0 and p_prime == 0:
            alpha = 0.0
        elif p_current > 0:
            alpha = min(1.0, p_prime / p_current)
        else:
            alpha = 0.0

        # Vemos si el salto propuesto se acepta o no
        u = rng.random()
        # Si se acepta, actualizamos la posición y la densidad actual
        if u <= alpha:
            current_position = z_prime
            p_current = p_prime
            accepted_count += 1

        # Agregamos la posición actual a las muestras
        accepted_samples.append(current_position.copy())

    # Calculamos la tasa de aceptación
    acceptance_rate = accepted_count / num_iterations

    return np.array(accepted_samples), acceptance_rate

def metropolis_hastings_log(initial_state, log_target_pdf, proposal_sampler, log_proposal_pdf, iterations, seed = 1):
    # Establecemos una semilla para la replicabilidad de la imagen
    rng = np.random.default_rng(seed)

    initial_state = initial_state
    best_state = initial_state.copy()
    current_state = initial_state.copy()
    
    # Calculamos la probabilidad logarítmica del estado inicial
    current_log_target = log_target_pdf(initial_state)
    best_log_target = current_log_target
    
    historial = []

    for i in range(iterations):
        # Proponemos un nuevo estado
        proposed_state = proposal_sampler(current_state)
        
        # Calculamos log-probabilidad del nuevo estado
        proposed_log_target = log_target_pdf(proposed_state)
        
        # Calculamos las log-probabilidades de transición Q (Hastings)
        log_proposal_forward = log_proposal_pdf(proposed_state, current_state)
        log_proposal_backward = log_proposal_pdf(current_state, proposed_state)
        
        # Calculamos el Log-Ratio de aceptación
        log_ratio = (proposed_log_target - current_log_target) + (log_proposal_backward - log_proposal_forward)
        
        # Criterio de aceptación en espacio logarítmico
        # Si log_ratio >= 0 (el ratio normal era >= 1), aceptamos siempre.
        if log_ratio >= 0 or np.log(rng.random()) < log_ratio:
            current_state = proposed_state
            current_log_target = proposed_log_target
            
            # Guardamos el mejor absoluto encontrado
            if current_log_target > best_log_target:
                best_log_target = current_log_target
                best_state = current_state.copy()
                
        historial.append(current_log_target)
        
    return best_state, historial

def metropolis_hastings_log(initial_state, log_target_pdf, proposal_sampler, log_proposal_pdf, iterations, seed = 1):
    # Establecemos una semilla para la replicabilidad de la imagen
    rng = np.random.default_rng(seed)

    initial_state = initial_state
    best_state = initial_state.copy()
    current_state = initial_state.copy()

    # Calculamos la probabilidad logarítmica del estado inicial
    current_log_target = log_target_pdf(initial_state)
    best_log_target = current_log_target
    
    historial = []

    for i in range(iterations):
        # Proponemos un nuevo estado
        proposed_state = proposal_sampler(current_state)
        
        # Calculamos log-probabilidad del nuevo estado
        proposed_log_target = log_target_pdf(proposed_state)
        
        # Calculamos las log-probabilidades de transición Q (Hastings)
        log_proposal_forward = log_proposal_pdf(proposed_state, current_state)
        log_proposal_backward = log_proposal_pdf(current_state, proposed_state)
        
        # Calculamos el Log-Ratio de aceptación
        log_ratio = (proposed_log_target - current_log_target) + (log_proposal_backward - log_proposal_forward)
        
        # Criterio de aceptación en espacio logarítmico
        # Si log_ratio >= 0 (el ratio normal era >= 1), aceptamos siempre.
        if log_ratio >= 0 or np.log(rng.random()) < log_ratio:
            current_state = proposed_state
            current_log_target = proposed_log_target
            
            # Guardamos el mejor absoluto encontrado
            if current_log_target > best_log_target:
                best_log_target = current_log_target
                best_state = current_state.copy()
                
        historial.append(current_log_target)
        
    return best_state, historial
