# Métodos de Monte Carlo basados en Cadenas de Markov para el muestreo de una distribución de probabilidad: formulación y experimentación

## Descripción general
En este repositorio presentamos el código fuente y las simulaciones computacionales desarrolladas para el Trabajo de Fin de Grado (TFG) en Matemáticas. Exploramos y programamos diversos algoritmos de la familia MCMC, analizando empíricamente su comportamiento y convergencia a la hora de muestrear distribuciones de probabilidad.

## Estructura del proyecto
El código del repositorio se articula en torno a dos módulos principales, los cuales dan soporte a las simulaciones y experimentos descritos a lo largo de todos los capítulos de la memoria:

* **`samplers.py`**: Contiene la implementación de los algoritmos generadores de cadenas de Markov y otros métodos de muestreo (Gibbs, Random Walk Metropolis, Metropolis-Hastings, Rechazo, Box-Muller, transiciones compuestas).
* **`aux.py`**: Agrupa las funciones matemáticas y herramientas auxiliares requeridas para el cálculo de densidades objetivo, log-probabilidades, distribuciones específicas, modelado de grafos, análisis de matrices de transición y configuración de representaciones gráficas.

Las referencias internas dentro de estos módulos (Capítulo 1, Capítulo 2 y Capítulo 3) actúan como marcadores lógicos para vincular directamente los bloques de código con los apartados teóricos, formulaciones y casos prácticos desarrollados en el documento escrito del TFG.

## Tecnologías empleadas
Para el desarrollo de las simulaciones hemos utilizado el siguiente entorno científico:
* **Python 3**: Lenguaje base para la programación de los algoritmos.
* **NumPy & SciPy**: Para el manejo matricial, cálculo de densidades multivariantes y generación de secuencias de números pseudoaleatorios de forma reproducible.
* **Matplotlib**: Para la representación gráfica de las trayectorias de las cadenas, histogramas y mapeo de curvas de nivel.
* **NetworkX**: Para el modelado topológico y el diseño de heurísticas de salto sobre estructuras de redes.

## Equipo académico
* **Autor:** Mateo Vega Pueyo
* **Tutor:** Pablo Morales Álvarez
* **Institución:** Universidad de Granada (UGR) 
* **Convocatoria:** Junio de 2026
