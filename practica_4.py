import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from itertools import combinations
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos
datos_originales = pd.read_csv('irisbin.csv', header=None)
caracteristicas_originales = datos_originales.iloc[:, :-3].values
etiquetas_originales = datos_originales.iloc[:, -3:].values

# Reducción de dimensionalidad con PCA
pca = PCA(n_components=2)
caracteristicas_pca = pca.fit_transform(caracteristicas_originales)

# Dividir los datos en conjuntos de entrenamiento y prueba
caracteristicas_entrenamiento, caracteristicas_prueba, etiquetas_entrenamiento, etiquetas_prueba = train_test_split(caracteristicas_pca, etiquetas_originales, test_size=0.2, random_state=42)

# Crear y entrenar un perceptrón multicapa
modelo_mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
modelo_mlp.fit(caracteristicas_entrenamiento, etiquetas_entrenamiento)

# Validar resultados usando leave-k-out
def leave_k_out(caracteristicas, etiquetas, k):
    n = len(caracteristicas)
    precisiones = []
    for indices in combinations(range(n), k):
        mascara = np.ones(n, dtype=bool)
        mascara[list(indices)] = False
        caracteristicas_entrenamiento, caracteristicas_validacion = caracteristicas[mascara], caracteristicas[~mascara]
        etiquetas_entrenamiento, etiquetas_validacion = etiquetas[mascara], etiquetas[~mascara]
        modelo_mlp.fit(caracteristicas_entrenamiento, etiquetas_entrenamiento)
        etiquetas_predichas = modelo_mlp.predict(caracteristicas_validacion)
        precisiones.append(accuracy_score(etiquetas_validacion, etiquetas_predichas))
    return precisiones

# Calcular el error esperado de clasificación, promedio y desviación estándar
k_out_precisions = leave_k_out(caracteristicas_originales, etiquetas_originales, 1)  # Usando leave-one-out como ejemplo
error_esperado = 1 - np.mean(k_out_precisions)
precision_promedio = np.mean(k_out_precisions)
desviacion_estandar = np.std(k_out_precisions)

print("Error esperado de clasificación:", error_esperado)
print("Precisión promedio:", precision_promedio)
print("Desviación estándar de la precisión:", desviacion_estandar)

# Asignar un valor de color único a cada clase usando una paleta de colores
paleta_colores = sns.color_palette("husl", len(np.unique(etiquetas_originales)))
colores_clases = [paleta_colores[i] for i in range(len(np.unique(etiquetas_originales)))]

# Crear un diccionario que mapea las clases a los colores
mapeo_clase_color = {clase_unica: color for clase_unica, color in zip(np.unique(etiquetas_originales), colores_clases)}

# Asignar los colores a cada muestra en función de su clase
colores = [mapeo_clase_color[etiqueta[0]] for etiqueta in etiquetas_originales]

# Graficar la distribución de clases para el dataset Irisbin después de la reducción de dimensionalidad
plt.figure(figsize=(8, 6))
for clase, color in mapeo_clase_color.items():
    plt.scatter(caracteristicas_pca[etiquetas_originales[:, 0] == clase, 0],
                caracteristicas_pca[etiquetas_originales[:, 0] == clase, 1],
                c=[color],
                label=f'Clase {int(clase)}')

plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Distribución de clases para el dataset Irisbin después de PCA')
plt.legend(title='Clases')
plt.show()
