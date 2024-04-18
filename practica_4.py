import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#!Importamos la clase de la red neuronal
from mpl import NeuralNet
    
#!Funcion para leer el archivo csv
def readFile(f):

    data = pd.read_csv(f)

    #*El archivo tiene cuatro columnas como entradas y tres de salidas
    inputs = data.iloc[:,:4]
    outputs = data.iloc[:,4:]

    #*Usando la libreria de sklearn vamos a dividir el set de datos en 80% para train y 20% para test
    inputTrain, inputTest, outputTrain, outputTest = train_test_split(inputs, outputs, test_size=0.2, random_state=2)

    print(f"{inputTrain.shape}, {inputTest.shape}, {outputTrain.shape}, {outputTest.shape}")

    #*Regresa todos los sets creados
    return inputs, outputs, inputTrain, inputTest, outputTrain, outputTest

#!Creamos las variables para guardar inputs y los tensores
#*Leemos el archivo
inputs, outputs, inputTrain, inputTest, outputTrain, outputTest = readFile("irisbin.csv")

#*Variables para guardar los tensores de la red neuronal
#*Los tensores se van a configurar para que los ejecute el cpu
tenInputTrain = torch.from_numpy(inputTrain.values).float().to("cpu")
tenInputTest = torch.from_numpy(inputTest.values).float().to("cpu")
tenOutputTrain = torch.from_numpy(outputTrain.values).float().to("cpu")
tenOutputTest = torch.from_numpy(outputTest.values).float().to("cpu")

#!Creamos las variables que seran utilizadas para la red neuronal
learningRate = 0.001
epochs = 5000
#*Funcion de perdida
lossFun = nn.BCELoss()

#*Funcion para convertir nuestros output donde -1=0 y 1=1
def transformOutput(output):
    #*Recorre todo el tensor y convierte cada output con el valor correspondiente
    return torch.where(output == -1, torch.tensor(0.0), torch.tensor(1.0))

#*Transforma la prediccion en el output que esperamos
def trasnformPredict(output):
    return torch.where(output > 0.5, torch.tensor(1.0), torch.tensor(-1.0))

#!Creamos nuestro objeto de la red neuronal y empieza el entrenamiento

myNet = NeuralNet(4, 3, 3, 3)

#*Optimizador para calcular el gradiente descendiente
optim = torch.optim.Adam(params=myNet.parameters(), lr=learningRate)

print(f"Comienza el entrenamiento con {epochs} epocas...")
for i in range(epochs):
    #*Se obtiene una prediccion en base al tensor de entradas de entrenamiento
    prediction = myNet(tenInputTrain)
    #*Calcula la funcion de perdida
    #*Convierte nuestro output esperado a terminos binarios, esto con el fin
    #* de que la red neuronal nos de valores esperados
    perdida = lossFun(prediction, transformOutput(tenOutputTrain))
    #*Cada 200 epocas imprime la precision durante el entrenamiento
    if i % 200 == 0:
        #!Convierte los tensores a arrays de numpy para calcular la precision
        predictNump = trasnformPredict(prediction).cpu().detach().numpy()
        outputNump = transformOutput(tenOutputTrain).cpu().detach().numpy()
        #*Se hace la suma de los elementos para obtener un array con las predicciones correctas
        buenasPred = np.sum(predictNump == outputNump)
        #*Obtenemos el total de outputs
        lenOutput = len(outputNump)
        #*Calcula la precision diviendo el numero de outputs esperado con el numero de predicciones buenas
        accuracy = buenasPred / lenOutput
        #*Imprimimos en porcentaje
        print(f"Precision de la red en la epoca {i}: {accuracy*100:.2f}%")
    #*Hace el backpropagation y con el optimizador se recalculan los pesos
    perdida.backward()
    optim.step()
    optim.zero_grad()

#!Se realiza el test despues del entrenamiento
prediction = myNet(tenInputTest)
print(f"Tensor obtenido:\n\n {trasnformPredict(prediction)}\nTensor esperado:\n\n {tenOutputTest}")