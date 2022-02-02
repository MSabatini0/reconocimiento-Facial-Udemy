import cv2 as cv
import numpy as np
import os

ruta_datos= 'C:/Users/msaba/Programas/Visual Studio/reconocimiento facial/data'

listadatos=os.listdir(ruta_datos)

ids=[]
rostros=[]
id=0

for lista in listadatos:
    ruta=ruta_datos+ '/' + lista
    for foto in os.listdir(ruta):
        ids.append(id)
        rostros.append(cv.imread(ruta + '/' + foto, 0))
        #imagenes=cv.imread(ruta + '/' + foto, 0)
    id+=1

modelo1=cv.face.EigenFaceRecognizer_create()
print('inicio entrenamiento')
modelo1.train(rostros, np.array(ids))

modelo1.write('C:/Users/msaba/Programas/Visual Studio/reconocimiento facial/entrenamiento1.xml')
