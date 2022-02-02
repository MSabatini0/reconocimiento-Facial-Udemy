import cv2 as cv
import os

ruta_datos= 'C:/Users/msaba/Programas/Visual Studio/reconocimiento facial/data'

listadatos=os.listdir(ruta_datos)

modelo1=cv.face.EigenFaceRecognizer_create()
modelo1.read('C:/Users/msaba/Programas/Visual Studio/reconocimiento facial/entrenamiento1.xml')

ruidos=cv.CascadeClassifier('C:/Users/msaba/Programas/Visual Studio/reconocimiento facial/haarcascade_frontalface_default.xml')

camara=cv.VideoCapture(0)

while True:
    _,captura=camara.read()
    grises=cv.cvtColor(captura,cv.COLOR_BGR2GRAY)
    idcaptura=grises.copy()
    caras=ruidos.detectMultiScale(grises, 1.3, 5)

    for (x,y,e1,e2) in caras: #cara= x,y,e1,e2, e1 y e2 son ancho y alto de la cara, respectivamente
        capturacara=idcaptura[y:y+e2, x:x+e1]
        capturacara=cv.resize(capturacara, (160,160), interpolation=cv.INTER_CUBIC)
        resultado=modelo1.predict(capturacara) #arroja 2 valores: primero etiqueta, segundo valor prediccion
        print(resultado)
        if resultado[1]<8000: #este valor es diferente para cada cara, esto hay que refinarlo para hacer un mejor reconocimiento
            cv.putText(captura, '{}'.format(listadatos[resultado[0]]),(x,y-20), 1, 1.3, (255,0,0), 1, cv.LINE_AA)
            cv.rectangle(captura, (x,y), (x+e1, y+e2), (0,255,0),2)
        else:
            cv.putText(captura, 'no encontrado' ,(x,y-20), 1, 1.3, (255,0,0), 1, cv.LINE_AA)
            cv.rectangle(captura, (x,y), (x+e1, y+e2), (0,255,0),2)

    cv.imshow('Resultado', captura)

    if cv.waitKey(1)==ord('q'):
        break

camara.release()
cv.destroyAllWindows()