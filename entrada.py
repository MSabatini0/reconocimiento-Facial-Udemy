import cv2 as cv
import os

ruta1= 'C:/Users/msaba/Programas/Visual Studio/reconocimiento facial/fotos'

if not os.path.exists(ruta1):
    os.mkdir(ruta1)

id=0

ruidos= cv.CascadeClassifier(r'C:\Users\msaba\Programas\Visual Studio\reconocimiento facial\haarcascade_frontalface_default.xml')
camara=cv.VideoCapture(0)#'C:/Users/msaba/Programas/Visual Studio/reconocimiento facial/Para analizar/video 2.avi') #0= activa camara

while True:
    _,captura=camara.read()
    grises=cv.cvtColor(captura,cv.COLOR_BGR2GRAY)
    idcaptura=captura.copy()
    caras=ruidos.detectMultiScale(grises, 1.2, 3) #devuelve coordenadas de cada cara encontrada
    
    for (x,y,e1,e2) in caras: #cara= x,y,e1,e2, e1 y e2 son ancho y alto de la cara, respectivamente
        cv.rectangle(captura, (x,y), (x+e1, y+e2), (255,0,0), 2)
        capturacara=idcaptura[y:y+e2, x:x+e1]
        capturacara=cv.resize(capturacara, (160,160), interpolation=cv.INTER_CUBIC)
        cv.imwrite(ruta1+'/imagen {}.jpg'.format(id), capturacara)
        id+=1


    cv.imshow("resultado", captura)

    if id==50:
        break

camara.release()
cv.destroyAllWindows()
