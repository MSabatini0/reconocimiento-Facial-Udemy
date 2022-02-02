import cv2 as cv

origen= 'C:/Users/msaba/Programas/Visual Studio/reconocimiento facial/Para analizar'
archivo= 'foto 5.jpg'
ruta=origen + '/' + archivo

face_detector= cv.CascadeClassifier(r'C:\Users\msaba\Programas\Visual Studio\reconocimiento facial\haarcascade_frontalface_default.xml')

imagen=cv.imread(ruta)
imagen=cv.resize(imagen, (700,500), interpolation=cv.INTER_CUBIC)
grises=cv.cvtColor(imagen,cv.COLOR_BGR2GRAY)
caras=face_detector.detectMultiScale(grises, 
scaleFactor=1.2,
minNeighbors=3,
minSize=(30,30),
flags=cv.CASCADE_SCALE_IMAGE)

for cara in caras:
    cv.rectangle(imagen, cara, (255,0,0), 2)

print(f'se encontraron {len(caras)} caras')


cv.imshow('foto', imagen)

cv.waitKey(0)
cv.destroyAllWindows()