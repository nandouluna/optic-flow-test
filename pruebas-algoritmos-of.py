from cmath import nan, pi
from certifi import where
import cv2
from cv2 import transpose
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import convolve as filter2
#from picamera.array import PiRGBArray # Genera un agreglo de tres dimensiones (BGR)
#from picamera import PiCamera #Provee a Python una interfaz para el modulo de camara
import time
import os
from datetime import datetime
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

def mostrar_imagen(name, image):
    if image is None:
        return

    cv2.imshow(name, image)

def dibujar_vectores_u(u, v, img, halfwindow=1):
    img =  cv2.resize(img,(0,0), fx=10, fy=10, interpolation=cv2.INTER_AREA)
    for j in range(halfwindow, u.shape[0]-halfwindow, 1):
        for i in range(halfwindow, u.shape[1]-halfwindow,1):
            dx = u[j,i]
            dy= v[j,i]
            if(not(np.isnan(dx))):
                #cv2.arrowedLine(img, (i*10+3,j*10), ((i+int(dx))*10,j*10), (255,0,0), 4, 8)
                cv2.arrowedLine(img, (i*10+3,j*10), ((i+int(dx))*10,(j+int(dy))*10), (255,0,0), 4, 8)
                #cv2.arrowedLine(img, (i*10,j*10), (i*10,(j+int(dy))*10), (0,255,0), 4, 8)
    return img


def dibujar_vectores_v(u,v,img,halfwindow=1):
    suma=0
    img =  cv2.resize(img,(0,0), fx=10, fy=10, interpolation=cv2.INTER_AREA)
    for i in range(halfwindow, v.shape[0]-halfwindow, 1):
        for j in range(halfwindow, v.shape[1]-halfwindow,1):
            dy = v[i,j]
            #print(dy)
            dx = u[i,j]
            #print(dx)
            if(not(np.isnan(dy))and not(np.isnan(dx))):
                #cv2.arrowedLine(img, (j*10+3,i*10),  (j*10, (i+int(dy))*10), (0,255,0), 4, 8 )
                cv2.arrowedLine(img, (j*10+3,i*10),  ((j+int(dx))*10, (i+int(dy))*10), (0,255,0), 4, 8 )
    return img

def obtener_derivadas(img1, img2):
    #mascaras de convolución
    x_kernel = np.array([[-1, 1], [-1, 1]]) * 0.25
    y_kernel = np.array([[-1, -1], [1, 1]]) * 0.25
    t_kernel = np.array([[1, 1], [1, 1]]) * 0.25

    fx = filter2(img1,x_kernel) + filter2(img2,x_kernel)
    fy = filter2(img1, y_kernel) + filter2(img2, y_kernel)
    ft = filter2(img1, -t_kernel) + filter2(img2, t_kernel)

    cv2.imshow("ix", fx)
    cv2.imshow("iy", fy)
    cv2.imshow("it", ft)

    return [fx,fy, -ft]

def calculaHS(beforeImg, afterImg, alpha, it):
    #Valores iniciales
    u = np.zeros((beforeImg.shape[0], beforeImg.shape[1]))
    v = np.zeros((beforeImg.shape[0], beforeImg.shape[1]))
      
    fx, fy, ft = obtener_derivadas(beforeImg, afterImg)

    kernel_promedio = np.array([[1 / 12, 1 / 6, 1 / 12], [1 / 6, 0, 1 / 6], [1 / 12, 1 / 6, 1 / 12]], float)
    #kernel_promedio = np.array([[-1 / 12, 8/ 12, 0 / 12, -8 / 12, 1 / 12]], float)
    
    cont_iteraciones = 0
    while True:
        cont_iteraciones += 1
        u_promedio = filter2(u, kernel_promedio)
        v_promedio = filter2(v, kernel_promedio)
        p = fx * u_promedio + fy * v_promedio + ft
        d = alpha**2 + fx**2 + fy**2
        u_prev = u
        v_prev = v
        u = u_promedio - fx * (p / d)
        v = v_promedio - fy * (p / d)
        diff_u = np.linalg.norm(u - u_prev, 2)
        diff_v = np.linalg.norm(v - v_prev, 2)
        #if  (diff_u < delta and diff_v < delta) or cont_iteraciones >=20:
        if cont_iteraciones >=it:
            #print("numero de iteraciones: ", cont_iteraciones)
            break
    return [u,v]

def calculaLK(beforeImg, afterImg, windowSize=15):
    #Calculo de las derivadas parciales
    ix, iy, it = obtener_derivadas(beforeImg, afterImg)
    u = np.zeros((beforeImg.shape[0], beforeImg.shape[1]))
    v = np.zeros((beforeImg.shape[0], beforeImg.shape[1]))

    halfwindow= int(windowSize/2)

    for i in range(halfwindow, ix.shape[0]-halfwindow,1):
        for j in range(halfwindow, ix.shape[1]-halfwindow, 1):
            #Se selecciona la región de la ventana en Ix, Iy e It
            tempIx= ix[i-halfwindow:i+halfwindow+1,j-halfwindow:j+halfwindow+1]          
            tempIy= iy[i-halfwindow:i+halfwindow+1,j-halfwindow:j+halfwindow+1]
            tempIt= it[i-halfwindow:i+halfwindow+1,j-halfwindow:j+halfwindow+1]
            
            d= (np.sum(pow(tempIx,2))*np.sum(pow(tempIy,2)))-(pow((np.sum(tempIx*tempIy)),2))

            if(d!=0):
                u[i,j]= (((-1)*(np.sum(pow(tempIy,2)*np.sum(tempIx*tempIt))))+(np.sum(tempIx*tempIy)*np.sum(tempIy*tempIt)))/d
                v[i,j]= ((np.sum(tempIx*tempIt)*np.sum(tempIx*tempIy))-(np.sum(pow(tempIx,2))*np.sum(tempIy*tempIt)))/d
            
    return [u,v]

def flujoOptico(alpha, delta, method=1, kernel_size=0, iter=1, verbose=False):
    
    of_data=[]
    of_labels=[]
    
    tiempos=[]

    pwm=[]

    #Configuración de la camara
    #camera = PiCamera()
    print(cv2.__version__)

    w=160
    h=128
    
    half_kernel=0
    if(kernel_size>0):
        half_kernel=(int)(kernel_size/2)
    else:
        kernel_size=3


    #Tamaño de las ventanas
    vx= 30
    vy= kernel_size
    
    half_h= (int)(h/2)
    half_w= (int)(w/2)
    half_vy= (int)(vy/2)
    
    #ubicación ventana 1
    ux1=0
    uy1=half_h-half_vy
    #ubicación ventana 2
    ux2= w-vx
    uy2= half_h-half_vy
    #ubicación ventana 3
    ux3= half_w-half_vy
    uy3= 0
    #ubicación ventana 4
    ux4= half_w-half_vy
    uy4= h-vx

    #Se asigna la resolución de la camara
    #camera.resolution = (w,h)
    
    #camera.framerate = fps      
     
    # Generates a 3D RGB array and stores it in rawCapture
    #raw_capture = PiRGBArray(camera, size=(w,h))
    band=False
    time.sleep(0.1)
    
    img_before= np.zeros((w,h))
    
    f=0
    k=0
    i=1
    img_before= cv2.imread("imgOF5/imgOF (1).jpg")

    #Se cargan los datos
    """data= np.loadtxt("OF_data5.txt")

    X= data[:,0:data.shape[1]-1]

    #print(X.shape)
    #print(X)

    #X=(X-np.min(X))/(np.max(X)-np.min(X))
    Y= data[:,-1]

    #print(Y.shape)

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
    # Feature Scaling

    # Fitting the classifier into the Training set

    #classifier= tree.DecisionTreeClassifier()
    classifier= make_pipeline(StandardScaler(), SVC(gamma='auto'));
    #classifier = AdaBoostClassifier(n_estimators=300,random_state=0,) 
    #classifier = RandomForestClassifier(n_estimators = 600, criterion = 'entropy', random_state = 0)
    classifier.fit(X_Train,Y_Train)"""

    l=0
    
    #for frame in camera.capture_conrtinuous(raw_capture, format="bgr", use_video_port=True):
    for i in range(1,888,2):#de2000
        print(i)
        
        img_before=cv2.imread("imgOF30/imgOF_"+str(i)+".png")

        img_after= cv2.imread("imgOF30/imgOF_"+str(i+1)+".png")

        
        cv2.imshow("img", img_before)
        #time.sleep(1/30)
        cv2.imshow("img", img_after)
        start = time.time()
        s = 2
        img_b = img_before.copy()
        img_a = img_after.copy()
        img_b = cv2.resize(img_b, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.rectangle(img_b, ((ux1*s), (uy1*s)),(((ux1+vx)*s), ((uy1+vy)*s)), (255, 0, 0), 1)
        cv2.rectangle(img_b, ((ux2*s), (uy2*s)),(((ux2+vx)*s), ((uy2+vy)*s)), (255, 0, 0), 1)
        cv2.rectangle(img_b, ((ux3*s), (uy3*s)),(((ux3+vy)*s), ((uy3+vx)*s)), (255, 0, 0), 1)
        cv2.rectangle(img_b, ((ux4*s), (uy4*s)),(((ux4+vy)*s), ((uy4+vx)*s)), (255, 0, 0), 1)

        #time.sleep(1)

        img_a = cv2.resize(img_a, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.rectangle(img_a, ((ux1*s), (uy1*s)),(((ux1+vx)*s), ((uy1+vy)*s)), (255, 0, 0), 1)
        cv2.rectangle(img_a, ((ux2*s), (uy2*s)),(((ux2+vx)*s), ((uy2+vy)*s)), (255, 0, 0), 1)
        cv2.rectangle(img_a, ((ux3*s), (uy3*s)),(((ux3+vy)*s), ((uy3+vx)*s)), (255, 0, 0), 1)
        cv2.rectangle(img_a, ((ux4*s), (uy4*s)),(((ux4+vy)*s), ((uy4+vx)*s)), (255, 0, 0), 1)
        mostrar_imagen("frame completo", img_a)

        #cv2.imwrite("imgof_"+str(f)+"COMPLETO.jpg",img_a)

        img1_COLOR = img_before[uy1:uy1+vy, ux1:ux1+vx, :]
        img2_COLOR = img_after[uy1:uy1+vy, ux1:ux1+vx, :]

        img1_G = cv2.cvtColor(img_before[uy1:uy1+vy, ux1:ux1+vx, :], cv2.COLOR_BGR2GRAY)
        img2_G = cv2.cvtColor(img_after[uy1:uy1+vy, ux1:ux1+vx, :], cv2.COLOR_BGR2GRAY)

        #Obtención de los cuatro ventanas del frame
        img1 = cv2.cvtColor(img_before[uy1:uy1+vy, ux1:ux1+vx, :], cv2.COLOR_BGR2GRAY).astype(np.float)
        img2 = cv2.cvtColor(img_after[uy1:uy1+vy, ux1:ux1+vx, :], cv2.COLOR_BGR2GRAY).astype(np.float)
        img3 = cv2.cvtColor(img_before[uy2:uy2+vy, ux2:ux2+vx, :], cv2.COLOR_BGR2GRAY).astype(np.float)
        img4 = cv2.cvtColor(img_after[uy2:uy2+vy, ux2:ux2+vx, :], cv2.COLOR_BGR2GRAY).astype(np.float)
        img5 = cv2.cvtColor(img_before[uy3:uy3+vx, ux3:ux3+vy, :], cv2.COLOR_BGR2GRAY).astype(np.float)
        img6 = cv2.cvtColor(img_after[uy3:uy3+vx, ux3:ux3+vy, :], cv2.COLOR_BGR2GRAY).astype(np.float)
        img7 = cv2.cvtColor(img_before[uy4:uy4+vx, ux4:ux4+vy, :], cv2.COLOR_BGR2GRAY).astype(np.float)
        img8 = cv2.cvtColor(img_after[uy4:uy4+vx, ux4:ux4+vy, :], cv2.COLOR_BGR2GRAY).astype(np.float)
        #tiempos.append(time.time()-start)
        #start = time.time()

        tpv=0

        if(method == 1):  # Metodo de Horn-Shunk
            tvp=2912#2912
            half_kernel=1
            u1, v1 = calculaHS(img1, img2, alpha=10, it=10)
            u2, v2 = calculaHS(img3, img4, alpha=10, it=10)
            u3, v3 = calculaHS(img5, img6, alpha=10, it=10)
            u4, v4 = calculaHS(img7, img8, alpha=10, it=10)

            img_u1 = dibujar_vectores_u(u1, v1, img_before[uy1:uy1+vy, ux1:ux1+vx, :], 1)
            img_u2 = dibujar_vectores_u(u2, v2, img_before[uy2:uy2+vy, ux2:ux2+vx, :], 1)
            img_v3 = dibujar_vectores_v(u3, v3, img_before[uy3:uy3+vx, ux3:ux3+vy, :],1)
            img_v4 = dibujar_vectores_v(u4, v4, img_before[uy4:uy4+vx, ux4:ux4+vy, :], 1)

            #print(u1.shape)
            #print(u1[half_kernel:vy-half_kernel,half_kernel:vx-half_kernel].shape)

            of_test=np.zeros(2912)

            of_test[0:364] = u1[half_kernel:vy-half_kernel,half_kernel:vx-half_kernel].flatten(order='C')
            of_test[364:728] = v1[half_kernel:vy-half_kernel,half_kernel:vx-half_kernel].flatten(order='C')
            of_test[728:1092] = u2[half_kernel:vy-half_kernel, half_kernel:vx-half_kernel].flatten(order='C')
            of_test[1092:1456] = v2[half_kernel:vy-half_kernel,half_kernel:vx-half_kernel].flatten(order='C')
            of_test[1456:1820] = u3[half_kernel:vx-half_kernel,half_kernel:vy-half_kernel].flatten(order='F')
            of_test[1820:2184] = v3[half_kernel:vx-half_kernel,half_kernel:vy-half_kernel].flatten(order='F')
            of_test[2184:2548] = u4[half_kernel:vx-half_kernel,half_kernel:vy-half_kernel].flatten(order='F')
            of_test[2548:2912] = v4[half_kernel:vx-half_kernel,half_kernel:vy-half_kernel].flatten(order='F')

            """of_test=np.zeros(8) #128

            of_test[0] = np.mean(u1[half_kernel:vy-half_kernel,half_kernel:vx-half_kernel])
            of_test[1] = np.mean(v1[half_kernel:vy-half_kernel,half_kernel:vx-half_kernel])
            of_test[2] = np.mean(u2[half_kernel:vy-half_kernel, half_kernel:vx-half_kernel])
            of_test[3] = np.mean(v2[half_kernel:vy-half_kernel,half_kernel:vx-half_kernel])
            of_test[4] = np.mean(u3[half_kernel:vx-half_kernel,half_kernel:vy-half_kernel])
            of_test[5] = np.mean(u3[half_kernel:vx-half_kernel,half_kernel:vy-half_kernel])
            of_test[6] = np.mean(u4[half_kernel:vx-half_kernel,half_kernel:vy-half_kernel])
            of_test[7] = np.mean(v4[half_kernel:vx-half_kernel,half_kernel:vy-half_kernel])"""
            

            """print("SUMAS DE VENTANAS")
            print(np.sum(of_data[0:364]))
            print(np.sum(of_data[364:728]))
            print(np.sum(of_data[728:1092]))
            print(np.sum(of_data[1092:1456]))
            print(np.sum(of_data[1456:1820]))
            print(np.sum(of_data[1820:2184]))
            print(np.sum(of_data[2184:2548]))
            print(np.sum(of_data[2548:2912]))

            print("PROMEDIO DE VENTANAS")
            print(np.mean(of_data[0:364]))
            print(np.mean(of_data[364:728]))
            print(np.mean(of_data[728:1092]))
            print(np.mean(of_data[1092:1456]))
            print(np.mean(of_data[1456:1820]))
            print(np.mean(of_data[1820:2184]))
            print(np.mean(of_data[2184:2548]))
            print(np.mean(of_data[2548:2912]))"""
        if(method == 2):  # Metodo de Lucas-Kanade
            tvp=128 #128
            half_kernel=int(kernel_size/2)
            #ix1, iy1, it1 = obtener_derivadas(img1, img2)
            #ix2, iy2, it2 = obtener_derivadas(img3, img4)
            #cv2.imwrite("Prueba IX/imgof_"+str(f)+"IZQ.jpg",ix1.astype(np.uint8))
            #cv2.imwrite("Prueba IY/imgof_"+str(f)+"IZQ.jpg",iy1.astype(np.uint8))
            #cv2.imwrite("Prueba IT/imgof_"+str(f)+"IZQ.jpg",it1.astype(np.uint8))
            #cv2.imwrite("Prueba IX/imgof_"+str(f)+"DER.jpg",ix2.astype(np.uint8))
            #cv2.imwrite("Prueba IY/imgof_"+str(f)+"DER.jpg",iy2.astype(np.uint8))
            #cv2.imwrite("Prueba IT/imgof_"+str(f)+"DER.jpg",it2.astype(np.uint8))

            u1, v1 = calculaLK(img1, img2, kernel_size)
            u2, v2 = calculaLK(img3, img4, kernel_size)
            u3, v3 = calculaLK(img5, img6, kernel_size)
            u4, v4 = calculaLK(img7, img8, kernel_size)

            img_u1 = dibujar_vectores_u(u1, v1, img_before[uy1:uy1+vy, ux1:ux1+vx, :], half_kernel)
            img_u2 = dibujar_vectores_u(u2, v2, img_before[uy2:uy2+vy, ux2:ux2+vx, :], half_kernel)
            img_v3 = dibujar_vectores_v(u3, v3, img_before[uy3:uy3+vx, ux3:ux3+vy, :], half_kernel)
            img_v4 = dibujar_vectores_v(u4, v4, img_before[uy4:uy4+vx, ux4:ux4+vy, :], half_kernel)


            of_test=np.zeros(128) #128

            """of_test[0] = np.mean(u1[half_kernel,half_kernel:vx-half_kernel])
            of_test[1] = np.mean(v1[half_kernel,half_kernel:vx-half_kernel])
            of_test[2] = np.mean(u2[half_kernel,half_kernel:vx-half_kernel])
            of_test[3] = np.mean(v2[half_kernel,half_kernel:vx-half_kernel])
            of_test[4] = np.mean(np.transpose(u3[half_kernel:vx-half_kernel,half_kernel]))
            of_test[5] = np.mean(np.transpose(v3[half_kernel:vx-half_kernel,half_kernel]))
            of_test[6] = np.mean(np.transpose(u4[half_kernel:vx-half_kernel,half_kernel]))
            of_test[7] = np.mean(np.transpose(v4[half_kernel:vx-half_kernel,half_kernel]))"""
            

            of_test[0:16] = u1[half_kernel,half_kernel:vx-half_kernel]
            of_test[16:32] = v1[half_kernel,half_kernel:vx-half_kernel]
            of_test[32:48] = u2[half_kernel,half_kernel:vx-half_kernel]
            of_test[48:64] = v2[half_kernel,half_kernel:vx-half_kernel]
            of_test[64:80] = np.transpose(u3[half_kernel:vx-half_kernel,half_kernel])
            of_test[80:96] = np.transpose(v3[half_kernel:vx-half_kernel,half_kernel])
            of_test[96:112] = np.transpose(u4[half_kernel:vx-half_kernel,half_kernel])
            of_test[112:128] = np.transpose(v4[half_kernel:vx-half_kernel,half_kernel])

        of_data.append(of_test)
        #time.sleep(1)
    
        #of_test = (of_test-np.min(X))/(np.max(X)-np.min(X))

        #print(of_test)
        """texto = "Sin movimiento"

        if(len((np.argwhere(np.isnan(of_test))))==0):
            Y_Pred= classifier.predict(of_test.reshape(1,-1))
            
            if(Y_Pred[0]==1):
                texto = "Sin movimiento"
            elif(Y_Pred[0]==2) :
                texto = "Adelante"
            elif(Y_Pred[0]==3) :
                texto = "Atras"
            elif(Y_Pred[0]==4) :
                texto = "Derecha"
            elif(Y_Pred[0]==5) :
                texto = "Izquierda"""

        """of_data.append(u1[half_kernel,half_kernel:vx-half_kernel])
        of_data.append(v1[half_kernel,half_kernel:vx-half_kernel])
        of_data.append(u2[half_kernel,half_kernel:vx-half_kernel])
        of_data.append(v2[half_kernel,half_kernel:vx-half_kernel])
        of_data.append(u3[half_kernel:vx-half_kernel,half_kernel])
        of_data.append(v3[half_kernel:vx-half_kernel,half_kernel])
        of_data.append(u4[half_kernel:vx-half_kernel,half_kernel])
        of_data.append(v4[half_kernel:vx-half_kernel,half_kernel])"""



        #time.sleep(1/30)
        

        mostrar_imagen("u1", img_u1)
        mostrar_imagen("u2", img_u2)
        mostrar_imagen("v3", img_v3)
        mostrar_imagen("v4", img_v4)
        

        #cv2.imwrite("PruebaIZQ1/imgof_"+str(f)+".jpg",img_u1)
        #cv2.imwrite("PruebaDER1/imgof_"+str(f)+".jpg",img_u2)



        #time.sleep(1/30)
        tiempos.append(time.time()-start)
        """
        key = cv2.waitKey()
        if key == 115: #stop
            of_labels.append(1)
            l=1
        if key == 102: #forward
            of_labels.append(2)
            l=2
        if key == 98: #backward
            of_labels.append(3)
            l=3
        if key == 114: #right
            of_labels.append(4)
            l=4
        if key == 108: #left
            of_labels.append(5)
            l=5"""
            
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
  
        #org
        org = (20, 40)
        
        #fontScale
        fontScale = 0.5
        
        # Blue color in BGR
        color = (255, 0, 0)
        
        # Line thickness of 2 px
        thickness = 1
        
        # Using cv2.putText() method
        img_a = cv2.putText(img_a, texto, org, font, fontScale, color, thickness, cv2.LINE_AA)

        

        cv2.imwrite("Prueba/imgof_"+str(f)+".jpg",img_a)

        """
        mostrar_imagen("frame completo", img_a)

        ##tiempos.append(time.time()-start)
        ##start = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        

        #print("Iteración", f)
        f += 1

        
        #start = time.time()


        #limpsia el flujo para obtener la siguiente captura
        #raw_capture.truncate(0)
        #img_befssore=img_after.copy()

        #print("Tiempo transcurrido", (end-start))

    print(len(of_data))
        
    cv2.destroyAllWindows()

    if(verbose):
        of_txt= np.loadtxt("output/Recorrido 5 Flujo optico LK con k 15 2022-09-09 08_39_03.txt")
        of_labels=of_txt[:,-1]
        now = datetime.now()
        current_time= now.strftime("%Y-%m-%d %H_%M_%S")
        of = np.zeros((len(of_data),tvp+1))
        ts= np.zeros((int(len(of_data)))) 
        for i in range(0, len(of_data),1):
            ts[i]=tiempos[i]
            of[i,0:tvp]=of_data[i]
            of[i,tvp]= round(of_labels[i])       
        if(method == 1):
            np.savetxt('output/Recorrido 5 Flujo optico HS k '+str(kernel_size)+ current_time+".txt",of)
        if(method == 2):
            np.savetxt('output/Recorrido 5 Flujo optico LK con k '+str(kernel_size)+ ' '+current_time+".txt",of)
        np.savetxt('output/Recorrido 5 Tiempos de OF en '+ current_time+".txt",ts)

if __name__ == '__main__':
    
    flujoOptico(alpha=2, delta=5, method=1,kernel_size=15, iter=5, verbose=True)

    #flujoOptico(alpha=2, delta=10, method=2, kernel_size=15, iter=10)
