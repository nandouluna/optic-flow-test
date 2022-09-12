import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def evaluar_promedios(of_test, metodo):
    y= np.ones(of_test.shape[0])

    for i in range(0,of_test.shape[0],1):
        
        if(metodo==1):
            u1= of_test[i,0:364] 
            v1= of_test[i,364:728] 
            u2= of_test[i,728:1092] 
            v2= of_test[i,1092:1456] 
            u3= of_test[i,1456:1820]
            v3= of_test[i,1820:2184] 
            u4= of_test[i,2184:2548] 
            v4= of_test[i,2548:2912]

        if(metodo==2):
            u1= of_test[i,0:16] 
            v1= of_test[i,16:32] 
            u2= of_test[i,32:48] 
            v2= of_test[i,48:64] 
            u3= of_test[i,64:80] 
            v3= of_test[i,80:96] 
            u4= of_test[i,96:112] 
            v4= of_test[i,112:128]
        print(np.sum(u1))
        print(np.sum(u2))
        umbral=0.3
        if(np.sum(u1)>-umbral and np.sum(u2)<umbral):
            y[i]=1
        if(np.sum(u1)<-umbral and np.sum(u2)>umbral ):
            y[i]=2
        if(np.sum(u1)>umbral and np.sum(u2)<-umbral ):
            y[i]=3
        if(np.sum(u1)<-umbral and np.sum(u2)<-umbral):
            y[i]=4
        if(np.sum(u1)>umbral and np.sum(u2)>umbral):
            y[i]=5
    return y
        
        




if __name__ == '__main__':
    
    data= np.loadtxt("output/Recorrido 3 Flujo optico LK con k 15 2022-09-09 08_27_04.txt")

    X= data[:,0:data.shape[1]-1]

    
    X=X/X.max(axis=0)

    Y= data[:,-1]

    #X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.30, random_state = 0)

    #y_tested=evaluar_promedios(X_Test,2)

    #print (y_tested)
    
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
        # Feature Scaling
 
        # Fitting the classifier into the Training set

    #classifier= tree.DecisionTreeClassifier()
    #classifier= make_pipeline(StandardScaler(), SVC(gamma='auto'));
    #classifier = AdaBoostClassifier(n_estimators=300,random_state=0,) 
    classifier = RandomForestClassifier(n_estimators = 600, criterion = 'gini', random_state = 0)
    classifier.fit(X_Train,Y_Train)
    #Pruebas
    Y_Pred = classifier.predict(X_Test)


    #Metricas
    # Matriz de confusión
    cm = confusion_matrix(Y_Test, Y_Pred)
    print("Matriz de confusión")
    print(cm)
    # Accuracy
    print("Accuracy")
    print(accuracy_score(Y_Test, Y_Pred))
    # Recall
    print("Recall")
    print(recall_score(Y_Test, Y_Pred, average=None))
    print("Precision")
    # Precision
    print(precision_score(Y_Test, Y_Pred , average=None))
    print("F1 Score")
    # F1 Score
    print(f1_score(Y_Test, Y_Pred , average=None))
    #Mostrar la matriz de confusión
    fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(5, 5),
    cmap=plt.cm.Blues, colorbar=True)
    plt.xlabel('Valores de predicción', fontsize=12)
    plt.ylabel('Valores reales', fontsize=12)
    plt.title('Matriz de confusión', fontsize=12)
    plt.show()