import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score

data = pd.read_csv("HousePriceExample.csv", header=0)
col_a = list(data.Size)
col_b = list(data.Rooms)

col_d = []
for i in range(0,len(col_a)):
    col_d.append([col_a[i],col_b[i]])
    
#col_d

col_c = list(data.Price)

X_train =col_d
Y_train=col_c

#### to solve continues ( create mached datatype )


# STEP 1: training and building DT model
dt = tree.DecisionTreeClassifier()
dt=dt.fit(X_train,Y_train)


# STEP 2: testing our DT model

X_test = [[205,1]] 



# do prediction 

Y_prediction = dt.predict(X_test) 

print(Y_prediction)
