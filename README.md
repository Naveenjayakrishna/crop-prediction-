Import numpy as np

Import matplotlib.pyplot as mtp

Import pandas as pd

Data_set=pd.read_csv(“/content/sample_data/Crop_recommendation.csv”)

X=data_set.iloc[:,:-1].values

y=data_set.iloc[:,7].values

from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan,strategy=’mean’)

imputer=imputer.fit(x[:,1:8])

x[:,1:8]=imputer.transform(x[:,1:8])

print(x[:,1:8])

from sklearn.preprocessing import LabelEncoder

label_encoder_x=LabelEncoder()

x[:,6]=label_encoder_x.fit_transform(x[:,6])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder_x=LabelEncoder()

x[:,6]=label_encoder_x.fit_transform(x[:,6])

Onehot_encoder=OneHotEncoder()

X=onehot_encoder.fit_transform(x).toarray()

Labelencoder_y=LabelEncoder()

y=labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

From sklearn.preprocessing import StandardScaler

St_x=StandardScaler()

X_train=st_x.fit_transform(x_train)

X_test=st_x.transform(x_test)

X_train

X_test

y_train

y_test

from sklearn.tree import DecisionTreeClassifier 

classifier= DecisionTreeClassifier(criterion=’entropy’, random_state=0) 

classifier.fit(x_train, y_train)

Y_pred= classifier.predict(x_test)

From sklearn.metrics import confusion_matrix

Cm= confusion_matrix(y_test, y_pred)
