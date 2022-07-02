import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error

rf = RandomForestClassifier()

df = pd.read_csv("tested.csv")
# print("Dataset: Titanic\n",df,end="\n-----------------------------------------------\n")

#Dealing with Missing values
# print("Checking for Missing values in dataset: \n",df.isnull().sum(),end="\n---------------------------\n")
df['Age'] = df['Age'].fillna((df['Age'].mean()))
df['Fare'] = df['Fare'].fillna((df['Fare'].mean()))

#x and y values
drop_list = ['PassengerId','Name','Ticket','Embarked']
x = df.drop(drop_list,axis=1)
y = df['Embarked']
# print("X, Y values-------\n",x,y)

#Encoding Features
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

x['Sex'] = le.fit_transform(x['Sex'])
x['Cabin'] = le.fit_transform(x['Cabin'])

y =le.fit_transform(y)
# print(x,y)
#feature selection
'''
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_)
feat_imp = pd.Series(model.feature_importances_,index=x.columns)
feat_imp.nlargest(4).plot(kind='barh')
plt.show()
'''

#Traning and Testing Of dataset
x_train,x_test,y_train,y_test= train_test_split(x,y,random_state=0,test_size=0.3)
rf.fit(x_train,y_train)

# Predection and Error
y_pred = rf.predict(x_test)
er = mean_squared_error(y_test,y_pred,squared=False)
print("Mean Squre Error: ",er)