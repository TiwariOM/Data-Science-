import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB

logr=LogisticRegression()
rfclass=RandomForestClassifier()
gbclass=GradientBoostingClassifier(n_estimators=10)
dtclass=DecisionTreeClassifier(random_state=0)
sv=svm.SVC()
nn=MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(7,8), random_state=0)
nb=MultinomialNB()

df=pd.read_csv("IRIS.csv")
# print(df)

X=df.drop("species",axis=1)       #specifying X-coordinates i.e independent variables
# X=X.drop("Id",axis=1)    Use this if ID is specified
print(X)
Y=df["species"]                   #specifying Y-coordinates i.e targeted value
print(Y)

X_train, X_test, Y_train, Y_test=train_test_split(X,Y,random_state=0,test_size=0.3)

# print(X_train)
# print(X_test)
# print(Y_train)
# print(Y_test)

train1=logr.fit(X_train,Y_train)
train2=rfclass.fit(X_train,Y_train)
train3=gbclass.fit(X_train,Y_train)
train4=dtclass.fit(X_train,Y_train)
train5=sv.fit(X_train,Y_train)
train6=nn.fit(X_train,Y_train)
train7=nb.fit(X_train,Y_train)

Y_pred1=logr.predict(X_test)
Y_pred2=rfclass.predict(X_test)
Y_pred3=gbclass.predict(X_test)
Y_pred4=dtclass.predict(X_test)
Y_pred5=sv.predict(X_test)
Y_pred6=nn.predict(X_test)
Y_pred7=nb.predict(X_test)

# print(Y_pred+ "   "+Y_test)

print("The accuracy score for logistic regression:")
print(accuracy_score(Y_pred1,Y_test))
print("The accuracy score for Random Forest Classifier:")
print(accuracy_score(Y_pred2,Y_test))
print("The accuracy score for Gradient Boosting Classifier:")
print(accuracy_score(Y_pred3,Y_test))
print("The accuracy score for Decision Tree Classifier:")
print(accuracy_score(Y_pred4,Y_test))
print("The accuracy score for Support Vector Machine :")
print(accuracy_score(Y_pred5,Y_test))
print("The accuracy score for Neural Network:")
print(accuracy_score(Y_pred6,Y_test))
print("The accuracy score for Naive bayes:")
print(accuracy_score(Y_pred7,Y_test))
'''The accuracy score for logistic regression:
0.9777777777777777
The accuracy score for Random Forest Classifier:
0.9777777777777777
The accuracy score for Gradient Boosting Classifier:
0.9777777777777777
The accuracy score for Decision Tree Classifier:
0.9777777777777777
The accuracy score for Support Vector Machine :
0.9777777777777777
The accuracy score for Neural Network:
0.9555555555555556
The accuracy score for Naive bayes:
0.6
'''