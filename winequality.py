import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
wine = pd.read_csv('winequalityN.csv')
wine.head()
pred_test = wine.iloc[3]
pred_test['type'] = 1
pred_test.drop(['quality','total sulfur dioxide'],inplace=True)
#pred_test.drop('total_sulfur_dioxide',inplace=True)
print(pred_test)
wine.shape
wine.isnull().sum()
wine.describe()
wine.dropna()
wine.info()
wine['type'].value_counts()
sns.countplot(x="type", data=wine)
wine['type'].value_counts(normalize=True)
#Checking distribution and outlier for each variable
plt.figure(2)
plt.subplot(121)
sns.distplot(wine['alcohol'])
plt.subplot(122)
wine['alcohol'].plot.box(figsize=(15,5))
#repeat this for all the variables and understand the distribution
#bivariate analysis to check quality with all the other variables
plt.figure(figsize=(10,7))
sns.barplot(x='quality',y='alcohol',data=wine)
#Plotting all variables for their distribution and relation
sns.pairplot(wine)
#checking correlation
wine.corr()
#buidling heatmap
plt.figure(figsize=(15,10))
sns.heatmap(wine.corr(), cmap='coolwarm')
#Dropping highly correlated variables - in this case total sulfur dioxide
wine_new = wine.drop('total sulfur dioxide',axis=1)
#Convert categorical value to dummies
wine_ml = pd.get_dummies(wine_new, drop_first=True)
wine_ml.head()
wine_ml.dtypes
wine_ml.dropna(inplace=True)
X = wine_ml.drop('quality',axis=1)
X.isnull().sum()
Y = wine_ml['quality'].apply(lambda y: 1 if y > 7 else 0)
print(Y)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
x_standard = scaler.transform(X)
scaler = StandardScaler()
pred_test = np.asarray(pred_test).reshape(1,-1)
scaler.fit(pred_test)
pred_test_std = scaler.transform(pred_test)
X = x_standard
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=.2,random_state=123)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
y_pred = logreg.predict(X_test)
pred_test_output = logreg.predict(pred_test_std)
pred_test_output
from sklearn.metrics import accuracy_score
print("Logistic regression:",accuracy_score(Y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, Y_train)
rfc_pred = rfc.predict(X_test)

print("Random Forest:",accuracy_score(Y_test, rfc_pred))
'''
type                        1
fixed acidity             7.2
volatile acidity         0.23
citric acid              0.32
residual sugar            8.5
chlorides               0.058
free sulfur dioxide      47.0
density                0.9956
pH                       3.19
sulphates                 0.4
alcohol                   9.9
Name: 3, dtype: object
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6497 entries, 0 to 6496
Data columns (total 13 columns):
 #   Column                Non-Null Count  Dtype  
---  ------                --------------  -----  
 0   type                  6497 non-null   object 
 1   fixed acidity         6487 non-null   float64
 2   volatile acidity      6489 non-null   float64
 3   citric acid           6494 non-null   float64
 4   residual sugar        6495 non-null   float64
 5   chlorides             6495 non-null   float64
 6   free sulfur dioxide   6497 non-null   float64
 7   total sulfur dioxide  6497 non-null   float64
 8   density               6497 non-null   float64
 9   pH                    6488 non-null   float64
 10  sulphates             6493 non-null   float64
 11  alcohol               6497 non-null   float64
 12  quality               6497 non-null   int64  
dtypes: float64(11), int64(1), object(1)
memory usage: 660.0+ KB
0       0
1       0
2       0
3       0
4       0
       ..
6491    0
6492    0
6494    0
6495    0
6496    0
Name: quality, Length: 6463, dtype: int64
Logistic regression: 0.9682907965970611
Random Forest: 0.974477958236659'''