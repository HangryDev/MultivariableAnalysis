# Py 9.1
import numpy as np
import pandas as pd
# 데이터 읽기
titanic = pd.read_csv("c:/data/mva/titanic.csv")
titanic.head(3)

#기술통계량 구하기
titanic.describe()
# 자료 (행의 수, 열의 수)
titanic.shape

# Py 9.2
# 빈도표 구하기
pd.crosstab(titanic['Survived'], titanic['Sex'], margins=True) 
pd.crosstab(titanic['Survived'], titanic['Class'], margins=True)
pd.crosstab(titanic['Survived'], titanic['Age'], margins=True)

# Py 9.3
# 문자형을 이산형으로 변환
titanic['Age'] = titanic['Age'].replace({'Child':0, 'Adult':1})
titanic['Sex'] = titanic['Sex'].replace({'Male':0, 'Female':1})
titanic['Class'] = titanic['Class'].replace({'First':1, 'Second':2,  'Third':3, 'Crew':4})
titanic.head(2)

# X 데이터와 y 데이터
X = titanic[["Class","Age","Sex"]]
y = titanic["Survived"]

# Py 9.4
# 나무모형 생성
from sklearn.tree import DecisionTreeClassifier
titanic_tree = DecisionTreeClassifier(max_depth=3, min_samples_split=50)
titanic_tree.fit(X, y)
# 적합된 나무모형을 이용한 분류
y_pred = titanic_tree.predict(X)
# Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y, y_pred)
print(cm)

# 기타 분류 성능 지표
cm_report = classification_report(y, y_pred)
print(cm_report)


# Py 9.5
# Tree 그리기
from sklearn.tree import plot_tree
plt.figure(figsize=(11,7))
plot_tree(titanic_tree, feature_names=X.columns,
 class_names=['No','Yes'], filled=True, fontsize=9)
plt.show()


# Py 9.6
import numpy as np
import pandas as pd
# 데이터 읽기
cu = pd.read_csv("c:/data/mva/cusummary.csv", index_col='Model')
cu.head()
# 자료 (행의 수, 열의 수)
cu.shape
# 결측값 케이스 없애기
cu = cu.dropna()
cu.shape

# Py 9.7
# X 데이터와 y 데이터
X = cu.drop('Price', axis=1)
y = cu['Price']
# 이산변수 혹은 가변수(dummy variable) 만들기
X['Reliability'] = X['Reliability'].replace({'Much worse':1, 'worse':2,
 'average':3, 'better':4, 'Much better':5})
dX = pd.get_dummies(data=X, drop_first=True)
dX.head()
# 변수 이름 보기
dX.columns

# Py 9.8
# 회귀나무모형 생성
from sklearn.tree import DecisionTreeRegressor
cu_tree = DecisionTreeRegressor(max_depth=3, min_samples_split=15)
cu_tree.fit(dX, y)
# 추정값 구하기
y_pred = cu_tree.predict(dX)
y_pred[0:5]

# Tree 그리기
from sklearn.tree import plot_tree
plt.figure(figsize=(11,7))
plot_tree(cu_tree, feature_names=dX.columns, filled=True, fontsize=9)
plt.show()