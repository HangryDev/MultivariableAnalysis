# Py 7.1
import numpy as np
import pandas as pd
# 데이터 읽기
alcohol = pd.read_csv("c:/data/mva/alcohol.csv")
alcohol.head()
# 기술통계량 구하기
alcohol.describe()

# Py 7.2
# 변수 선택
X = alcohol.iloc[:, 1:]
y = alcohol["TYPE"]
X.head()
y.head()

# 선형판별분석 실행
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis
clf.fit(X, y)
clf.fit_transform(X, y)

# Py 7.3
# 분류하기
pred_class = clf.predict(X)
pred_class[0:5]
# 사후확률(posterior prob) 구하기
pred_posterior = clf.predic_proba
pred_posterior[0:5, :]

# Py 7.4
# 분류표 구하기
from sklearn.metrics import confusion_matrix
confusion_matrix(y, pred_class)

# 오분류율 구하기
from sklearn.metrics import accuracy_score
print('Accuracy = '+str(accuracy_score(y, pred_class)))
print('Error rate = '+str(1-accuracy_score(y, pred_class)))

