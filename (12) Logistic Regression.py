# Py 8.1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 데이터 읽기
mower = pd.read_csv("c:/data/mva/mower.csv")
mower.head()
# 변수 선택
y = mower["owner"]
X = mower[["income", "size"]]
# 로지스틱 회귀분석 실행
from sklearn.linear_model import LogisticRegression
mower_clf = LogisticRegression()
mower_clf.fit(X,y)
# 로지스틱 회귀모형 절편
mower_clf.intercept_
# 로지스틱 회귀모형 계수
mower_clf.coef_
# 분류 클래스
mower_clf.classes_

# Py 8.2
# 두 그룹에 속할 확률
mower_clf.predict_proba(X) [0:7]
# 분류 결과
mower_clf.predict(X)

# 분류표 구하기
from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y, mower_clf.predict(X))
cm
# accuracy 계산하기
pred_class = mower_clf.predict(X)
print('Accuracy = '+str(accuracy_score(y, pred_class)))

# 세분화된 분류표
cm_report = classification_report(y, mower_clf.predict(X))
print(cm_report)

# PY 8.3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 데이터 읽기
mower = pd.read_csv("c:/data/mva/mower.csv")
# 변수 선택
y = mower["owner"]
aX = mower[["income", "size"]]
import statsmodels.api as sm
# 상수 더하기
aX = sm.add_constant(aX)
# array 변환
ay = y.to_numpy()
iy = [0]*len(ay)

for i in range(0, len(ay)) :
    if(ay[i] =='yes') :
       iy[i] = 1
    else :
       iy[i] = 0

# 로지스틱 회귀모형 적합하기
mower_sm = sm.Logit(iy, aX)
mower_logit = mower_sm.fit()
mower_logit.params

# Py 8.4

# 자료의 분류
mower_logit.predict(aX)
mower_pred = (mower_logit.predict(aX) >= 0.5).astype(int)
mower_pred
# 분류표
mower_logit.pred_table()
# 로지스틱 회귀모형 적합 결과
mower_logit.summary()

