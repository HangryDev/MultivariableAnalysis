# Py 5.1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 데이터 읽기
auto = pd.read_csv("c:/data/mva/auto.csv")
auto.head()

# Py 5.2
# 변수 선택
X = auto.iloc[:, 1:]

# 각 케이스 이름
autoName = auto["autoName"]
autoName = list(autoName)

# z-표준화
from sklearn.preprocessing import StandardScaler
zX = StandardScaler().fit_transform(X)


# 0-1 변환
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
z01X = scaler.fit_transform(X)

#거리행렬 구하기
from sklearn.metrics import pairwise_distances
z01X_dist = pairwise_distances(z01X, metric='euclidean')
z01X_dist.shape

#Py 5.3 
#MDS 실행
from sklearn.manifold import MDS
cmds = MDS(n_components=2, random_state=0, dissimilarity = 'precomuted')
mds1 = cmds.fit(z01X_dist)

# 그림 그리기
plt.figure()
plt.scatter(mds1_coords[:,0], mds1_coords[:,1], facecolors='none', edgecolors='none')
labels = autoName

# MDS 각 케이스에 라벨 붙이기
for label, x, y in zip (labels, mds1_coords[:,0], mds1_coords[:,1]) :
 plt.annotate(label, (x,y), xycoords='data')
plt.xlabel('First dimension')
plt.ylabel('Second dimension')
plt.title('Metric MDS')
plt.show()
