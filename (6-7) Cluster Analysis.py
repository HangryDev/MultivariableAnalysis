import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 데이터 읽기
beer = pd.read_csv("c:/data/mva/beerbrand.csv", index_col='name')
beer.head()

# 기술통계량 구하기
beer.describe()

#표준화 패키지 불러오기
from sklearn.preprocessing import StandardScaler
#표준화 시행
zbeer = StandardScaler().fit_transform(beer)

 Py 4.2
# 패키지 불러오기
import scipy.cluster.hierarchy as sch
# 계층적 군집분석 시행하기: 최단연결법
slink = sch.linkage(zbeer, 'single')
# method = 'single', 'complete', 'average', 'median', 'ward'
# 덴드로그램 그리기
plt.figure(figsize=(7,5))
sch.dendrogram(slink,  leaf_rotation=80,  leaf_font_size=10,  labels = beer.index)
plt.title("Dendrogram of Single linkage")
plt.show()

# Py 4.3
# 계층적 군집분석 시행: 와드의 방법
wlink = sch.linkage(zbeer, 'ward')
plt.figure(figsize=(7,5))
sch.dendrogram(wlink,  leaf_rotation=80,  leaf_font_size=10,  labels=beer.index )
plt.title("Dendrogram of Ward's method")
plt.show()

# Py 4.4# 계층적 군집분석: 중심연결법
clink = sch.linkage(zbeer, 'centroid')
# 덴드로그램 그리기
# 덴드로그램 그리기
plt.figure(figsize=(7,5))
sch.dendrogram(clink,  leaf_rotation=80,  leaf_font_size=10,  labels = beer.index )
plt.title("Dendrogram of Centroid linkage")
plt.show() 

#py 4.5
from sklearn.cluster import AgglomerativeClustering
wcluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean',  linkage='ward')
# 소속군집
member = wcluster.fit_predict(zbeer)
member

# 군집별 평균계산
member1 = pd.DataFrame(member, columns=['cluster'], index=beer.index)
data_combined = beer.join(member1)
data_combined.groupby('cluster').mean()

# Py 4.6
# K-means 군집분석
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# 표준화
zbeer = StandardScaler().fit_transform(beer)
# k-평균 군집분석: 군집수 = 2
kmc = KMeans(n_clusters=2)
kmc.fit(zbeer)

# Py 4.6
# K-means 군집분석
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# 표준화
zbeer = StandardScaler().fit_transform(beer)
# k-평균 군집분석: 군집수 = 2
kmc = KMeans(n_clusters=2)
kmc.fit(zbeer)

# 군집 중심 알기
kmc.cluster_centers_
# 소속군집 알기
kmc.labels_

# Py 4.7
# 소속 군집 산점도
plt.figure(figsize=(5,5))
plt.scatter(x=beer['calories'], y=beer['sodium'], c=kmc.labels_)
plt.show()




