import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 읽기
heptathlon = pd.read_csv("c:/data/mva/heptathlon.csv")
heptathlon.head(3)

#변수이름 확인하기
heptathlon.columns
#기술통계량 구하기 - 소수점 이하 2자리 반올림 표시
round(heptathlon.describe(), 2)

# 변환: 변수최댓값 - 변숫값
heptathlon.hurdles = np.max(heptathlon.hurdles) - heptathlon.hurdles
heptathlon.run200m = np.max(heptathlon.run200m) - heptathlon.run200m
heptathlon.run800m = np.max(heptathlon.run800m) - heptathlon.run800m
heptathlon.head()

# 분석변수 선택하기
feature = ['hurdles','highjump','shot','run200m','longjump','javelin','run800m']
hep_data = heptathlon[feature]

from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(hep_data)

pca_init = PCA(n_components = len(hep_data.columns))
pca_init.fit(x)
pca_init.explained_variance_
np.cumsum(pca_init.explained_variance_ratio)

#스크리 그림 그리기
plt.figure()
plt.subplot(121)

plt.title('Scree Plot')
plt.xlabel('Components')
plt.ylabel('Explained Variance')
plt.plot(pca_init.explained_variance_, 'o-')
plt.subplot(122)
plt.plot(np.cumsum(pca_init.explained_variance_ratio_), 'o-')
plt.title('Cumulative Scree Plot')
plt.xlabel('Components')
plt.ylabel('Explained Variance Ratio')
plt.show()

# Py 2.4
# 주성분분석 – 주성분 수 2개 추출
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
hep_pca = pca.fit_transform(x)
# dir(pca)
# 주성분분산
pca.explained_variance_
# 주성분분산 비율
pca.explained_variance_ratio_

# 주성분계수
np.round(pca.components_, 3)
# 주성분점수
hep_pca[0:5,:]

# Py 2.5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 데이터 읽기
beer = pd.read_csv("c:/data/mva/beer.csv")
beer.head()

# 기술통계량 구하기
beer.describe()

# Py 2.6
# 주성분분석 – 주성분 수 3으로 함
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca_beer = pca.fit_transform(beer)

# 주성분분산
pca.explained_variance_

# 주성분 표준편차
np.sqrt(pca.explained_variance_)

# 주성분분산 비율
pca.explained_variance_ratio_

# 주성분계수
np.round(pca.components_, 3)



