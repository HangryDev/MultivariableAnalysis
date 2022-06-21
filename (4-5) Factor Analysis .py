# Py 3.1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 데이터 읽기
med = pd.read_csv("c:/data/mva/medFactor.csv")
med.head(3)

#기술통계량 구하기
med.describe()

#py 3.2

#인자분석 적정성 검정 (추가)
from factor.analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value=calculate_bartlett_sphericity(med)
chi_square_value, p_value
from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calcualte_kmo(med)
kmo_model

# 초기 인자분석
from factor_analyzer import FactorAnalyzer
fa = FactorAnalyzer(rotation=None)
# help(FactorAnalyzer): 클래스 코드 보기
fa.fit(med)
# 고윳값 구하기
ev, v = fa.get_eigenvalues()
ev

# 스크리 그림 그리기
plt.scatter(range(1, med.shape[1]+1), ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalues')
plt.grid()
plt.show()

# Py 3.3
# 인자 수를 3으로 한 인자분석 – 인자회전 Varimax
fa_varimax = FactorAnalyzer(n_factors=3, rotation='varimax', method='principal')
fa_varimax.fit(med)
# 인자적재계수
fa_varimax.loadings_

#인자 공통성(communality)
fa_varimax.get_communalities()

#인자고유분산 : 1-공통성
fa_varimax.get_uniqueness()

#인자분산
fa_varimax.get_factor_variance()

# Py 3.4
# Oblimin 인자회전
fa_obm = FactorAnalyzer(n_factors=3, rotation='oblimin', method='principal')
fa_obm.fit(med)
# 인자적재계수
fa_obm.loadings_

# 인자 공통성(communality)
fa_obm.get_communalities()

# 인자고유분산: 1-공통성
fa_obm.get_uniquenesses()

# 인자분산
fa_obm.get_factor_variance()






