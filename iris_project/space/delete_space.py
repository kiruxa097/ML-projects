import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

X = [[1,2], [np.nan, 3], [4, 1]]
y = [1, 0, 1]

"""
Проблема данных данных:
Пусть X - массив признаков каких-либо входных данных,
y - массив ответов (значений)
Если к этим данным применить логистическую регрессию, то данная модель выдаст ошибку
Решение (на примере датасета: ирис):
"""

from sklearn.datasets import load_iris

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Искусственно создадим проблему, заполнив некоторые признаки пропусками:

df.iloc[0, 0] = np.nan
df.iloc[5, 2] = np.nan

print(f"Кол-во признаков с пропусками до удаления: {len(df)}")

# Первый способ решения проблемы пропусков - удаление этих пропусков:

df_clean = df.dropna()

print(f"Кол-во признаков после удаления признаков с nan: {len(df_clean)}")