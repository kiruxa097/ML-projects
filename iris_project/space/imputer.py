from sklearn.impute import KNNImputer
import numpy as np

X = [[1, 2, 3],
     [np.nan, 2, 3],
     [1, np.nan, 3],
     [5, 6, 7]]

imputer = KNNImputer(n_neighbors=2)
X_filled = imputer.fit_transform(X)

print("Было:\n", X)
print("\n Стало (после KNNImputer): \n", X_filled)