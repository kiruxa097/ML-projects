from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Порядковые данные (размер одежды: S < M < L)
data = pd.DataFrame({
    'size': ['S', 'M', 'L', 'M', 'S', 'L'],
    'price': [100, 150, 120, 110, 130, 140]
})

encoder = LabelEncoder()
data["size_encoded"] = encoder.fit_transform(data["size"])

print("Исходный:")
print(data)
print(f"\nПорядок кодирования: S-{encoder.transform(['S'])[0]}, M-{encoder.transform(['M'])[0]}, L-{encoder.transform(['L'])[0]}")