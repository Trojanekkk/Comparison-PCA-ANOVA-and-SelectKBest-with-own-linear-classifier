import numpy as np
from sklearn.preprocessing import StandardScaler
from linear_classifier import linear_classifier

# Importowanie zbioru danych, wydzielenie etykiet i wzorców
dataset = np.genfromtxt("datasets/australian.csv", delimiter=",")
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)

# Przeporwadzenie standaryzacji zbioru
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Inicjalizowanie nowego obiektu
lc = linear_classifier()

# Uczenie
lc.fit(X, y)

# Predykcja
print(lc.predict(X))

# Przeprowadzenie PCA, ANOVA
# Walidacja krzyżowa
# Obliczenie jakości
# Porównanie jakości dla PCA, ANOVA (testy parowe, testy globalne na 7 rangach)