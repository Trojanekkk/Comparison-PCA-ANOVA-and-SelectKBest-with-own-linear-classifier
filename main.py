import numpy as np
from sklearn.preprocessing import StandardScaler
from linear_classifier import linear_classifier

def accuracy (y, y_pred):
    if (len(y) != len(y_pred)):
        raise ValueError("Incorrect size of y")

    points = 0
    for index, label in enumerate(y_pred):
        if label == y[index]:
            points = points + 1 / len(y_pred)
    
    return points

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
y_pred = lc.predict(X)
score = accuracy(y, y_pred)
print(score)

# Przeprowadzenie PCA, ANOVA
# Walidacja krzyżowa
# Obliczenie jakości
# Porównanie jakości dla PCA, ANOVA (testy parowe, testy globalne na 7 rangach)