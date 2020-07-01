import numpy as np
from scipy.stats import ttest_ind
from tabulate import tabulate

# Wczytanie wyników eksperymentu
scores = np.load("results.npy")

# Lista nazw testowanych metod redukcji danych
reducers = ["PCA", "ANOVA", "SelectKBest"]

# Wstępne parametry dla statystyki T Studenta
alfa = 0.05
t_statistic = np.zeros((len(reducers), len(reducers)))
p_value = np.zeros((len(reducers), len(reducers)))

# Obliczenie wartości statystyki
for i in range(len(reducers)):
    for j in range(len(reducers)):
        t_statistic[i][j], p_value[i][j] = ttest_ind(scores[i], scores[j])

# Utworzenie kolumny nazw testowanych metod, sformatowanie i wyświetlenie wyników
column = np.array([[reducers[0]], [reducers[1]], [reducers[2]]])

t_statistic_table = np.concatenate((column, t_statistic), axis=1)
p_value_table = np.concatenate((column, p_value), axis=1)

print(tabulate(t_statistic_table, reducers, floatfmt=".2f") + "\n")
print(tabulate(p_value_table, reducers, floatfmt=".2f"))