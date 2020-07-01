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
column_header = np.array([[reducers[0]], [reducers[1]], [reducers[2]]])

t_statistic_table = np.concatenate((column_header, t_statistic), axis=1)
p_value_table = np.concatenate((column_header, p_value), axis=1)

print("t_statistic\n" + tabulate(t_statistic_table, reducers, floatfmt=".2f") + "\n")
print("p_value\n" + tabulate(p_value_table, reducers, floatfmt=".2f") + "\n")

# Wyznaczenie i wyświetlenie macierzy przewagi
advantage = np.zeros((len(reducers), len(reducers)))
advantage[t_statistic > 0] = 1

print("advantage\n" + tabulate(np.concatenate((column_header, advantage), axis=1), reducers) + "\n")

# Wyznaczenie i wyświetlenie macierzy istotności
significance = np.zeros((len(reducers), len(reducers)))
significance[p_value <= alfa] = 1

print("significance\n" + tabulate(np.concatenate((column_header, significance), axis=1), reducers) + "\n")

stat_better = significance * advantage
print("Statistically significantly better\n" + tabulate(np.concatenate((column_header, stat_better), axis=1), reducers))



