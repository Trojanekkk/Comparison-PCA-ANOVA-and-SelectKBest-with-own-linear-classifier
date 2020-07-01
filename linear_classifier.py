import numpy as np
import math

class linear_classifier:
    # Zainicjowanie obiektu
    def __init__ (self):
        pass

    # Uczenie
    def fit (self, X, y):
        # Sprawdzenie wymiarów zbiorów
        if (len(X) != len(y)):
            raise ValueError("Incorrect size of X and y")

        self.X = X
        self.y = y

        # Ekstrakcja unikatowych klas
        self.uniqueClasses = []
        for label in self.y:
            if label not in self.uniqueClasses:
                self.uniqueClasses.append(label)
        if (len(self.uniqueClasses) != 2):
            raise ValueError("Classifier is for datasets with 2-classes only")

        # center = []
        # for uniqueClass in classes:
        #     for i in range(0, len(X[0])):
        #         sum = 0
        #         print()
        #         for j in range(0, len(X)):
        #             if y[j] == uniqueClass:  
        #                 print(X[j][i])
        #                 sum += X[j][i]
        #         center.append(sum/)
            
        # print(center)

        # Obliczenie centroidów dla każdej z klas
        self.centroids = []
        for cls in self.uniqueClasses:
            tmp = []
            for index, element in enumerate(self.X):
                if (self.y[index] == cls):
                    tmp.append(element)
            self.centroids.append(np.mean(tmp, axis=0))
        
        return self

    # Testowanie
    def predict(self, X):
        # Weryfikacja czy wykonano proces uczenia
        if not self.centroids:
            raise ValueError("Complete the 'Fit' function first")

        # Sprawdzenie rozmiaru zbioru uczącego
        for i in X:
            if (len(i) != len(self.centroids[0])):
                raise ValueError("Incorrect size of X")

        # Predykcja poprzez obliczanie odległości od centroidów każdej z klas
        y_pred = []
        for i in range(len(X)):
            d = []
            for cls_i, cls in enumerate(self.uniqueClasses):
                sqr_d = 0
                for j in range(len(X[0])):
                    sqr_d += (X[i][j] - self.centroids[cls_i][j]) ** 2
                d.append(math.sqrt(sqr_d))
            y_pred.append(d.index(min(d)))

        return y_pred


                


        

