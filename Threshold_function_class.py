import numpy as np
import matplotlib.pyplot as plt


class ThresholdFunction:

    DomainDimension = 0
    DomainRings = None
    ImageDimension = None
    LinFormCoeff = []
    Borders = []

    MaxLinForm = 0
    MinLinForm = 0
    __F = None
    Grid = None

    def __init__(self, dimension_im, rings_dim=None, coefficients=None, borders=None, cube=None):
        if cube is None:
            self.DomainRings = np.asarray(rings_dim)
        else:
            self.DomainRings = np.asarray([cube[0]] * cube[1])
        self.ImageDimension = dimension_im
        self.DomainDimension = np.prod(self.DomainRings)

        if coefficients is None:
            self.LinFormCoeff = np.random.randint(-10000, 10000, len(self.DomainRings))
        else:
            self.LinFormCoeff = np.asarray(coefficients)

        self.MaxLinForm = np.dot(self.DomainRings - 1, np.where(self.LinFormCoeff < 0, 0, self.LinFormCoeff))
        self.MinLinForm = np.dot(self.DomainRings - 1, np.where(self.LinFormCoeff > 0, 0, self.LinFormCoeff))

        if borders is None:
            self.Borders = np.random.randint(self.MinLinForm, self.MaxLinForm + 1, self.ImageDimension - 1)
        else:
            self.Borders = np.asarray(borders)
        self.Borders = np.sort(self.Borders)

        self.Grid = np.indices(self.DomainRings).reshape((len(self.DomainRings), -1)).T
        self.__F = self.get_value(self.Grid).reshape(self.DomainRings)

    def show_options(self):
        print("LinFormCoeff = ", self.LinFormCoeff)
        print("DomainRings = ", self.DomainRings)
        print("DomainDimension = ", self.DomainDimension)
        print("MaxLinForm = ", self.MaxLinForm)
        print("MinLinForm = ", self.MinLinForm)
        print("Borders = ", self.Borders)
        print("ImageDimension = ", self.ImageDimension)
        print("F = \n", self.__F)

    def normalization(self):
        eps = np.max(np.abs(self.LinFormCoeff))
        self.LinFormCoeff = 1.*self.LinFormCoeff / eps
        self.Borders = 1.*self.Borders / eps
        self.MaxLinForm /= eps
        self.MinLinForm /= eps

    def get_value(self, x):
        # На вход: набор векторов, на выходе: набор значений фунцкции в кольце
        scalar = np.dot(x, self.LinFormCoeff)
        R = np.zeros(scalar.shape)
        for i, border in enumerate(self.Borders):
            R[scalar >= border] = i + 1
        return R

    def calculate_increase_coeff(self, i):
        sum = 0
        k = self.DomainRings[i]
        tempGrid = self.Grid.copy()
        for l in range(0, k-2 + 1):
            tempGrid[:, i] = l
            R = self.__F[tuple(zip(*tempGrid))]
            for e in range(l+1, k-1 + 1):
                tempGrid[:, i] = e
                sum += self.__F[tuple(zip(*tempGrid))] - R
        return np.sum(sum)/k

    def increase_coeff(self, i):
        sum = 0
        k = self.DomainRings[i]
        tempGrid = self.Grid.copy()
        for e in range(0, k-1 + 1):
            tempGrid[:, i] = e
            sum += (2*e + 1 - k) * self.__F[tuple(zip(*tempGrid))]
        return np.sum(sum)/k

    def geometric_algoritm(self):
        pass

    def check(self):
        l = list(map(lambda i: self.calculate_increase_coeff(i), range(len(self.DomainRings))))
        F = list(map(lambda i: self.Grid[self.get_value(self.Grid) == i], range(self.ImageDimension)))
        F = [F[i] for i in range(len(F)) if len(F[i]) != 0]
        maxF=[]
        minF=[]
        argMax=[]
        argMin=[]
        for i in range(0, len(F)):
            maxF.append(np.max(np.dot(F[i], l)))
            argMax.append(np.argmax(np.dot(F[i], l)))
            minF.append(np.min(np.dot(F[i], l)))
            argMin.append(np.argmin(np.dot(F[i], l)))
        for i in range(1,len(F)-1):
            if minF[i+1] <= maxF[i]:
                # print("maxF[", i, "] =", maxF[i], ">=", minF[i+1], "= minF[", i+1, "]")
                # print(argMax)
                # print(argMin)
                return False
        return True

    @staticmethod
    def correction(self, l, u, v):
        l = l - u + v
        return l

    def draw(self):
        colors = "rbgcmykw"
        F = list(map(lambda i: self.Grid[self.get_value(self.Grid) == i], range(self.ImageDimension)))
        for i in range(0,len(F)):
            plt.plot(F[i][:,0],F[i][:,1],'o', color=colors[i%len(F)])
        plt.show()


t1 = ThresholdFunction(5, cube=(5, 2))
while (t1.check() == True):
    t1 = ThresholdFunction(4, cube=(10, 2))
    # t1 = ThresholdFunction(5, cube=(5, 5), coefficients=(1, -25, 14, -43, 43), borders=(-164, -69, 53, 110))
    # t1 = ThresholdFunction(np.random.randint(2, 100),np.random.randint(2, 10, np.random.randint(1, 5)) )

print(t1.show_options())
print(list(map(lambda i: t1.calculate_increase_coeff(i), range(len(t1.DomainRings)))))
t1.draw()
    # answer = t1.get_value(t1.Grid).reshape(t1.DomainDimension)
    # a = [(0, 0, 1, 0), (0, 0, 0, 1)]
    # print("d", answer[tuple(zip(*a))])
    # print(t1.Grid[8])
    # t1.normalization()
    # t1.show_options()

