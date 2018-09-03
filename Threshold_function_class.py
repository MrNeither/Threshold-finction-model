import numpy as np


class ThresholdFunction:

    DomainDimension = None
    ImageDimension = None
    LinFormCoeff = []
    Borders = []

    MaxLinForm = 0
    MinLinForm = 0
    BorderList = None
    __F = None
    Grid = None

    def __init__(self, dimension_im, rings_dim=None, coefficients=None, borders=None, cube=None):
        if cube is None:
            self.DomainDimension = np.asarray(rings_dim)
        else:
            self.DomainDimension = np.asarray([cube[0]] * cube[1])
        self.ImageDimension = dimension_im

        if coefficients is None:
            self.LinFormCoeff = np.random.randint(-10000, 10000, len(self.DomainDimension))
        else:
            self.LinFormCoeff = np.asarray(coefficients)

        self.MaxLinForm = np.dot(self.DomainDimension - 1, np.where(self.LinFormCoeff < 0, 0, self.LinFormCoeff))
        self.MinLinForm = np.dot(self.DomainDimension - 1, np.where(self.LinFormCoeff > 0, 0, self.LinFormCoeff))

        if borders is None:
            self.Borders = np.random.randint(self.MinLinForm, self.MaxLinForm + 1, self.ImageDimension - 1)
        else:
            self.Borders = np.asarray(borders)
        self.Borders = np.sort(self.Borders)
        self.BorderList = {self.Borders[i]: i for i in range(self.ImageDimension - 1)}

        self.Grid = np.indices(self.DomainDimension).reshape((len(self.DomainDimension), -1)).T
        self.__F = self.get_value(self.Grid).reshape(self.DomainDimension)
        # self.F.update({float("-inf"): self.ImageDimension - 1})

    def show_options(self):
        print("LinFormCoeff = ", self.LinFormCoeff)
        print("DomainDimension = ", self.DomainDimension)
        print("MaxLinForm = ", self.MaxLinForm)
        print("MinLinForm = ", self.MinLinForm)
        print("Borders = ", self.Borders)
        print("ImageDimension = ", self.ImageDimension)
        print("F = ", self.__F)

    def normalization(self):
        # Todo Придумать нормализацию, в которой среднее между max и min будет равно 0.5
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
            R[scalar > border] = i + 1
        return R

    def correction(self, u, v):
        self.LinFormCoeff = self.LinFormCoeff + u - v
        return self.LinFormCoeff

    def calculate_increase_coeff(self, i):
        sum = 0
        k = self.DomainDimension[i]
        tempGrid = self.Grid.copy()
        for l in range(0, k-2 + 1):
            tempGrid[:, i] = l
            R = self.__F[tuple(zip(*tempGrid))]
            for e in range(l+1, k-1 + 1):
                tempGrid[:, i] = e
                sum += self.__F[tuple(zip(*tempGrid))] - R
        # return np.sum(sum)/k
        return sum/k

    def increase_coeff(self, i):
        sum = 0
        k = self.DomainDimension[i]
        tempGrid = self.Grid.copy()
        for e in range(0, k-1 + 1):
            tempGrid[:, i] = e
            sum += (2*e + 1 - k) * self.__F[tuple(zip(*tempGrid))]
        # return np.sum(sum)/k
        return sum/k

    def RF(self,x):
        return self.__F[tuple(zip(*x))]



for i in range(1):
    t1 = ThresholdFunction(5, cube=(5, 5), coefficients=(1, -25, 14, -43, 43), borders=(-164, -69, 53, 110))
    # t1 = ThresholdFunction(np.random.randint(2, 100),np.random.randint(2, 10, np.random.randint(1, 5)) )
    # t1.show_options()
    # t1 = ThresholdFunction(5, cube=(5, 2))
    print(i)
    t1.show_options()
    print(t1.calculate_increase_coeff(0))
    print(t1.calculate_increase_coeff(1))
    # t1.correction((1,0,0,0),(0,0,0,1))

    print(np.prod(t1.DomainDimension))
    # answer = t1.get_value(t1.Grid).reshape(t1.DomainDimension)
    # a = [(0, 0, 1, 0), (0, 0, 0, 1)]
    # print("d", answer[tuple(zip(*a))])
    # print(t1.Grid[8])
    # t1.normalization()
    # t1.show_options()

