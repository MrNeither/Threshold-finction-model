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


        # Todo Правильно почтать минимальную и максимальную линейные формы
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
        print(self.__F)
        # self.F.update({float("-inf"): self.ImageDimension - 1})

    def show_options(self):
        print("LinFormCoeff = ", self.LinFormCoeff)
        print("DomainDimension = ", self.DomainDimension)
        print("MaxLinForm = ", self.MaxLinForm)
        print("MinLinForm = ", self.MinLinForm)
        print("Borders = ", self.Borders)
        print("ImageDimension = ", self.ImageDimension)
        print("F = ", self.BorderList)

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




for i in range(1):
    # t1 = ThresholdFunction(7, np.random.randint(2, 150, np.random.randint(2, 100, 1)))
    # t1 = ThresholdFunction(7, cube=(3, 3))
    t1 = ThresholdFunction(7, (2, 2, 2, 2))
    t1.show_options()

    answer = t1.get_value(t1.Grid).reshape(t1.DomainDimension)
    a = [(0,0,1,0),(0,0,0,1)]
    print("dadsad   ", answer[tuple(zip(*a))])
    # t1.normalization()
    # t1.show_options()

