from Threshold_function_class import ThresholdFunction as TFunc
import numpy as np

c = 0
g = 0
while c < 100:
    t1 = TFunc(np.random.randint(2, 10), np.random.randint(3, 15, size=2))

    # t1.show_options()
    if t1.check():
        t1.write_options("infoGOOD " + str(g))
        t1.draw2d("GOOD " + str(g))
        g += 1
        # t1.draw2d()

    else:
        t1.write_options("infoBAD " + str(c))
        t1.draw2d("BAD " + str(c))
        # t1.draw2d()
        c += 1
print("complete")
