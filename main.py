import Threshold_function_class as TFunc
import numpy as np

bad = 0
good = 0
while good < 0 and bad < 0:
    t1 = TFunc.ThresholdFunction(np.random.randint(2, 14), np.random.randint(3, 10, size=np.random.randint(2, 7)))
    t1.normalization()
    if t1.check():
        t1.setMethodType('new')
        if t1.check():
            pass
        else:
            print('BAD')
        # t1.write_options("infoGOOD " + str(good))
        # t1.draw2d("GOOD " + str(good))
        good += 1
        t1.normalization()
        # t1.draw2d()

    else:
        t1.setMethodType('new')
        if t1.check():
            print('GOOD')
        # t1.write_options("infoBAD " + str(bad))
        # t1.draw2d("BAD " + str(bad))
        # t1.draw2d()
        t1.normalization()
        bad += 1

t1 = TFunc.ThresholdFunction3D(np.random.randint(2, 14), np.random.randint(3, 10, size=3))
# t1 = TFunc.ThresholdFunction3D(2, (2, 2, 2))
t1.check()
t1.draw3d('demo.png')
print("complete")

