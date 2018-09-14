import numpy as np
g1=np.load('/Users/ben_work/g1.npy');ll1=np.load('/Users/ben_work/ll1.npy');al1=np.load('/Users/ben_work/al1.npy');l1=np.load('/Users/ben_work/l1.npy');a1=np.load('/Users/ben_work/a1.npy');c1=np.load('/Users/ben_work/c1.npy')

import matplotlib.pyplot as plt


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.

    Rows are scores for each class.
    Columns are predictions (samples).
    """
    scoreMatExp = np.exp(np.asarray(x))
    return scoreMatExp / scoreMatExp.sum(1).reshape(81,-1)

plt.imshow(softmax(a1[:,0,:]).T,aspect='auto')
b1=softmax(a1[:,0,:])
plt.colorbar()
plt.show()