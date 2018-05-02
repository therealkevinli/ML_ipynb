'''

from IPython.display import display, HTML
web = HTML('<iframe src = http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data width=300 height = 200><iframe>')
display(HTML('<iframe src = http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data width=300 height = 200><iframe>'))
display(web)
print("hi")
'''

import numpy as np
X = np.array([[1., 0.], [2., 1.], [0., 0.]])
y = np.array([0, 1, 2])
from scipy.sparse import coo_matrix
X_sparse = coo_matrix(X)
from sklearn.utils import shuffle
(X, X_sparse, y) = shuffle(X, X_sparse, y, random_state=0)
print(X)

print("this is X_sparse:",X_sparse)
'''
I think the output looks like this
array([[ 0.,  0.],
       [ 2.,  1.],
       [ 1.,  0.]])
'''
