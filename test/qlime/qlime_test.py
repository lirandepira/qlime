from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from unittest import TestCase

from qlime.model import qnn
from qlime.optimizer import spsa
from qlime.optimizer import objective
from qlime.qlime import explain

class QlimeTesting(TestCase):
     def test_explain(self):
        # Load the Iris dataset
        iris = load_iris()

        # Normalize the data by subtracting the minimum value and dividing by the maximum value
        iris.data = (iris.data-iris.data.min())/iris.data.max()
        # Use the first two classes of the Iris dataset only (100 of 150 data points)
        X_train, X_test, Y_train, Y_test = train_test_split(
            iris.data[0:100,[0,1]],
            iris.target[0:100],
            test_size=0.75,
            random_state=0)

        a = 2.5
        c = 0.25
        maxiter = 1e3
        shots = int(1e4)

        alpha = 0.602
        gamma = 0.101

        f = lambda x: objective(x,X_train,Y_train,shots=shots) # Objective function
        x0 = np.pi*np.random.randn(4)

        xsol = spsa(f, x0, a=a, c=c,
                    alpha=alpha, gamma=gamma,
                    maxiter=maxiter, verbose=True)

        # Explain the local behavior of the model
        explain(1,X_train,lambda x : qnn(x,xsol))

