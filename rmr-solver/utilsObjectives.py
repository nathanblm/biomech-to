"""
Author:             FJ van Melis
Created on:         October 18th 2024.
Last updated on:    December 4th 2024.

PURPOSE:
Class setting the objective for the optimization problem.

USAGE:
See runRMRsolver().py for an example on how to use this class!
"""

from typing import Any, List, Dict, Tuple, Optional, Union
import numpy as np
import numpy.typing as npt


class ActSquared:
    def __init__(self, actuatorWeights: npt.ArrayLike) -> None:
        """
        Sets up an objective function based on minimization of activation squared.

        :Parameters:
        actuatorWeights: ``array_like`` |
            Weights corresponding to the actuators of the model.
        """
        self.weight = actuatorWeights

    def __call__(self, x: npt.ArrayLike, *args) -> Any:
        """ 
        Callable cost function for scipy.minimize() in RMR solver.
        """

        cost = self.weight.dot(np.square(x))

        return cost
    
    def getWeight(self) -> npt.NDArray:
        return self.weight
