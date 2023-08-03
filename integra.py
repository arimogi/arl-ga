#!/bin/python

import random
import pandas as pd
import math
from scipy.stats import norm
import numpy as np
from scipy.integrate import quad

def integrand(x, a, b):
    return a*x**2 + b

a = 2
b = 1
I = quad(integrand, 0, 1, args=(a, b))
print(I)