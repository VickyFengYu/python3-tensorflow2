# -*- coding: utf-8 -*-
"""
Created on Fri May  4 16:25:39 2018

@author: Frank
"""

from KnnContentBased.MovieLens import MovieLens
# from ContentKNNAlgorithm import ContentKNNAlgorithm
# from Evaluator import Evaluator

import random
import numpy as np


def LoadMovieLensData():
    ml = MovieLens()
    print("Loading movie ratings...")
    data = ml.loadMovieLensLatestSmall()
    print("\n")
    print("Computing movie popularity ranks...")
    rankings = ml.getPopularityRanks()
    return ml, data, rankings


np.random.seed(0)
random.seed(0)

# Load up common data set
(ml, evaluationData, rankings) = LoadMovieLensData()
