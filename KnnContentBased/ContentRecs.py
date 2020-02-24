# -*- coding: utf-8 -*-
"""
"""

import random

import numpy as np
from ContentKNNAlgorithm import ContentKNNAlgorithm
from Evaluator import Evaluator
from surprise import NormalPredictor

from KnnContentBased.MovieLens import MovieLens


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

# Construct an Evaluator to evaluate them
evaluator = Evaluator(evaluationData, rankings)

contentKNN = ContentKNNAlgorithm()
evaluator.AddAlgorithm(contentKNN, "ContentKNN")

# Random recommendations,for comparison purposes
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

evaluator.Evaluate(False)
# evaluator.SampleTopNRecs(ml)
