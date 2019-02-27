import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sklearn.metrics

from cross_validate import *
from dataset import Dataset, BaseSplit
from dist_models import SIFT
from evaluation import *
from model_wrapper import *
import utils


def main():
	test_sift()


def test_sift():
	model = DirectDistanceModel(SIFT())
	split = BaseSplit(Dataset(os.path.join(utils.get_eyez_dir(), 'Recognition', 'Databases', 'Rot ScleraNet', 'temp')))
	model.evaluate(split.gallery, split.probe)


if __name__ == '__main__':
	main()

