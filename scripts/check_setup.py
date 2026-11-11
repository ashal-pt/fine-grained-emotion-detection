import platform
import sys

import datasets
import numpy
import pandas
import sklearn

from emotion_nlp.config import DATASET_CONFIG, DATASET_NAME, PROJECT_ROOT, SEED, set_seed

set_seed()
print("Python        :", sys.version.split()[0], f"({platform.system()})")
print("numpy         :", numpy.__version__)
print("pandas        :", pandas.__version__)
print("scikit-learn  :", sklearn.__version__)
print("datasets      :", datasets.__version__)
print("Project root  :", PROJECT_ROOT)
print("Dataset       :", DATASET_NAME, "/", DATASET_CONFIG)
print("Seed          :", SEED)
print("Setup OK")