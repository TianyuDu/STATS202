import sys

import numpy as np
import pandas as pd

sys.path.append("../")
from util import data_proc
from CONSTANTS import PANSS


def get_data():
    df = data_proc.load_whole("../data/")
    selected = PANSS + ["PANSS_Total"]
    initial_observations = df[df.VisitDay == 0][selected]
    return initial_observations
