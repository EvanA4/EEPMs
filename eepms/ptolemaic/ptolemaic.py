import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import math
from datetime import datetime
from geomodel import GeoModel


def main():
    df = pd.read_csv(os.path.join("csvs", "expected", "mercury.csv"))
    print(df.head())
    
    model = GeoModel(datetime.now(), random.random()*math.pi*2)
    model.graph_params()


if __name__ == "__main__":
    main()

'''
For the five planets: Mercury, Venus, Mars, Jupiter, Saturn
'''
