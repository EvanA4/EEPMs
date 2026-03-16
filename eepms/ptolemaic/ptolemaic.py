import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import math
from datetime import datetime
from geomodel import RandGeoModel


def main():
    df = pd.read_csv(os.path.join("csvs", "expected", "mercury.csv"))

    datetimes = [datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S") for timestamp in df["Timestamp"].to_list()][::4]
    longitudes = df["Longitude"][::4].to_list()

    best_model = None
    best_preds = None
    min_loss = 100_000_000
    
    print("Iterating...", end="", flush=True)
    for i in range(10000):
        model = RandGeoModel(datetimes[0], longitudes[0])
        preds = [model.predict_pos(dt)[2] for dt in datetimes]
        losses = [(preds[i]-longitudes[i])**2 for i in range(len(datetimes))]
        loss = np.mean(losses)
        if (loss < min_loss):
            best_preds = preds
            best_model = model
    print("done", flush=True)

    plt.figure(figsize=(10, 6))
    plt.scatter(datetimes, longitudes, s=10)
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(datetimes, preds, s=10)
    plt.show()
    plt.close()

    best_model.graph_model()


if __name__ == "__main__":
    main()

'''
For the five planets: Mercury, Venus, Mars, Jupiter, Saturn
'''
