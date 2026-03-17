import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import math
from datetime import datetime
from geomodel import RandGeoModel


def model_loss(model, datetimes, longitudes):
    preds = [model.predict_pos(dt)[2] for dt in datetimes]
    losses = [(preds[i]-longitudes[i])**2 for i in range(len(datetimes))]
    return np.mean(losses)


def make_gen(tourn, GEN_SIZE):
    output = []
    for i in range(GEN_SIZE):
        first = random.randint(0, len(tourn)-1)
        second = random.randint(0, len(tourn)-1)
        model = RandGeoModel(
            tourn[first][0].start_time,
            tourn[first][0].start_long,
            tourn[first][0].crossover(tourn[second][0])
        )
        model.mutate()
        output.append(model)
    return output


def main():
    df = pd.read_csv(os.path.join("csvs", "expected", "mercury.csv"))
    datetimes = [datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S") for timestamp in df["Timestamp"].to_list()][::4]
    longitudes = df["Longitude"][::4].to_list()

    GEN_SIZE = 1000
    NUM_GENS = 20

    gen_results = [] # list of tuples of ( best_models, best_losses )
    gen = [RandGeoModel(datetimes[0], longitudes[0]) for i in range(GEN_SIZE)]
    tourn = []
    
    print("Gen |       Loss", flush=True)
    for i in range(NUM_GENS):
        print(f"{i:<3d}", end="", flush=True)
        gen_data = [
            (
                gen[i],
                model_loss(gen[i], datetimes, longitudes)
            )
            for i in range(len(gen))
        ]
        gen_data.sort(key=lambda x: x[1])

        tourn = gen_data[:5]
        gen_results.append(tourn)

        if i != NUM_GENS-1:
            gen = make_gen(tourn, GEN_SIZE)

        print(f" | {tourn[0][1]:5.4f}", flush=True)
    print()

    # Loss vs. Generation
    plt.figure(figsize=(10, 6))
    plt.title("Loss vs. Generation")
    plt.xlabel("Generation")
    plt.ylabel("Loss")
    losses = [result[0][1] for result in gen_results]
    plt.plot(range(NUM_GENS), losses)
    plt.show()
    plt.close()

    # print and graph best model
    best_model = tourn[0][0]
    best_model.print_props()
    best_model.graph_model(datetime.now())

    # Expected Planetary Path
    plt.figure(figsize=(10, 6))
    plt.title("Planetary Paths")
    plt.xlabel("Date")
    plt.ylabel("Longitude")
    plt.scatter(datetimes, longitudes, s=10, c='g', label="Expected", alpha=.5)
    plt.scatter(datetimes, [best_model.predict_pos(dt)[2] for dt in datetimes], s=10, c='r', label="Actual", alpha=.5)
    plt.legend()
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()

'''
For the five planets: Mercury, Venus, Mars, Jupiter, Saturn
'''
