import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import math
from datetime import datetime
from geomodel import RandGeoModel


class RGM_Evolver:
    GEN_SIZE = 1000
    TOURN_SIZE = 10
    NUM_GENS = 1
    PENALTY = 50_000
    SMALL_EPICYCLE = .1


    def __init__(self):
        # read data
        df = pd.read_csv(os.path.join("csvs", "expected", "mercury.csv"))
        self.datetimes = [datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S") for timestamp in df["Timestamp"].to_list()][::4]
        self.longitudes = df["Longitude"][::4].to_list()

        # approximate average angular velocity of planet
        longitude_range = 0
        for i in range(len(self.longitudes)-1):
            if self.longitudes[i+1]-self.longitudes[i] < -300:
                print("loop detected!")
                if i == 0: longitude_range += 360 - self.longitudes[0]
                else: longitude_range += 360
        longitude_range += self.longitudes[-1]
        longitude_range = math.radians(longitude_range)
        TIME_DELTA = self.datetimes[1] - self.datetimes[0]
        DAYS_RANGE = TIME_DELTA.days + TIME_DELTA.seconds/86400
        self.avg_av = longitude_range / DAYS_RANGE

        print("AVERAGE AV:", self.avg_av)

        # populate first generation
        self.gen = [
            RandGeoModel(self.datetimes[0], self.longitudes[0], self.avg_av)
            for _ in range(self.GEN_SIZE)
        ]


    def model_loss(self, model: RandGeoModel, verbose=False):
        # mean squared error
        preds = [model.predict_pos(dt)[2] for dt in self.datetimes]
        sq_losses = [(preds[i]-self.longitudes[i])**2 for i in range(len(self.datetimes))]
        mean_sq_loss = 0 # np.mean(sq_losses)

        # stagnant penalty (planet doesn't move)
        stagnant_penalty = (1 - model.properties[RandGeoModel.IDX_ED_AV]/self.avg_av) * self.PENALTY

        # unreasonably small epicycle
        radii_penalty = 0 #self.PENALTY if model.properties[RandGeoModel.IDX_RADII] < self.SMALL_EPICYCLE else 0

        if verbose:
            print(
                f"mean_sq_loss:{mean_sq_loss:15f}\n"
                f"stagnant_penalty:{stagnant_penalty:15f}\n"
                f"radii_penalty:{radii_penalty:15f}\n"
            )

        return mean_sq_loss + stagnant_penalty + radii_penalty


    def reproduce(self, tourn, GEN_SIZE):
        output = []
        for _ in range(GEN_SIZE):
            first = random.randint(0, len(tourn)-1)
            second = random.randint(0, len(tourn)-1)
            model = RandGeoModel(
                tourn[first][0].start_time,
                tourn[first][0].start_long,
                self.avg_av,
                tourn[first][0].crossover(tourn[second][0])
            )
            model.mutate()
            output.append(model)
        return output


    def simulate(self):
        gen_results: list[tuple[list[RandGeoModel], list[float]]] = []
        tourn: list[tuple[RandGeoModel, float]] = []
        
        print("Gen |       Loss", flush=True)
        for i in range(self.NUM_GENS):
            print(f"{i:<3d}", end="", flush=True)
            gen_data = [
                (self.gen[i], self.model_loss(self.gen[i]))
                for i in range(self.GEN_SIZE)
            ]
            gen_data.sort(key=lambda x: x[1])

            tourn = gen_data[:self.TOURN_SIZE]
            gen_results.append(tourn)

            if i != self.NUM_GENS-1:
                self.gen = self.reproduce(tourn, self.GEN_SIZE)

            print(f" | {tourn[0][1]:5.4f}", flush=True)
        print()

        # Loss vs. Generation
        plt.figure(figsize=(10, 6))
        plt.title("Loss vs. Generation")
        plt.xlabel("Generation")
        plt.ylabel("Loss")
        losses = [result[0][1] for result in gen_results]
        plt.plot(range(self.NUM_GENS), losses)
        plt.show()
        plt.close()

        # print and graph best model
        best_model = tourn[0][0]
        best_model.print_props()
        best_model.graph_model(datetime.now())
        self.model_loss(best_model, True)

        # Expected Planetary Path
        plt.figure(figsize=(10, 6))
        plt.title("Planetary Paths")
        plt.xlabel("Date")
        plt.ylabel("Longitude")
        plt.scatter(self.datetimes, self.longitudes, s=10, c='g', label="Expected", alpha=.5)
        plt.scatter(self.datetimes, [best_model.predict_pos(dt)[2] for dt in self.datetimes], s=10, c='r', label="Actual", alpha=.5)
        plt.legend()
        plt.show()
        plt.close()