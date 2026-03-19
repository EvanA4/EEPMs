import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import math
from datetime import datetime

from geomodel import RandGeoModel, RGM_Initializer


class RGM_Evolver:
    GEN_SIZE = 1000
    TOURN_SIZE = 10
    NUM_GENS = [0, 10, 10, 5]
    PERFECT = 10
    PENALTY = 1
    SMALL_EPICYCLE = .1
    STRENGTHS = [0, 1, 1, .2]


    def __init__(self):
        # read data
        self.gen: list[RandGeoModel] = []
        df = pd.read_csv(os.path.join("csvs", "expected", "mercury.csv"))
        self.datetimes = [datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S") for timestamp in df["Timestamp"].to_list()[::4]]
        self.longitudes = df["Longitude"].to_list()[::4]
        # self.cumu_longs = self.get_cumu_longs()
        self.stage = 1

        # approximate average angular velocity of planet
        self.longitude_range = math.radians(self.get_long_range(self.longitudes))
        td = self.datetimes[-1] - self.datetimes[0]
        days = td.days + td.seconds/86400
        self.avg_av = self.longitude_range / days

        # approximate synodic period
        synodic = self.get_synodic_p(self.longitudes)
        self.synodic_av = 2*math.pi / synodic

        # two elements of retrograde hump:
        # 1. average days between peak and trough
        # 2. average difference in longitude between peak and trough

        # find average drop in longitude with each step, only looking at retrograde (in degrees)
        self.retrostep, self.prostep = self.get_steps(self.longitudes)

        print(
            f"Longitude range:          {self.longitude_range/(2*math.pi)}\n"
            f"Average angular velocity: {self.avg_av}\n"
            f"Synodic period:           {synodic}\n"
            f"    angular velocity:     {self.synodic_av}\n"
            f"Avg Retrostep:            {self.retrostep}\n"
            f"Avg Prostep:              {self.prostep}\n"
        )

    
    def get_hump(self, longs: list[float]):
        pass


    def get_steps(self, longs: list[float], is_radians=False):
        half_jump = math.pi if is_radians else 180
        total_retrostep = 0
        num_retrosteps = 0
        total_prostep = 0
        num_prosteps = 0
        for i in range(1, len(longs)):
            diff = longs[i]-longs[i-1]
            wrapped_down = diff < -half_jump
            wrapped_up = diff > half_jump
            if not wrapped_down and diff < 0:
                total_retrostep += diff
                num_retrosteps += 1
            if not wrapped_up and diff > 0:
                total_prostep += diff
                num_prosteps += 1
        return (
            -total_retrostep/num_retrosteps if num_retrosteps else 0,
            total_prostep/num_prosteps if num_prosteps else 0
        )


    def get_synodic_p(self, longs: list[float], is_radians=False):
        half_jump = math.pi if is_radians else 180
        first_retro = None
        passed_first_downturn = False
        for i in range(1, len(self.datetimes)):
            wrapped_down = longs[i]-longs[i-1] < -half_jump
            if longs[i]-longs[i-1] < 0 and not wrapped_down:
                if first_retro is None:
                    first_retro = self.datetimes[i]
                elif passed_first_downturn:
                    td = self.datetimes[i] - first_retro
                    return td.days + td.seconds/86400
            elif not wrapped_down and not passed_first_downturn and first_retro is not None:
                passed_first_downturn = True
        return 0.
    

    def get_cumu_longs(self, longs: list[float], is_radians=False):
        half_jump = math.pi if is_radians else 180
        full_jump = 2*half_jump
        offset = 0
        cumu_longs = []
        for i in range(1, len(longs)):
            wrapped_down = longs[i]-longs[i-1] < -half_jump
            wrapped_up = longs[i]-longs[i-1] > half_jump
            if wrapped_down: offset += full_jump
            elif wrapped_up: offset -= full_jump
            cumu_longs.append(longs[0] + offset)

        return cumu_longs


    def get_long_range(self, longs: list[float], is_radians=False):
        half_jump = math.pi if is_radians else 180
        full_jump = 2*half_jump
        offset = 0
        for i in range(1, len(longs)):
            wrapped_down = longs[i]-longs[i-1] < -half_jump
            wrapped_up = longs[i]-longs[i-1] > half_jump
            if wrapped_down: offset += full_jump
            elif wrapped_up: offset -= full_jump
        return longs[-1] - longs[0] + offset


    def min_long_diff(self, first: float, second: float):
        # assumes longitudes are in degrees
        long_top = max(first, second)
        long_bottom = min(first, second)
        start_diff_frac = min(long_top-long_bottom, 360-long_top+long_bottom) / 180
        return start_diff_frac


    def model_eval(self, model: RandGeoModel, verbose=False):
        # mean squared error
        preds = [model.predict_pos(dt)[2] for dt in self.datetimes]
        errs = [self.min_long_diff(preds[i],self.longitudes[i]) for i in range(len(self.datetimes))]
        sq_errs = [err**2 for err in errs]
        msqe_penalty = float(np.mean(sq_errs) * self.PENALTY * 5)
        if self.stage == 1:
            if verbose:
                print(f"msqe_penalty: {msqe_penalty}\n")
            return self.PERFECT - msqe_penalty

        # starting longitude should be the same
        start_penalty = self.min_long_diff(preds[0], self.longitudes[0]) / 2 * self.PENALTY

        # planet's longitude should change the same amount
        preds_range = math.radians(self.get_long_range(preds))
        long_penalty = math.fabs(preds_range - self.longitude_range) / self.longitude_range * self.PENALTY

        # planet should enter retrograde and have similar retrostep
        pred_retrostep, pred_prostep = self.get_steps(preds)
        step_penalty = math.fabs(pred_retrostep-self.retrostep)/self.retrostep * self.PENALTY
        step_penalty += math.fabs(pred_prostep-self.prostep)/self.prostep * self.PENALTY

        # planet-around-epicycle speed should be related to synodic period
        synodic_penalty = math.fabs(model.properties[RandGeoModel.IDX_PE_AV]-self.synodic_av)/self.synodic_av * self.PENALTY

        if verbose:
            print(
                f"msqe_penalty: {msqe_penalty}\n"
                f"start_penalty: {start_penalty}\n"
                f"long_penalty: {long_penalty}\n"
            )
        return self.PERFECT - step_penalty # - start_penalty - long_penalty - step_penalty - synodic_penalty


    def reproduce(self, tourn: list[tuple[RandGeoModel, float]]):
        strength = self.STRENGTHS[self.stage]
        output = []
        for _ in range(self.GEN_SIZE):
            first = random.randint(0, len(tourn)-1)
            second = random.randint(0, len(tourn)-1)
            ini = RGM_Initializer()
            ini.child(tourn[first][0], tourn[second][0], strength)
            output.append(RandGeoModel(ini))
        return output


    def simulate(self, stage: int, best_model=None):
        # populate first generation
        self.stage = stage
        self.gen = []
        for _ in range(self.GEN_SIZE):
            ini = RGM_Initializer()
            if stage == 1:
                ini.stage1(self.datetimes[0], self.longitudes[0], self.avg_av)
            elif stage == 2:
                ini.stage2(best_model, self.synodic_av)
            else:
                ini.stage3(best_model)
            self.gen.append(RandGeoModel(ini))

        gen_results: list[list[tuple[RandGeoModel, float]]] = []
        
        print("Gen | Fitness", flush=True)
        for i in range(self.NUM_GENS[stage]):
            print(f"{i:<3d}", end="", flush=True)
            gen_data = [
                (self.gen[i], self.model_eval(self.gen[i]))
                for i in range(self.GEN_SIZE)
            ]
            gen_data.sort(key=lambda x: x[1], reverse=True)

            tourn = gen_data[:self.TOURN_SIZE]
            gen_results.append(tourn)

            if i != self.NUM_GENS[stage]-1:
                self.gen = self.reproduce(tourn)

            print(f" | {tourn[0][1]:>7.4f}", flush=True)
        print()

        # Loss vs. Generation
        plt.figure(figsize=(10, 6))
        plt.title("Fitness vs. Generation")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        fitnesses = [result[0][1] for result in gen_results]
        plt.plot(range(self.NUM_GENS[stage]), fitnesses)
        plt.show()
        plt.close()

        # print and graph best model
        best_model = gen_results[-1][0][0]
        best_model.print_props()
        best_model.graph_model()
        self.model_eval(best_model, True)

        # Expected Planetary Path
        plt.figure(figsize=(10, 6))
        plt.title("Planetary Paths")
        plt.xlabel("Date")
        plt.ylabel("Longitude")
        plt.yticks(range(0, 361, 30))
        plt.scatter(self.datetimes, self.longitudes, s=10, c='g', label="Expected", alpha=.5)
        plt.scatter(
            self.datetimes, [best_model.predict_pos(dt)[2] for dt in self.datetimes],
            s=10, c='r', label="Actual", alpha=.5
        )
        plt.legend()
        plt.show()
        plt.close()

        return gen_results[-1][0]