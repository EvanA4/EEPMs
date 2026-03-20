from evolver import RGM_Evolver
from sys import argv, stderr


PLANETS = ["mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"]

def main():
    if len(argv) != 2 or argv[1] not in PLANETS:
        print(f"usage: {argv[0]} planet", file=stderr)
        return 1

    evo = RGM_Evolver(argv[1])
    deferent_model, _ = evo.simulate(1, 5, 1)
    epicycle_model, _ = evo.simulate(2, 20, 1, deferent_model)
    epicycle_model, _ = evo.simulate(2, 10, .1, epicycle_model)
    evo.simulate(3, 30, .05, epicycle_model)


if __name__ == "__main__":
    main()


'''
TODO evolve deferent first (eccentricity, eccentric angle, ed_av, radii=0, epicycle angle) with msqe
TODO evolve epicycle second (radii, planet angle, pe_av)
TODO maybe final pass over all parameters with very tiny mutation?
'''