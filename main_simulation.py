"""
Generate a random network a run it without the graphical component. Helps to see how quick it is to run one simulation.
Author: Son Pham
"""
from train_sim.simulation import Simulation, SimulationParams
from train_sim.track_map_utils import TrackMapGenerator


def main():
    """ Main function """

    # Track generation input
    sim_params = SimulationParams(
        time_limit=120.0,
        default_interval=0.02,
        num_trains=6,
        train_length_ratio=0.7,
        train_speed=1000,
        train_breakdown_timeout=1.0,
        train_breakdown_probability_per_unit_distance=0.0,
        sim_rng_seed=802,
        problem_rng_seed=2000
    )

    # Generate a random track
    tmg = TrackMapGenerator(
        average_section_length=500,
        num_transitions=10,
        rng_seed=42)
    track_map = tmg.generate_next_map()[0]
    simulation = Simulation(track_map=track_map, sim_params=sim_params)

    # Run the simulation until it ends
    while not simulation.sim_ended:
        simulation.step()

    # Print the statistics regarding the simulation
    print("Simulation result:")
    print("Total time:    {0:.3f}".format(simulation.sim_time))
    print("Total reward:  {0:.3f}".format(simulation.get_simulation_reward()))
    print("Individual train time:")
    for i, train in enumerate(simulation.train_fleet):
        if train in simulation.train_goals_reached_time:
            print("- Train {0:d}: {1:.3f}".format(i, simulation.train_goals_reached_time[train]))
        else:
            print("- Train {0:d}: Didn't reach destination".format(i))


if __name__ == "__main__":
    main()
