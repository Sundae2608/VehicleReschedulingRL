"""
Main file that performs multi-agent training.
"""

import numpy as np
import random
import torch

from datetime import datetime
from rl.agent import PolicyAgent
from rl.models.dddqn import DDDQNParams, DDDQNPolicy
from rl.observation import TreeObserver
from train_sim.simulation import Simulation, SimulationParams
from train_sim.track_map_utils import TrackMapGenerator
from utils.timer import Timer


class TrainingParams:
    def __init__(self, n_episodes, n_evaluation_episodes, checkpoint_interval, eps_start, eps_end, eps_decay,
                 num_threads, save_replay_buffer):
        self.n_episodes = n_episodes
        self.n_evaluation_episodes = n_evaluation_episodes
        self.checkpoint_interval = checkpoint_interval
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.num_threads = num_threads
        self.save_replay_buffer = save_replay_buffer


class TrackMapGenerationParams:
    def __init__(self, num_transitions_range, num_trains_range):
        self.num_transitions_range = num_transitions_range
        self.num_trains_range = num_trains_range


class ObservationParams:
    def __init__(self, observation_depth):
        self.observation_depth = observation_depth


class ModelParams:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size


def main():
    """ Main function """

    # Seed
    random.seed(42)

    # Unique ID for this training
    now = datetime.now()
    training_id = now.strftime('%y%m%d%H%M%S')

    # Track generation input
    sim_params = SimulationParams(
        time_limit=20.0,
        default_interval=0.02,
        num_trains=6,
        train_length_ratio=0.7,
        train_speed=1000,
        train_breakdown_timeout=1.0,
        train_breakdown_probability_per_unit_distance=0.0,
        sim_rng_seed=802,
        problem_rng_seed=2000
    )

    # Observation parameters
    obs_param = ObservationParams(
        observation_depth=3
    )

    # Model parameters
    model_params = DDDQNParams(
        hidden_size=128,
        buffer_size=100000,
        batch_size=128,
        update_every=8,
        learning_rate=0.5e-4,
        tau=1e-3,
        gamma=0.99,
        buffer_min_size=0,
        use_gpu=True,
    )

    # Training parameters
    training_params = TrainingParams(
        n_episodes=2500,
        n_evaluation_episodes=10,
        checkpoint_interval=50,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.99,
        num_threads=8,
        save_replay_buffer=False
    )

    # Track generation params
    track_params = TrackMapGenerationParams(
        num_trains_range=[6, 12],
        num_transitions_range=[10, 20]
    )

    # Create Track Map Generator. This will generate a different map for each episode.
    tmg_list = []
    for num_transitions in range(track_params.num_transitions_range[0], track_params.num_transitions_range[1] + 1):
        tmg_list.append(
            TrackMapGenerator(
                average_section_length=400,
                num_transitions=num_transitions)
        )
    num_trains_list = list(range(track_params.num_trains_range[0], track_params.num_trains_range[1] + 1))

    eps = training_params.eps_start
    policy = None

    # For each episode, run a series of agents
    print("Episode ID, Score, Completion, Episilon, % Break, % U-turn, % Left, % Right")
    for episode_idx in range(training_params.n_episodes):
        # Setup timer
        simulation_timer = Timer()

        # Update training parameters
        # Set up the simulation
        tmg = random.choice(tmg_list)
        track_map = tmg.generate_next_map()[0]
        sim_params.num_trains = random.choice(num_trains_list)
        simulation = Simulation(track_map=track_map, sim_params=sim_params)
        action_types = np.array([0, 0, 0, 0])
        action_num = 0

        # Create the agent and assign to each train.
        train = simulation.train_fleet[0]
        observer = TreeObserver(train, simulation, obs_param.observation_depth)
        state_size = len(observer.get_observation())
        action_size = 4
        if not policy:
            policy = DDDQNPolicy(state_size, action_size, model_params)
        for train in simulation.train_fleet:
            agent = PolicyAgent(
                train=train,
                observer=TreeObserver(train, simulation, obs_param.observation_depth),
                policy=policy,
                eps=eps,
                rng=simulation.rng)
            train.set_agent(agent)

        # Let the simulation run, and record the action and observation
        while not simulation.sim_ended:
            # Step forward the simulation
            simulation_timer.start()
            simulation.step()
            simulation_timer.end()

            # Add the experience into the memory for each agent
            simulation_history = []
            for train in simulation.train_fleet:
                state, action = train.get_last_state_and_actions()
                reward = 0.0 if simulation.train_goals_reached[train] else -1.0
                done = simulation.train_goals_reached[train]
                curr_state, _ = train.get_agent_state_and_actions()
                simulation_history.append((state, action, reward, curr_state, done))
                action_types[action] += 1
                action_num += 1

            # Step the policy
            for state, action, reward, curr_state, done in simulation_history:
                policy.step(state, action, reward, curr_state, done)

        # Update training parameters
        eps = max(training_params.eps_end, training_params.eps_decay * eps)

        # Collect information about training
        tasks_finished = sum(simulation.train_goals_reached[train] for train in simulation.train_fleet)
        completion = tasks_finished / len(simulation.train_fleet)
        normalized_score = simulation.get_simulation_reward()
        action_types = action_types / action_num

        # Print logs
        if episode_idx % training_params.checkpoint_interval == 0:
            policy.save('./checkpoints/' + training_id + '-' + str(episode_idx))

            if training_params.save_replay_buffer:
                policy.save_replay_buffer('./replay_buffers/' + training_id + '-' + str(episode_idx) + '.pkl')

            # Do a proper evaluation every check points
            total_score = 0
            total_completion_percentage = 0
            for eval_episode_idx in range(training_params.n_evaluation_episodes):
                tmg = random.choice(tmg_list)
                track_map = tmg.generate_next_map()[0]
                sim_params.num_trains = random.choice(num_trains_list)
                simulation = Simulation(track_map=track_map, sim_params=sim_params)

                # Set up the trains
                train = simulation.train_fleet[0]
                observer = TreeObserver(train, simulation, obs_param.observation_depth)
                for train in simulation.train_fleet:
                    agent = PolicyAgent(
                        train=train,
                        observer=TreeObserver(train, simulation, obs_param.observation_depth),
                        policy=policy,
                        eps=0.0,
                        rng=simulation.rng)
                    train.set_agent(agent)

                # Let the simulation run, and record the action and observation
                while not simulation.sim_ended:
                    simulation.step()

                tasks_finished = sum(simulation.train_goals_reached[train] for train in simulation.train_fleet)
                completion = tasks_finished / len(simulation.train_fleet)
                normalized_score = simulation.get_simulation_reward()

                total_score += normalized_score
                total_completion_percentage += completion
                avg_score = total_score / training_params.n_evaluation_episodes
                avg_completion = total_completion_percentage / training_params.n_evaluation_episodes

            print("Evaluation {0:d}: Average score {1:f}, Average completion {2:f}".format(
                episode_idx, avg_score, avg_completion))

        # Print information for PDF processing.
        print("{0:d}, {1:f}, {2:f}, {3:f}, {4:f}, {5:f}, {6:f}, {7:f}".format(
            episode_idx, normalized_score, completion, eps,
            action_types[0], action_types[1], action_types[2], action_types[3]
        ))


if __name__ == "__main__":
    main()
