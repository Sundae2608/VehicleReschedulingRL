import heapq
import numpy as np

from typing import Dict, Set, List, Tuple, Union

from train_sim.train import Train, LongTrain, TrainType
from train_sim.track_map import TrackMap
from train_sim.tracks import Station, Track
from train_sim.transitions import Node, Transition


class SimulationParams:
    def __init__(self,
                 time_limit: float,
                 default_interval: float,
                 num_trains: int,
                 train_length_ratio: float,
                 train_speed: float,
                 train_breakdown_timeout: float,
                 train_breakdown_probability_per_unit_distance: float,
                 sim_rng_seed: int = 10000,
                 problem_rng_seed: int = 20000):

        # Running parameters
        # Time limit of the simulation
        self.time_limit = time_limit
        # Default interval of time in which the simulation will advance for each step
        self.default_interval = default_interval

        # Train parameters
        # Number of trains
        self.num_trains = num_trains
        # Speed of the train
        self.train_speed = train_speed
        # The length of train, relative to the track that the train resides in
        self.train_length_ratio = train_length_ratio
        # The amount of time the train is unable to move when breaking down
        self.train_breakdown_timeout = train_breakdown_timeout
        # The probability that the train will break down, per unit distance
        self.train_breakdown_probability_per_unit_distance = train_breakdown_probability_per_unit_distance

        # Behavior random number generator
        self.sim_rng_seed = sim_rng_seed

        # Problem generation rng
        self.problem_rng_seed = problem_rng_seed


class Simulation:
    def __init__(self,
                 track_map: TrackMap,
                 sim_params: SimulationParams):
        self.track_map: TrackMap = track_map
        self.sim_params = sim_params

        self.sim_time_limit = sim_params.time_limit
        self.sim_default_interval = sim_params.default_interval
        self.rng = np.random.default_rng(sim_params.sim_rng_seed)
        self.problem_rng = np.random.default_rng(sim_params.problem_rng_seed)

        # Post process information. These information is stored after each step for easy look up.
        # This variable map a track with the list of train that the track currently occupies
        self.occupied_tracks: Dict[Track, Set[Train]] = {}

        # A sorted list from node1 to node2 showing parts of the track currently being occupied by trains
        # For example, if a track has the following data
        # [(0, 10, Train1), (13, 15, Train2)].
        # That means that the part of the track from 0 to 10 is occupied by Train1, 13 to 15 is occupied by Train2.
        # Anywhere between 10 and 13, and after 15 is free.
        self.occupied_track_parts: Dict[Track, List[Tuple[Train, float, float, Node]]] = {}

        # Call reset_problem, this will shuffle the initial positions and stations, and create a new problem for the
        # simulation
        self.train_fleet = []
        self.train_begins: Dict[Train, Tuple[Track, Node, float, List[Union[Node, Transition]]]] = {}
        self.stations: List[Station] = []
        self.train_goals: Dict[Train, Station] = {}
        self.train_goals_reached = {}
        self.train_goals_reached_time = {}
        self.sim_ended: bool = False
        self.sim_time = 0
        self.reset_problem()

    def add_train(self, speed, track, track_position, direction_node):
        """
        Ad a point train to the simulation
        :param speed: The speed of the train
        :param track:
        :param track_position:
        :param direction_node:
        :return:
        """
        # Check to make sure track position is legal
        if track_position < 0 or track_position > track.distance:
            raise RuntimeError("Track position must be within the track")

        # Check to make sure direction node must be part of the track
        if direction_node != track.node1 and direction_node != track.node2:
            raise RuntimeError("Directional node must be within the track")

        # Create new train and add to the train fleet
        train = Train(speed, track, track_position, direction_node, self)
        self.train_fleet.append(train)

    def add_long_train(self, speed, train_length, track, direction_node, track_position,
                       breakdown_timeout, breakdown_probability_per_unit_distance, tracks_and_transitions):
        """
        Add a long train, meaning a train
        :param speed: Speed of the train
        :param train_length: Length of the train
        :param track: Track that the train resides in.
        :param track_position: The position in the track that the train currently resides.
        :param direction_node: The node that the train is heading toward
        :param breakdown_timeout: How much time the train stop if receiving a breakdown
        :param breakdown_probability_per_unit_distance: Probability of receiving a breakdown per unit distance.
        :param tracks_and_transitions:
        :return:
        """
        # Check to make sure track position is legal
        if track_position < 0 or track_position > track.distance:
            raise RuntimeError("Track position must be within the track")

        # Check to make sure direction node must be part of the track
        if direction_node != track.node1 and direction_node != track.node2:
            raise RuntimeError("Directional node must be within the track")

        # Create new train and add to the train fleet
        train = LongTrain(speed, train_length, track, direction_node, track_position,
                          breakdown_timeout, breakdown_probability_per_unit_distance,
                          tracks_and_transitions, self)
        self.train_fleet.append(train)
        return train

    def add_station(self, track, position):
        """
        Add a station on a track, at a certain position
        :param track: The track in which the station stays on
        :param position: The position from Node1 of the station
        :return: The newly created station
        """
        station = track.add_station(position)
        self.stations.append(station)
        return station

    def assign_destination_station(self, train: Train, station: Station):
        """
        Assgin destination to the train
        :return:
        """
        if train not in self.train_fleet or station not in self.stations:
            raise RuntimeError("Train and station must already be created as part of the track map")
        train.set_destination(station)
        self.train_goals[train] = station
        self.train_goals_reached[train] = False

    def reset_problem(self):
        """
        Reset the problem. This means that while the map stays the same, the starting and ending positions of trains
        change:
        """
        self.train_fleet = []
        self.train_begins = {}
        self.stations = []
        self.train_goals = {}
        self.train_goals_reached = {}
        self.train_goals_reached_time = {}
        self.sim_ended = False
        self.sim_time = 0

        # Add the train based on simulation parameters.
        # Choose tracks for trains
        tracks_for_train = self.problem_rng.choice(a=list(self.track_map.tracks), size=self.sim_params.num_trains,
                                                   replace=False)
        tracks_for_station = self.problem_rng.choice(a=list(self.track_map.tracks), size=self.sim_params.num_trains)

        # For each train, pick two tracks for start and destination:
        for track_train, track_station in zip(tracks_for_train, tracks_for_station):
            speed = self.sim_params.train_speed
            train_length = track_train.distance * self.sim_params.train_length_ratio
            direction_node = self.problem_rng.choice(track_train.get_nodes())
            track_position = track_train.distance * (self.sim_params.train_length_ratio + (1 - self.sim_params.train_length_ratio) * self.rng.random())
            breakdown_timeout = self.sim_params.train_breakdown_timeout
            breakdown_probability_per_unit_distance = self.sim_params.train_breakdown_probability_per_unit_distance
            tracks_and_transitions = []

            # Place the train on one track, place the station on the other track.
            train = self.add_long_train(speed, train_length, track_train, direction_node, track_position,
                                        breakdown_timeout, breakdown_probability_per_unit_distance, tracks_and_transitions)

            # Store the initial locations of the trains, used for reseting position later.
            self.train_begins[train] = (track_train, direction_node, track_position, tracks_and_transitions)

            # Place the station on another track, set it as the goal
            station = self.add_station(track_station, track_station.distance * self.problem_rng.random())
            self.assign_destination_station(train, station)


    def reset(self):
        """
        Reset the simulation to the beginning point
        """
        # Put the train back to their original destination
        for train in self.train_begins:
            track_train, direction_node, track_position, tracks_and_transitions = self.train_begins[train]
            if train.train_type == TrainType.LONG_TRAIN:
                train.reset_position(track_train, direction_node, track_position, tracks_and_transitions)
            else:
                train.reset_position(track_train, direction_node, track_position)

        # Reset simulation time and train task
        self.train_goals_reached = {}
        self.train_goals_reached_time = {}
        self.sim_ended: bool = False
        self.sim_time = 0

    def _post_step_process(self):
        """
        Process information in the simulation after the step. This step will process some convenient information about
        the simulation such as where is each train on the track, etc.
        """
        # Process the occupied track parts
        self.occupied_tracks.clear()
        self.occupied_track_parts.clear()
        for train in self.train_fleet:
            occupied_track_parts = train.occupied_track_parts()
            for track, pos1, pos2, node_toward in occupied_track_parts:
                if track not in self.occupied_tracks:
                    self.occupied_tracks[track] = {train}
                    self.occupied_track_parts[track] = [(train, pos1, pos2, node_toward)]
                else:
                    self.occupied_tracks[track].add(train)
                    self.occupied_track_parts[track].append((train, pos1, pos2, node_toward))

        # Sort the track parts
        for track in self.occupied_track_parts:
            self.occupied_track_parts[track].sort(key=lambda x: x[1])

        # Process where the simulation should end based on current ending criteria and
        destinations_reached = 0
        for train in self.train_goals_reached:
            if self.train_goals_reached[train]:
                destinations_reached += 1
        if destinations_reached == len(self.train_goals_reached.keys()):
            self.sim_ended = True
        if self.sim_time > self.sim_time_limit:
            self.sim_ended = True

    def step(self):
        """
        Step the simulation at a default interval. This helps match the API of the simulation with the typical API of
        an RL platform.
        """
        self.step_interval(self.sim_default_interval)

    def step_interval(self, interval):
        """
        Step forward the simulation by an input amount of time.
        :param interval: Time interval in which the simulation will proceed (in seconds)
        """
        # Step forward the simulation by some amount of time
        self.sim_time += interval
        for train in self.train_fleet:
            train.step_interval(interval)
            if train.pass_destination and not self.train_goals_reached[train]:
                self.train_goals_reached[train] = True
                self.train_goals_reached_time[train] = self.sim_time

        # Once the processing is done, perform post processing (for feature gather convenient)
        self._post_step_process()

    def get_simulation_reward(self):
        """
        Get the total reward of the train simulation
        """
        # Sum the reward for each train, and normalize by the number of trains
        reward = 0
        for train in self.train_fleet:
            if not self.train_goals_reached[train]:
                continue
            else:
                reward += (self.sim_time_limit - self.train_goals_reached_time[train]) / \
                          self.sim_time_limit
        return reward / len(self.train_fleet)
