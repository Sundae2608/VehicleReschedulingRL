"""
This file provides feature generator for the RL agents.
"""
import numpy as np

from typing import List, Tuple, Set

from train_sim.simulation import Simulation
from train_sim.track_map import Track
from train_sim.train import Train
from train_sim.transitions import TransitionType, Node


def normalize_obs(obs):
    return np.clip(obs, -1.0, 1.0)


class TrainFeatureProducer:
    def __init__(self, train: Train, simulation: Simulation):
        """
        This producer will produce features that are related to the status of the train
        :param train: the train that we are observing through.
        :param simulation: The simulation that the train operates on
        """
        self.train = train
        self.simulation = simulation

    def _reached_destination(self):
        """
        :return: Whether the train has already reached their destination
        """
        return [self.train in self.simulation.train_goals_reached]

    def _breakdown_features(self):
        """
        :return: A tuple of two features:
            - Whether the train is currently broken down or not.
            - The amount of time until the train recovered from break down:
        """
        if self.train.is_broken_down:
            return [True, self.train.breakdown_timeout]
        return [False, 0.0]

    def produce_features(self):
        """
        Produce the feature for this particular train.
        :return: An np.array representing the features
        """
        features = [
            self._reached_destination(),
            self._breakdown_features()
        ]
        return np.array([val for values in features for val in values]).astype(float)


class BranchFeatureProducer:
    def __init__(self, train: Train, head_track: Track, head_node_toward: Node, head_position: float,
                 track_node_pairs: List[Tuple[Track, Node]], simulation: Simulation):
        """
        Produce feature for this particular set of rail tracks
        :param train: The train whose agent we are producing features for
        :param head_track: The track where the head that we are considering resides in
        :param head_position: Where is the head in the track
        :param head_node_toward: The node that the head is pointing towards
        :param track_node_pairs: The list of track and nodes that we are extracting features
        :param simulation: The simulation that the train operates on
        """
        self.train = train
        self.head_track = head_track
        self.head_position = head_position
        self.head_node = head_node_toward
        self.track_node_pairs = track_node_pairs
        self.tracks = [track for track, _ in track_node_pairs] if self.track_node_pairs else None
        self.nodes = [node for _, node in track_node_pairs] if self.track_node_pairs else None
        self.simulation = simulation
        self.track_map = self.simulation.track_map

        # Pre-processing some useful features.
        self.has_track_node_pairs = track_node_pairs is not None and len(track_node_pairs) > 0
        self._branch_distance = 0 if not self.tracks else sum([t.distance for t in self.tracks])
        self._total_track_map_distance = self.track_map.total_track_distance
        self._num_trains = len(self.simulation.train_fleet)

    def _remaining_distance(self):
        """
        :return: The remaining distance in which the train needs to travel to complete the branch.
            If the train is not part of the branch, return the total distance.
        """
        if not self.has_track_node_pairs:
            return [-float("inf")]

        if self.tracks and self.head_track not in self.tracks:
            return [self._branch_distance]
        elif self.tracks:
            return [self._branch_distance - self.head_position]
        return [0.0]

    def _remaining_distance_fraction(self):
        """
        :return: The remaining distance in which the train needs to travel to complete the branch.
            If the train is not part of the branch, return the fraction of the remaining distance to total branch
            distance.
        """
        if not self.has_track_node_pairs:
            return [-float("inf")]
        remaining_distance = self._remaining_distance()[0]
        if self._branch_distance != 0:
            return [remaining_distance / self._branch_distance]
        else:
            return [0.0]

    def _target_in_branch_features(self):
        """
        :return: The remaining distance to the target if target lies in the current branch. 0 otherwise.
        """
        if not self.has_track_node_pairs:
            return [-float("inf"), -float("inf")]
        target_in_branch = False
        dist_to_target = 0
        for track, node_toward in self.track_node_pairs:
            if track == self.train.destination.track:
                target_in_branch = True
                if node_toward == track.node2:
                    dist_to_target += self.train.destination.position
                else:
                    dist_to_target += track.distance - self.train.destination.position
                break
            else:
                dist_to_target += track.distance
        if self.tracks and self.head_track == self.tracks[0]:
            # If the train is not in the track, remove the distance that the train already covers
            dist_to_target -= self.head_position
        if target_in_branch:
            return [True, dist_to_target]
        return [False, 0.0]

    def _target_in_branch_features_fraction(self):
        """
        :return: The remaining distance to the target if target lies in the current branch. 0 otherwise.
        """
        if not self.has_track_node_pairs:
            return [-float("inf"), -float("inf")]
        target_in_branch, dist_to_target = self._target_in_branch_features()

        if self._branch_distance != 0:
            return [target_in_branch, dist_to_target / self._branch_distance]
        return [target_in_branch, 0.0]

    def _agent_in_branch_features(self):
        """
        :return: A tuple in which:
            - The first value determines whether there is another agent in the branch.
            - The second value determines the distance to that agent
        """
        if not self.has_track_node_pairs:
            return [-float("inf"), -float("inf")]
        agent_in_branch = False
        dist_so_far = 0.0

        # Search the entire branch in the direction of the train, first to find all agents in the way.
        self_train_detected = False
        for track, node_toward in self.track_node_pairs:
            if track not in self.simulation.occupied_tracks:
                if self_train_detected:
                    dist_so_far += track.distance
                continue
            if node_toward == track.node2:
                for train, pos1, pos2, node in self.simulation.occupied_track_parts[track]:
                    if train == self.train:
                        self_train_detected = True
                    elif train != self.train and self_train_detected:
                        dist_so_far += pos1
                        agent_in_branch = True
                        break
                    else:
                        dist_so_far += track.distance
            else:
                for train, pos1, pos2, node in self.simulation.occupied_track_parts[track][::-1]:
                    if train == self.train:
                        self_train_detected = True
                    elif train != self.train and self_train_detected:
                        dist_so_far += track.distance - pos2
                        agent_in_branch = True
                        break
                    else:
                        dist_so_far += track.distance
        if self.tracks and len(self.tracks) > 0 and self.head_track == self.tracks[0]:
            # If the train is not in the track, remove the distance that the train already covers
            dist_so_far -= self.head_position
        if agent_in_branch:
            return [True, dist_so_far]
        else:
            return [False, 0.0]

    def _agent_in_branch_features_fraction(self):
        """
        :return: A tuple in which:
            - The first value determines whether there is another agent in the branch.
            - The second value determines the fraction of the remaining distance to total branch distance.
        """
        if not self.has_track_node_pairs:
            return [-float("inf"), -float("inf")]
        agent_in_branch, dist_so_far = self._agent_in_branch_features()
        if self._branch_distance != 0:
            return [agent_in_branch, dist_so_far / self._branch_distance]
        return [agent_in_branch, 0.0]

    def _conflict_with_crossing_branch_features(self):
        """
        :return: A tuple with three values.
            - The first value is a boolean showing whether there is a possible conflict with other agent in the branch.
            - The second value shows the distance of the train from that conflict point.
            - The third value shows the fraction of the distance to conflict point to the total distance in the branch.
        """
        if not self.has_track_node_pairs:
            return [-float("inf"), -float("inf"), -float("inf")]

        # Tracking variables
        is_conflict = False
        min_distance_to_conflict = float('inf')
        min_fraction_distance_to_conflict = float('inf')

        # Get all track node pairs that potentially conflict
        visited_nodes = set()
        dist_so_far = 0
        if self.tracks and len(self.tracks) > 0 and self.head_track == self.tracks[0]:
            # If the train is not in the track, remove the distance that the train already covers
            dist_so_far -= self.head_position
        for track, node in self.track_node_pairs:
            dist_so_far += track.distance
            trans = node.transition
            side_nodes = trans.get_node_from_the_two_sides(node)

            # Get all conflicting track path from the left branch
            for side_node in side_nodes:
                side_track_node_pairs: List[Tuple[Track, Node]] = []
                curr_node = side_node
                while curr_node is not None:
                    forward_track = curr_node.track
                    side_track_node_pairs.append((forward_track, curr_node))
                    next_trans = curr_node.track.get_other_node(curr_node).transition
                    curr_node = next_trans.get_node_from_going_forward(curr_node)
                    if curr_node is not None and curr_node not in visited_nodes:
                        visited_nodes.add(curr_node)
                        curr_track = curr_node.track
                        curr_node = curr_track.get_other_node(curr_node)
                    else:
                        break

                # Calculate the total distance of the track
                total_dist = dist_so_far + sum([t.distance for t, _ in side_track_node_pairs])

                # Check all trains in conflict path, and investigate how long it is from the end.
                base_dist = dist_so_far
                for path_track, path_node in side_track_node_pairs:
                    base_dist += path_track.distance
                    if path_track not in self.simulation.occupied_track_parts:
                        continue
                    for train, pos1, pos2, train_node in self.simulation.occupied_track_parts[path_track]:
                        if train_node == path_node:
                            if train_node == path_track.node2:
                                dist = base_dist - pos2
                            else:
                                dist = base_dist - path_track.distance + pos1
                            is_conflict = True
                            min_distance_to_conflict = min(min_distance_to_conflict, dist)
                            min_fraction_distance_to_conflict = min(dist / total_dist,
                                                                    min_fraction_distance_to_conflict)
        if is_conflict:
            return [True, min_distance_to_conflict, min_fraction_distance_to_conflict]
        return [False, float('inf'), float('inf')]

    def _conflict_with_crossing_branch_features_fraction(self):
        """
        :return: A tuple of two values:
            - The first value is a boolean showing whether there is a possible conflict with other agent in the branch.
            - The second value shows the fraction of the distance to conflict point to the total distance in the branch
        """
        if not self.has_track_node_pairs:
            return [-float("inf"), -float("inf")]
        is_cross_branch_conflict, min_distance_to_conflict, min_fraction_distance_to_conflict = \
            self._conflict_with_crossing_branch_features()
        return [is_cross_branch_conflict, min_fraction_distance_to_conflict]

    def _unusable_switch_features(self):
        """
        :return: A tuple of two values:
            - A boolean indicating whether there is an unusable switch in the path.
            - The distance to the unusable switch, relative to the total path branch distance.
        """
        if not self.has_track_node_pairs:
            return [-float("inf"), -float("inf")]
        has_unusable_switch = False
        dist_so_far = 0
        if self.tracks and len(self.tracks) > 0 and self.head_track == self.tracks[0]:
            # If the train is not in the track, remove the distance that the train already covers
            dist_so_far -= self.head_position
        for track, node in self.track_node_pairs[:-1]:
            dist_so_far += track.distance
            if node.transition.type != TransitionType.STRAIGHT:
                has_unusable_switch = True
                break
        if has_unusable_switch:
            return [True, dist_so_far]
        return [False, 0.0]

    def _unusable_switch_features_fraction(self):
        """
        :return: A tuple of two values:
            - A boolean indicating whether there is an unusable switch in the path.
            - The fraction of the distance to the unusable switch, relative to the total path branch distance.
        """
        if not self.has_track_node_pairs:
            return [-float("inf"), -float("inf")]
        has_unusable_switch, distance_to_switch = self._unusable_switch_features()
        if self._branch_distance != 0:
            return [has_unusable_switch, distance_to_switch / self._branch_distance]
        return [has_unusable_switch, 0.0]

    def _minimum_distance_to_target_if_branch_taken(self):
        """
        :return: Minimum distance to target if the branch is taken.
        """
        if not self.has_track_node_pairs:
            return [-float('inf')]
        through_track, through_node = self.track_node_pairs[0]
        return [self.track_map.minimum_distance_to_certain_point_through_track(
            self.head_track, self.head_node, self.head_position, through_track, through_node,
            self.train.destination.track, self.train.destination.position)]

    def _minimum_distance_to_target_if_branch_taken_fraction(self):
        """
        :return: Fraction of minimum distance to target if the branch is taken to total network distance.
        """
        if not self.has_track_node_pairs:
            return [-float("inf")]
        minimum_distance_to_target = self._minimum_distance_to_target_if_branch_taken()[0]
        return [minimum_distance_to_target / self._total_track_map_distance]

    def _trains_same_direction_features(self):
        """
        :return: A tuple with 4 values:
            - Number of trains in the branch that is going in the same direction.
            - Fraction of that number compared to total number of trains.
            - Speed ratio of the slow train.
            - Speed ratio of the fastest train.
        """
        if not self.has_track_node_pairs:
            return [-float("inf"), -float("inf"), -float("inf"), -float("inf")]
        self_train_detected = False
        train_set = set()
        min_fraction = float("inf")
        max_fraction = 0.0
        for track, node in self.track_node_pairs:
            if track not in self.simulation.occupied_track_parts:
                continue
            for train, _, _, node_toward in self.simulation.occupied_track_parts[track]:
                if train == self.train:
                    self_train_detected = True
                elif node == node_toward and train != self.train and self_train_detected:
                    train_set.add(train)
                    speed_fraction = train.speed / self.train.speed
                    min_fraction = min(min_fraction, speed_fraction)
                    max_fraction = max(max_fraction, speed_fraction)
        num_trains_in_branch = len(train_set)
        if num_trains_in_branch > 0:
            return [num_trains_in_branch, num_trains_in_branch / self._num_trains, min_fraction, max_fraction]
        return [0.0, 0.0, float("inf"), 0.0]

    def _trains_opposite_direction_features(self):
        """
        :return: A tuple of 4 values:
            - Number of trains in the branch that is going in the opposite direction.
            - Fraction of that number compared to total number of trains.
            - Speed ratio of the slow train.
            - Speed ratio of the fastest train.
        """
        if not self.has_track_node_pairs:
            return [-float("inf"), -float("inf"), -float("inf"), -float("inf")]
        self_train_detected = False
        train_set = set()
        min_fraction = float("inf")
        max_fraction = 0.0
        for track, node in self.track_node_pairs:
            if track not in self.simulation.occupied_track_parts:
                continue
            for train, _, _, node_toward in self.simulation.occupied_track_parts[track]:
                if train == self.train:
                    self_train_detected = True
                elif node != node_toward and train != self.train and self_train_detected:
                    train_set.add(train)
                    speed_fraction = train.speed / self.train.speed
                    min_fraction = min(min_fraction, speed_fraction)
                    max_fraction = max(max_fraction, speed_fraction)
        num_trains_in_branch = len(train_set)
        if num_trains_in_branch > 0:
            return [num_trains_in_branch, num_trains_in_branch / self._num_trains, min_fraction, max_fraction]
        return [0.0, 0.0, float("inf"), 0.0]

    def produce_features(self):
        """
        Produce the feature for this particular set of rail tracks.
        By indices, the feature is the following
        :return: An np.array representing the features
        """
        features = [
            self._remaining_distance_fraction(),
            self._target_in_branch_features_fraction(),
            self._unusable_switch_features_fraction(),
            self._agent_in_branch_features_fraction(),
            self._conflict_with_crossing_branch_features_fraction(),
            self._minimum_distance_to_target_if_branch_taken_fraction(),
            self._trains_same_direction_features(),
            self._trains_opposite_direction_features(),
        ]
        return np.array([val for values in features for val in values]).astype(float)

    def __repr__(self):
        return "Feature Producer: " + str(self.tracks) + " " + str(self.produce_features())


class Observer:
    def get_observation(self):
        return np.array([])


class TreeObserver(Observer):
    def __init__(self, train: Train, simulation: Simulation, observation_depth: int):
        """
        :param train: The train in which the observer observe through
        :param simulation: The simulation in which the train operations on
        :param observation_depth: The branching depth of the observer, specifying how many transitions forward that
            the observer will investigate to gather features.
        """
        self.train = train
        self.simulation = simulation
        self.observation_depth = observation_depth

    def _get_branch_feature_producers(self) -> List[BranchFeatureProducer]:
        """
        Get all feature producers.
        :return: A list of BranchFeatureProducer representing the observation
        """
        track = self.train.track
        node_toward = self.train.node_toward
        track_position = self.train.track_position
        visited_nodes = set()
        feature_producers: List[BranchFeatureProducer] = []
        self._get_branch_observation(
            track, node_toward, track_position, track, node_toward, visited_nodes,
            feature_producers, 0, self.observation_depth)

        u_turn_track, u_turn_node_toward, u_turn_position = self.train.get_u_turn_track_node_distance()
        self._get_branch_observation(
            u_turn_track, u_turn_node_toward, u_turn_position,
            u_turn_track, u_turn_node_toward, visited_nodes, feature_producers, 0, self.observation_depth)

        return feature_producers

    def get_observation(self):
        """
        Get the train observation
        :return: A numpy list of floating point values representing the observation of the train.
        """
        branch_feature_producers = self._get_branch_feature_producers()
        features = normalize_obs(np.array([val for fp in branch_feature_producers for val in fp.produce_features()]))
        return features

    def _get_branch_observation(self,
                                head_track: Track,
                                head_node: Node,
                                head_position: float,
                                branch_track: Track,
                                branch_node: Node,
                                visited_nodes: Set[Node],
                                resulting_feature_producers: List[BranchFeatureProducer],
                                depth_index: int, max_depth: int):
        """
        Get all observation within the branch
        :param head_track: The track where the head of the train resides
        :param head_node: The node where the head of the train is pointing towards
        :param head_position: The position of the head
        :param branch_track: The track that we are branching the observation
        :param branch_node: The node that we are branching the observation
        :param visited_nodes: A set of node that the branch operation visited
        :param resulting_feature_producers: A list of feature producer, constantly being added that the branching operation
            explores mode of the track map
        :param depth_index: Current depth that the agent is exploring
        :param max_depth: Maximum exploration depth
        :return:
        """
        # Stop exploring if we already reached maximum depth
        if depth_index >= max_depth:
            return

        # If the are tracks, search until the end of the transition to add them in
        track_node_pairs = []
        curr_track = branch_track
        curr_node = branch_node
        left = None
        right = None
        while curr_track is not None:
            # Append the track to the track list
            track_node_pairs.append((curr_track, curr_node))

            # Explore the next node
            next_trans = curr_node.transition
            forward = next_trans.get_node_from_going_forward(curr_node)
            left = next_trans.get_node_from_turning_left(curr_node)
            right = next_trans.get_node_from_turning_right(curr_node)
            if forward is not None and forward not in visited_nodes:
                visited_nodes.add(forward)
                curr_track = forward.track
                curr_node = curr_track.get_other_node(forward)
            else:
                break

        # Add the feature of the current set of tracks
        resulting_feature_producers.append(BranchFeatureProducer(self.train, head_track, head_node, head_position,
                                                                 track_node_pairs, self.simulation))

        # Add the set of features on the left turn
        if left is not None:
            visited_nodes.add(left)
            left_track = left.track
            left_toward_node = left_track.get_other_node(left)
        else:
            left_track = None
            left_toward_node = None
        self._get_branch_observation(
            head_track, head_node, head_position,
            left_track, left_toward_node, visited_nodes, resulting_feature_producers, depth_index + 1, max_depth)

        # Add the set of features on the right turn
        if right is not None:
            visited_nodes.add(right)
            right_track = right.track
            right_toward_node = right_track.get_other_node(right)
        else:
            right_track = None
            right_toward_node = None
        self._get_branch_observation(
            head_track, head_node, head_position,
            right_track, right_toward_node, visited_nodes, resulting_feature_producers, depth_index + 1, max_depth)