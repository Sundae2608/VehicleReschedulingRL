import numpy as np
import sys

from enum import Enum
from typing import Tuple, List

from train_sim.constants import *
from train_sim.tracks import Track, Station
from train_sim.transitions import Node, Transition, TransitionType


class TrainType(Enum):
    POINT_TRAIN = 0
    LONG_TRAIN = 1


class TrackTransitionLinkedList:
    def __init__(self, transition: Transition, track: Track, distance: float):
        self.transition = transition
        self.track = track
        self.distance_filled = distance
        self.next: TrackTransitionLinkedList = None
        self.prev: TrackTransitionLinkedList = None


class Train:
    def __init__(self,
                 speed: float, track: Track, track_position: float, direction_node: Node,
                 breakdown_timeout: float, breakdown_probability_per_unit_distance: float,
                 simulation):
        """
        Representing a train as a point on the map.
        """
        # Train type
        self.train_type = TrainType.POINT_TRAIN

        # Speed of the train (unit / sec)
        self.speed = speed

        # Breakdown time amount and breakdown probability
        self.breakdown_timeout = breakdown_timeout
        self.breakdown_probability_per_unit_distance = breakdown_probability_per_unit_distance

        # The track that the train is currently on, and the position is represented as how much the train has covered
        # the track, and direction that the train is heading toward
        self.track = track
        self.track_position = track_position
        self.node_toward = direction_node

        # Whether the train is broken down
        self.is_broken_down = False
        self.time_until_repair = 0

        # The simulation that the train currently operates in:
        self.simulation = simulation
        self.rng = simulation.rng

        # Destination
        self.destination: Station = None
        self.pass_destination = False

        # Controlling agent
        self.agent = None
        self._agent_obs = None
        self._agent_action = None

    def reset_position(self, begin_track, begin_track_position, begin_node_toward):
        """
        Reset the train to a newly assigned begin position
        :param begin_track: The new begin track the train will be placed
        :param begin_track_position: The new track position the train will be placed
        :param begin_node_toward: The direction that the train is headed toward
        """
        # Re-assign the train position
        self.track = begin_track
        self.track_position = begin_track_position
        self.node_toward = begin_node_toward

        # Reset break down status
        self.is_broken_down = False
        self.time_until_repair = 0

    def set_destination(self, destination: Station):
        self.destination = destination

    def set_agent(self, agent):
        self.agent = agent

    def get_agent_state_and_actions(self):
        """
        Get the agent state and action.
        :return: A tuple of two np.arrays, one representing the state vector and the other representing the action
            vector.
        """
        if not self.agent:
            # If there is no agent, return an empty observation and an array of default action
            return np.array([]), self.rng.choice([TRAIN_ACTION_TURN_LEFT, TRAIN_ACTION_TURN_RIGHT])
        return self.agent.get_state_and_actions()

    def get_last_state_and_actions(self):
        return self._agent_obs, self._agent_action

    def get_possible_ahead_room(self):
        """
        Given a train, check to see how much room the train can possibly move ahead.
        :return: True if the train can move ahead, false otherwise.
        """
        # Maximum head room is at most the remain distance to the transition
        headroom = sys.maxsize

        for other_train in self.simulation.train_fleet:
            if self == other_train:
                continue

            # Check the head if the other train is heading in the opposite direction
            if self.track == other_train.track and \
                    other_train.node_toward == self.track.get_other_node(self.node_toward):
                track_remain = self.track.distance - other_train.track_position - self.track_position
                headroom = min(headroom, track_remain)

            # Check the tail if the train ahead is blocking
            if other_train.train_type == TrainType.LONG_TRAIN:
                if other_train.linked_train_tail is None:
                    if self.track == other_train.track and self.node_toward == other_train.node_toward:
                        track_remain = other_train.track_position - other_train.length
                        headroom = min(headroom, track_remain)
                else:
                    if other_train.linked_train_tail.track == self.track:
                        track_remain = self.track.distance - self.track_position - \
                                       other_train.linked_train_tail.distance_filled
                        headroom = min(headroom, track_remain)
        headroom = max(headroom, 0)
        return headroom

    def u_turn(self):
        """
        Turn the train around
        """
        self.node_toward = self.track.get_other_node(self.node_toward)
        self.track_position = 0

    def check_pass_destination(self, distance: float):
        """
        Check if the train would pass the destination if they make the distance
        :param distance: Distance that the train wil make
        :return: True if the destination is passed through, False otherwise
        """
        if not self.destination:
            # Can't pass destination if there isn't any destination
            return False
        if self.track != self.destination.track:
            return False
        if self.node_toward == self.track.node2:
            return self.track_position <= self.destination.position < self.track_position + distance
        if self.node_toward == self.track.node1:
            return (self.track_position <= self.track.distance - self.destination.position <
                    self.track_position + distance)

    def add_head(self, add_distance):
        self.pass_destination = self.check_pass_destination(add_distance)
        self.track_position += add_distance
        if self.track_position > self.track.distance:

            if self.node_toward.transition.is_occupied:
                self.track_position = self.track.distance
            else:
                # If the train reach the an transition, it will now have to process the intersection
                spare_distance = self.track_position - self.track.distance

                # Pick a direction left or right
                if self.node_toward.transition.type != TransitionType.DEAD_END:
                    new_track, new_node = self.node_toward.transition.get_new_direction(
                        self.node_toward, self._agent_action)
                else:
                    self.u_turn()

                self.track = new_track
                self.node_toward = new_node
                self.track_position = 0
                self.step_interval(spare_distance)

    def step_interval(self, interval):
        """
        Simulate the train moving for a certain amount of time.
        :param interval: The amount of time passed
        """
        # A broken train must be repaired
        if self.is_broken_down:
            self.time_until_repair -= interval
            if self.time_until_repair < 0:
                self.is_broken_down = False
            return

        _, self._agent_action = self.get_agent_state_and_actions()
        if self._agent_action == TRAIN_ACTION_REVERSE:
            self.u_turn()
            return

        if self._agent_action == TRAIN_ACTION_BREAK:
            # Proceed only if the train is willing to move forward
            return

        # Calculate potential breakdown
        if self.breakdown_probability_per_unit_distance > 0:
            breakdown_distance = self.rng.geometric(self.breakdown_probability_per_unit_distance)
            if breakdown_distance < interval * self.speed:
                potential_distance = breakdown_distance
                self.is_broken_down = True
                self.time_until_repair = self.breakdown_timeout
            else:
                potential_distance = interval * self.speed
        else:
            potential_distance = interval * self.speed

        add_distance = potential_distance
        headroom = self.get_possible_ahead_room()
        if headroom == 0:
            self.u_turn()
            return
        add_distance = min(headroom, add_distance)

        self.add_head(add_distance)

    def get_u_turn_track_node_distance(self) -> Tuple[Track, Node, float]:
        return self.track, self.track.get_other_node(self.node_toward), self.track.distance - self.track_position

    def occupied_track_parts(self) -> List[Tuple[Track, float, float, Node]]:
        position = self.track.get_distance_from_node1(self.track_position, self.node_toward)
        return [(self.track, position, position, self.node_toward)]


class LongTrain(Train):
    def _create_linked_train_and_check_for_legal_train(
            self, track, track_position, transitions_and_tracks):
        """
        Check to make sure that the input track_length, tracks and transitions represent a legal placement of the train
        """
        # Check that the input tracks and transitions are even
        if len(transitions_and_tracks) % 2 != 0:
            raise RuntimeError("Length of tracks_and_transitions must be even since the input must be a Tracks and "
                               "Transitions one after the other")

        # Construct the linked train
        self.linked_train_head = None
        prev = None
        remain_dist = self.length - track_position
        for i in range(0, len(transitions_and_tracks) - 1, 2):
            if remain_dist > 0:
                raise RuntimeError("Train length did not cover input tracks and transitions")

            trans = transitions_and_tracks[i]
            track = transitions_and_tracks[i + 1]
            if not isinstance(track, Track) or not isinstance(trans, Transition):
                raise RuntimeError("tracks_and_transitions input must be a Transitions and Tracks one after the other")
            linked_node = TrackTransitionLinkedList(trans, track, min(track.distance, remain_dist))
            if prev is None:
                self.linked_train_head = linked_node
            else:
                prev.next = linked_node
                linked_node.prev = prev
            prev = linked_node
            remain_dist -= track.distance
        if remain_dist > 0:
            raise RuntimeError("Input tracks and transitions did not cover the entire train length")
        self.linked_train_tail = prev

        # Check to make sure the transitions are valid
        curr = self.linked_train_head
        while curr:
            if curr == self.linked_train_head:
                if not curr.transition.is_valid_connection(track, curr.track):
                    raise RuntimeError("Transition did not connect two tracks defined in the list")
            else:
                if not curr.transition.is_valid_connection(curr.track, curr.prev.track):
                    raise RuntimeError("Transition did not connect two tracks defined in the list")
            curr = curr.next

    def __init__(self, speed, train_length, track, direction_node, track_position,
                 breakdown_timeout, breakdown_probability_per_unit_distance, tracks_and_transitions, simulation):
        """
        Representing a long train, which a beginning, but also cover the track.
        :param speed:
        :param train_length:
        :param track:
        :param direction_node:
        :param track_position:
        :param tracks_and_transitions:
        """
        super().__init__(speed, track, track_position, direction_node, breakdown_timeout,
                         breakdown_probability_per_unit_distance, simulation)

        # Train type
        self.train_type = TrainType.LONG_TRAIN
        self.length = train_length

        # Check to make sure that the inputs tracks, track position and length is legal
        self.linked_train_head: TrackTransitionLinkedList = None
        self.linked_train_tail: TrackTransitionLinkedList = None
        self._create_linked_train_and_check_for_legal_train(
            track, track_position, tracks_and_transitions)
        
        # Internal variables showing cut distance
        self._cut_distance = 0

    def reset_position(self, begin_track, begin_track_position, begin_node_toward, tracks_and_transitions):
        super().reset_position(begin_track, begin_track_position, begin_node_toward)
        self.linked_train_head: TrackTransitionLinkedList = None
        self.linked_train_tail: TrackTransitionLinkedList = None
        self._create_linked_train_and_check_for_legal_train(
            begin_track, begin_track_position, tracks_and_transitions)

    def u_turn(self):
        """
        U turn for long train is somewhat complicate because the the tail of the train will now become the head of the
        train
        """
        remain_dist = self.length

        # If the train does not exist in other parts of the track, then simply flip the node
        if self.linked_train_head is None:
            self.node_toward = self.track.get_other_node(self.node_toward)
            self.track_position = self.track.distance - self.track_position + self.length
        else:
            # Flip all linked node direction
            curr = self.linked_train_head
            remain_dist = self.length
            temp_track = curr.track
            while curr:
                if curr == self.linked_train_head:
                    remain_dist -= self.track_position
                    curr.track = self.track
                    curr.prev = curr.next
                    curr.next = None
                    # Filled distance has to be cut a bit shorter since these spare distance will be made by the add
                    # function later.
                    # TODO: Perform _cut_distance at the same time as _add_distance
                    curr.distance_filled = self.track_position
                else:
                    remain_dist -= temp_track.distance
                    curr.track, temp_track = temp_track, curr.track
                    curr.distance_filled = curr.track.distance
                    curr.prev, curr.next = curr.next, curr.prev
                curr = curr.prev

            self.track = temp_track
            self.linked_train_head, self.linked_train_tail = self.linked_train_tail, self.linked_train_head
            self.track_position = remain_dist
            self.node_toward = self.track.node1 if self.track.node2 in self.linked_train_head.transition.get_nodes() \
                else self.track.node2

    def cut_tail(self, cut_distance):
        if self.linked_train_tail is None:
            return
        self.linked_train_tail.distance_filled -= cut_distance
        if self.linked_train_tail.distance_filled < 0:
            remain_dist = - self.linked_train_tail.distance_filled
            self.linked_train_tail.transition.unoccupy_transition()
            self.linked_train_tail = self.linked_train_tail.prev
            if self.linked_train_tail is not None:
                self.linked_train_tail.next = None
            else:
                self.linked_train_head = None
            self.cut_tail(remain_dist)

    def add_head(self, add_distance):

        # Calculate possible headroom
        headroom = self.get_possible_ahead_room()
        if headroom == 0:
            self.u_turn()
            return
        add_distance = min(headroom, add_distance)
        self.pass_destination = self.check_pass_destination(add_distance)

        self.track_position += add_distance
        if self.track_position > self.track.distance:
            # If the train reach the an transition, it will now have to process the intersection
            spare_distance = self.track_position - self.track.distance
            self.track_position = self.track.distance
            made_distance = add_distance - spare_distance
            self._cut_distance += made_distance

            # If transition is occupied, the train cannot progress
            if self.node_toward.transition.is_occupied:
                return

            if self.node_toward.transition.type != TransitionType.DEAD_END:
                # Add the new transition and previous track to the linked list
                new_linked_node = TrackTransitionLinkedList(self.node_toward.transition, self.track,
                                                            min(self.track.distance, self.length + made_distance))
                new_linked_node.transition.occupy_transition()
                new_linked_node.next = self.linked_train_head
                if self.linked_train_head is not None:
                    self.linked_train_head.prev = new_linked_node
                else:
                    self.linked_train_tail = new_linked_node
                self.linked_train_head = new_linked_node

                # Determine the new track and direction node
                new_track, new_node = self.node_toward.transition.get_new_direction(self.node_toward, self._agent_action)
                self.track = new_track
                self.node_toward = new_node
                self.track_position = 0
                self.add_head(spare_distance)
            else:
                # If the train reaches the end, don't perform an automatic u-turn.
                self.u_turn()
        else:
            self._cut_distance += add_distance

    def step_interval(self, interval):
        """
        Simulate the train moving for a certain amount of time.
        :param interval: The amount of time passed
        """
        # A broken train must be repaired
        if self.is_broken_down:
            self.time_until_repair -= interval
            if self.time_until_repair < 0:
                self.is_broken_down = False
            return

        self._agent_obs, self._agent_action = self.get_agent_state_and_actions()
        if self._agent_action == TRAIN_ACTION_REVERSE:
            self.u_turn()
            return

        if self._agent_action == TRAIN_ACTION_BREAK:
            # Proceed only if the train is willing to move forward
            return

        # Calculate potential breakdown
        if self.breakdown_probability_per_unit_distance > 0:
            breakdown_distance = self.rng.geometric(self.breakdown_probability_per_unit_distance)
            if breakdown_distance < interval * self.speed:
                potential_distance = breakdown_distance
                self.is_broken_down = True
                self.time_until_repair = self.breakdown_timeout
            else:
                potential_distance = interval * self.speed
        else:
            potential_distance = interval * self.speed

        self._cut_distance = 0
        self.add_head(potential_distance)
        self.cut_tail(self._cut_distance)

    def get_u_turn_track_node_distance(self) -> Tuple[Track, Node, float]:
        """
        :return: a tuple of 3 data points.
        """
        if self.linked_train_tail is None:
            return self.track, self.track.get_other_node(self.node_toward), self.track.distance - self.track_position + self.length
        track = self.linked_train_tail.track
        node = self.linked_train_tail.transition.get_node_from_track(track)
        dist = self.linked_train_tail.distance_filled
        u_turn_node = track.get_other_node(node)
        return track, u_turn_node, dist

    def occupied_track_parts(self) -> List[Tuple[Track, float, float, Node]]:
        tracks = []
        first_position1 = self.track.get_distance_from_node1(self.track_position, self.node_toward)
        first_position2 = self.track.get_distance_from_node1(
            0 if self.linked_train_head is not None else self.track_position - self.length, self.node_toward)
        tracks.append((
            self.track, min(first_position1, first_position2), max(first_position1, first_position2), self.node_toward
        ))

        curr = self.linked_train_head
        while curr:
            track = curr.track
            node_toward = track.get_other_node(curr.transition.get_node_from_track(track))
            position1 = track.get_distance_from_node1(0, node_toward)
            position2 = track.get_distance_from_node1(
                track.distance if curr.next else curr.distance_filled, node_toward)
            tracks.append((
                track, min(position1, position2), max(position1, position2), curr.transition.get_node_from_track(track)
            ))
            curr = curr.next
        return tracks
