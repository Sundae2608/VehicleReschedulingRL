import heapq

from typing import List, Tuple

from .transitions import *


class SwitchType(Enum):
    TOP_LEFT = 1
    TOP_RIGHT = 2
    BOTTOM_LEFT = 3
    BOTTOM_RIGHT = 4

    def get_all_types(self):
        return [
            self.TOP_LEFT,
            self.TOP_RIGHT,
            self.BOTTOM_LEFT,
            self.BOTTOM_RIGHT,
        ]


class DoubleSwitchType(Enum):
    TOP_LEFT_BOTTOM_RIGHT = 1
    BOTTOM_LEFT_TOP_RIGHT = 2

    def get_all_types(self):
        return [
            self.TOP_LEFT_BOTTOM_RIGHT,
            self.BOTTOM_LEFT_TOP_RIGHT,
        ]


class TrackState:
    def __init__(self):
        pass


class TrackMap:
    def __init__(self):
        """
        A track map contains a set of all tracks, nodes and transitions. The work flow to create new map is:
        - Create a new track
        """
        # Store all nodes, tracks and transitions in the map
        self.nodes = set()
        self.tracks = set()
        self.transitions = set()

        # ID count, used when creating new node, track and transition to ensure each has a unique ID
        self.node_next_id = 0
        self.track_next_id = 0
        self.transition_next_id = 0

        # Some useful features to store in the grapj
        self.total_track_distance = 0

        # State of the track, contain information about what train is in there.
        self.track_state = dict()

    def _create_node(self):
        """
        Create a node and add to the main map
        :return: Return the created node
        """
        node = Node(self.node_next_id)
        self.node_next_id += 1
        self.nodes.add(node)
        return node

    def create_track(self,
                     node1: Node,
                     node2: Node,
                     distance: float):
        """
        Create a track between two existing nodes in the map. The input must already exist in the map
        :param node1: Node 1
        :param node2: Node 2
        :param distance: The distance between two nodes
        :return: The newly created track.
        """
        if node1 not in self.nodes or node2 not in self.nodes:
            raise RuntimeError("Input nodes must first be added to the map")

        if node1.track or node2.track:
            raise RuntimeError("Input nodes must be unconnected to a track")

        track = Track(node1, node2, distance, self.track_next_id)
        self.track_next_id += 1
        node1.track = track
        node2.track = track
        self.tracks.add(track)
        self.total_track_distance += track.distance
        return track

    def create_tracks(self, track_connection_list: List[Tuple[Node, Node, float]]):
        """
        Create multiple tracks given the list of track connections
        :param track_connection_list: A List in which each connection is a list of 2 nodes, and distance between the two
        :return: A list of new tracks
        """
        tracks = []
        for node1, node2, distance in track_connection_list:
            tracks.append(self.create_track(node1, node2, distance))
        return tracks

    def create_transition_straight(self, x, y):
        """
        Create a straight connection between two nodes
        :param x, y: Position of the transition
        :return: Newly created transition and two nodes connected by the transition
        """
        node1 = self._create_node()
        node2 = self._create_node()
        t = TransitionStraight(x, y, node1, node2, self.transition_next_id)
        self.transition_next_id += 1
        node1.transition = t
        node2.transition = t
        self.transitions.add(t)
        return t, node1, node2

    def create_transition_simple_switch(self, x: float, y: float):
        """
        Create a simple switch between three nodes
        :param x, y: Position of the transition
        :return: Newly created transition and 3 nodes in the order of root, left, right
        """
        root_node = self._create_node()
        left_node = self._create_node()
        right_node = self._create_node()
        t = TransitionSimpleSwitch(x, y, root_node, left_node, right_node, self.transition_next_id)
        self.transition_next_id += 1
        root_node.transition = t
        left_node.transition = t
        right_node.transition = t
        self.transitions.add(t)
        return t, root_node, left_node, right_node

    def create_transition_three_way_switch(self, x: float, y: float):
        """
        Create a simple switch between three nodes
        :param x, y: Position of the transition
        :return: Newly created transition and 3 nodes in the order of node1, node2, node3
        """
        node1 = self._create_node()
        node2 = self._create_node()
        node3 = self._create_node()
        t = TransitionThreeWaySwitch(x, y, node1, node2, node3, self.transition_next_id)
        self.transition_next_id += 1
        node1.transition = t
        node2.transition = t
        node3.transition = t
        self.transitions.add(t)
        return t, node1, node2, node3

    def create_transition_diamond_crossing(self, x: float, y: float):
        """
        Create a diamond-crossing between 4 nodes
        :param x, y: Position of the transition
        :return: Newly created transition and 4 nodes in the order of top, bottom, left, right
        """
        top_node = self._create_node()
        bottom_node = self._create_node()
        left_node = self._create_node()
        right_node = self._create_node()
        t = TransitionDiamondCrossing(x, y, top_node, bottom_node, left_node, right_node, self.transition_next_id)
        self.transition_next_id += 1
        top_node.transition = t
        bottom_node.transition = t
        left_node.transition = t
        right_node.transition = t
        self.transitions.add(t)
        return t, top_node, right_node, bottom_node, left_node

    def create_transition_single_slip_switch(self, x: float, y: float, switch_type: SwitchType):
        """
        Create a single-slip switch between 4 nodes
        :param x, y: Position of the transition
        :param switch_type: Type of switch of the transition
        :return: Newly created single-slip Switch transition with 4 nodes in the order of top, bottom, left, right
        """

        top_node = self._create_node()
        bottom_node = self._create_node()
        left_node = self._create_node()
        right_node = self._create_node()
        if switch_type == SwitchType.BOTTOM_LEFT:
            switch = [bottom_node, left_node]
        elif switch_type == SwitchType.BOTTOM_RIGHT:
            switch = [bottom_node, right_node]
        elif switch_type == SwitchType.TOP_LEFT:
            switch = [top_node, left_node]
        elif switch_type == SwitchType.TOP_RIGHT:
            switch = [top_node, right_node]
        t = TransitionSingleSlipSwitch(x, y, top_node, bottom_node, left_node, right_node, switch,
                                       self.transition_next_id)
        self.transition_next_id += 1
        top_node.transition = t
        bottom_node.transition = t
        left_node.transition = t
        right_node.transition = t
        self.transitions.add(t)
        return t, top_node, right_node, bottom_node, left_node

    def create_transition_double_slip_switch(self, x: float, y: float, double_switch_type: DoubleSwitchType):
        """
        Create a single-slip switch between 4 nodes
        :param x, y: Position of the transition
        :param double_switch_type: Type of double switch
        :return: Newly created double-slip switch transition with 4 nodes in order of top, bottom, left, right
        """
        top_node = self._create_node()
        bottom_node = self._create_node()
        left_node = self._create_node()
        right_node = self._create_node()
        if double_switch_type == DoubleSwitchType.TOP_LEFT_BOTTOM_RIGHT:
            switch1 = [top_node, left_node]
            switch2 = [bottom_node, right_node]
        elif double_switch_type == DoubleSwitchType.BOTTOM_LEFT_TOP_RIGHT:
            switch1 = [bottom_node, left_node]
            switch2 = [top_node, right_node]
        t = TransitionDoubleSlipSwitch(x, y, top_node, bottom_node, left_node, right_node, switch1, switch2,
                                       self.transition_next_id)
        self.transition_next_id += 1
        top_node.transition = t
        bottom_node.transition = t
        left_node.transition = t
        right_node.transition = t
        self.transitions.add(t)
        return t, top_node, right_node, bottom_node, left_node

    def create_transition_dead_end(self, x: float, y: float):
        """
        Create a dead-end
        :param x, y: Position of the transition
        :return: Newly created dead-end transition with the in node
        """
        in_node = self._create_node()
        t = TransitionDeadEnd(x, y, in_node, self.transition_next_id)
        self.transition_next_id += 1
        in_node.transition = t
        self.transitions.add(t)
        return t, in_node

    def minimum_distance_from_track_position_to_transition(self, starting_track: Track, starting_node_toward: Node,
                                                           starting_track_position: float,
                                                           destination_transition: Transition):
        """
        Minimum distance from position to transition
        :param starting_track: Starting track
        :param starting_node_toward: Starting direction
        :param starting_track_position: Starting track position
        :param destination_transition: Transition that we are finding minimum distance towards.
        :return: Minimum distance to reach the transition on the map
        """
        visited_trans = set()
        distance_to_node = starting_track.distance - starting_track_position
        frontier_nodes = [(distance_to_node, starting_node_toward)]

        while len(frontier_nodes) > 0:
            dist, node = heapq.heappop(frontier_nodes)
            trans = node.transition
            if trans in visited_trans:
                continue
            visited_trans.add(trans)
            if node.transition == destination_transition:
                return distance_to_node

            # Try to go forward
            forward = node.transition.get_node_from_going_forward(node)
            if forward is not None:
                forward_track = forward.track
                forward_node = forward_track.get_other_node(forward)
                forward_distance = dist + forward_track.distance
                heapq.heappush(frontier_nodes, (forward_distance, forward_node))

            # Try to go left
            left = node.transition.get_node_from_turning_left(node)
            if left is not None:
                left_track = left.track
                left_node = left_track.get_other_node(left)
                left_distance = dist + left_track.distance
                heapq.heappush(frontier_nodes, (left_distance, left_node))

            # Try to go right
            right = node.transition.get_node_from_turning_right(node)
            if right is not None:
                right_track = right.track
                right_node = right_track.get_other_node(right)
                right_distance = dist + right_track.distance
                heapq.heappush(frontier_nodes, (right_distance, right_node))
        return float('inf')

    def minimum_distance_to_certain_point_through_track(self,
                                                        curr_track: Track, curr_node_toward: Node,
                                                        curr_track_position: float,
                                                        through_track: Track, through_node_toward: Node,
                                                        destination_track: Track,
                                                        destination_position_from_node1: float):
        """
        :param curr_track: Starting track
        :param curr_node_toward: Starting node direction
        :param curr_track_position: Staring position in the track
        :param through_track: The track in which the agent must pass through
        :param through_node_toward: The direction of the track in which the agent must follow
        :param destination_track: The track that we aims to get to.
        :param destination_position_from_node1: The destination track that we aims to get to.
        :return: Minimum distance to destination that go through certain track and direction.
        """
        # We will split into two searches. We will search from the current track to the through track, and from the
        # through track to the destination track.

        # First search
        trans = through_track.get_other_node(through_node_toward).transition
        minimum_first_part_distance = self.minimum_distance_from_track_position_to_transition(
            curr_track, curr_node_toward, curr_track_position, trans)

        if destination_track == through_track:
            if through_node_toward == destination_track.node1:
                return minimum_first_part_distance + destination_track.distance - destination_position_from_node1
            else:
                return minimum_first_part_distance + destination_position_from_node1

        # Second search. You can come from either direction of the destination track
        trans1 = destination_track.node1.transition
        second_part_distance1 = self.minimum_distance_from_track_position_to_transition(
            through_track, through_node_toward, 0.0, trans1) + destination_position_from_node1

        trans2 = destination_track.node2.transition
        second_part_distance2 = self.minimum_distance_from_track_position_to_transition(
            through_track, through_node_toward, 0.0, trans2
        ) + destination_track.distance - destination_position_from_node1

        return min(minimum_first_part_distance + second_part_distance1,
                   minimum_first_part_distance + second_part_distance2)
