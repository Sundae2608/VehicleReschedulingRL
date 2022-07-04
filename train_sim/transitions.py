from enum import Enum
from .tracks import Track, Node
from .constants import *


class TransitionType(Enum):
    STRAIGHT = 1
    SIMPLE_SWITCH = 2
    THREE_WAY_SWITCH = 3
    DIAMOND_CROSSING = 4
    SINGLE_SLIP_SWITCH = 5
    DOUBLE_SLIP_SWITCH = 6
    DEAD_END = 7

    def get_all_types(self):
        return [
            self.DOUBLE_SLIP_SWITCH,
            self.SINGLE_SLIP_SWITCH,
            self.DIAMOND_CROSSING,
            self.SIMPLE_SWITCH,
            self.THREE_WAY_SWITCH,
            self.DEAD_END,
            self.STRAIGHT
        ]


class Transition:
    def __init__(self, x: float, y: float,
                 transition_next_id: int):
        self.type = None
        self.nodes = []

        # ID of the transition.
        self.id = transition_next_id

        # Position of the transition
        self.x = x
        self.y = y

        # Whether this transition is current occupied by a train
        self.is_occupied: bool = False

    def get_new_direction(self,
                          in_node: Node, train_action) -> [Track, Node]:
        """
        Given the in_node and the direction, returning the new track and node representing the new path and direction
        that the train is heading towards. This method will be inherited
        :param in_node:
        :param train_action:
        :return: A pair of Track and Node, representing the path and direction that the train picked.
        """
        return None, None

    def get_nodes(self) -> [Node]:
        """
        :return: All the nodes handled by the transitions
        """
        return self.nodes

    def is_valid_connection(self, track1: Track, track2: Track) -> bool:
        """
        :param track1: Track 1
        :param track2: Track 2
        :return: True if Track 1 and Track 2 are validly connected by the transition
        """
        return False

    def occupy_transition(self):
        """
        Occupy the transtion, meaning that no other trains could collide
        :return:
        """
        self.is_occupied = True

    def unoccupy_transition(self):
        """
        Occupy the transtion, meaning that no other trains could collide
        :return:
        """
        self.is_occupied = False

    def get_node_from_track(self, track: Track):
        for node in self.nodes:
            if node.track == track:
                return node
        return None

    def get_node_from_turning_left(self, in_node: Node) -> Node:
        return None

    def get_node_from_turning_right(self, in_node: Node) -> Node:
        return None

    def get_node_from_going_forward(self, in_node: Node) -> Node:
        return None

    def get_node_from_the_two_sides(self, in_node: Node) -> Node:
        return [node for node in self.nodes if node != in_node and node != self.get_node_from_going_forward(in_node)]


class TransitionStraight(Transition):
    def __init__(self,
                 x: float, y: float,
                 node1: Node,
                 node2: Node,
                 transition_next_id: int):
        """
        A straight just connect two node together
        :param node1, node2: Two nodes involved in the transition.
        """
        super().__init__(x, y, transition_next_id)
        self.type = TransitionType.STRAIGHT
        self.node1 = node1
        self.node2 = node2
        self.nodes = [node1, node2]

    def get_new_direction(self,
                          in_node: Node, train_action) -> (Track, Node):

        if in_node == self.node1:
            out_node = self.node2
        else:
            out_node = self.node1
        new_track = out_node.track
        new_node = new_track.get_other_node(out_node)
        return new_track, new_node

    def is_valid_connection(self, track1: Track, track2: Track) -> bool:
        """
        :param track1: Track 1
        :param track2: Track 2
        :return: True if Track 1 and Track 2 are validly connected by the transition
        """
        return (self.node1.track == track1 and self.node2.track == track2) or \
               (self.node1.track == track2 and self.node2.track == track1)

    def get_node_from_turning_left(self, in_node: Node) -> Node:
        return None

    def get_node_from_turning_right(self, in_node: Node) -> Node:
        return None

    def get_node_from_going_forward(self, in_node: Node) -> Node:
        if in_node == self.node1:
            return self.node2
        elif in_node == self.node2:
            return self.node1
        return None


class TransitionSimpleSwitch(Transition):
    def __init__(self,
                 x: float, y: float,
                 root_node: Node,
                 left_node: Node,
                 right_node: Node,
                 transition_next_id: int):
        """
        In a simple switch:
        - Trains coming from the root can either turn left or right
        - Trains coming from left or right will always go to root node.
        :param root_node, left_node, right_node: Nodes involved in the transition.
        """
        super().__init__(x, y, transition_next_id)
        self.type = TransitionType.SIMPLE_SWITCH
        self.root_node = root_node
        self.left_node = left_node
        self.right_node = right_node
        self.nodes = [root_node, left_node, right_node]

    def get_new_direction(self,
                          in_node: Node, train_action) -> (Track, Node):
        if in_node == self.left_node or in_node == self.right_node:
            out_node = self.root_node
        elif train_action == TRAIN_ACTION_TURN_LEFT:
            out_node = self.left_node
        else:
            out_node = self.right_node
        new_track = out_node.track
        new_node = new_track.get_other_node(out_node)
        return new_track, new_node

    def is_valid_connection(self, track1: Track, track2: Track) -> bool:
        """
        :param track1: Track 1
        :param track2: Track 2
        :return: True if Track 1 and Track 2 are validly connected by the transition
        """
        return (self.root_node.track == track1 and self.left_node.track == track2) or \
               (self.root_node.track == track1 and self.right_node.track == track2) or \
               (self.root_node.track == track2 and self.left_node.track == track1) or \
               (self.root_node.track == track2 and self.right_node.track == track1)

    def get_node_from_turning_left(self, in_node: Node) -> Node:
        if in_node == self.root_node:
            return self.left_node
        return None

    def get_node_from_turning_right(self, in_node: Node) -> Node:
        if in_node == self.root_node:
            return self.right_node
        return None

    def get_node_from_going_forward(self, in_node: Node) -> Node:
        if in_node == self.left_node:
            return self.root_node
        elif in_node == self.right_node:
            return self.root_node
        return None


class TransitionThreeWaySwitch(Transition):
    def __init__(self,
                 x: float, y: float,
                 node1: Node,
                 node2: Node,
                 node3: Node,
                 transition_next_id: int):
        """
        In a 3-way switch:
        - node1 turn left to node3, turn right to node2
        - node2 turn left to node1, turn right to node3
        - node3 turn left to node2, turn right to node1
        :param node1, node2, node3: Nodes involved in the transition.
        """
        super().__init__(x, y, transition_next_id)
        self.type = TransitionType.THREE_WAY_SWITCH
        self.node1 = node1
        self.node2 = node2
        self.node3 = node3
        self.nodes = [node1, node2, node3]

    def get_new_direction(self,
                          in_node: Node, train_action) -> (Track, Node):
        if in_node == self.node1:
            if train_action == TRAIN_ACTION_TURN_LEFT:
                out_node = self.node3
            elif train_action == TRAIN_ACTION_TURN_RIGHT:
                out_node = self.node2
        elif in_node == self.node2:
            if train_action == TRAIN_ACTION_TURN_LEFT:
                out_node = self.node1
            elif train_action == TRAIN_ACTION_TURN_RIGHT:
                out_node = self.node3
        elif in_node == self.node3:
            if train_action == TRAIN_ACTION_TURN_LEFT:
                out_node = self.node2
            elif train_action == TRAIN_ACTION_TURN_RIGHT:
                out_node = self.node1
        new_track = out_node.track
        new_node = new_track.get_other_node(out_node)
        return new_track, new_node

    def is_valid_connection(self, track1: Track, track2: Track) -> bool:
        """
        :param track1: Track 1
        :param track2: Track 2
        :return: True if Track 1 and Track 2 are validly connected by the transition
        """
        return (self.root_node.track == track1 and self.left_node.track == track2) or \
               (self.root_node.track == track1 and self.right_node.track == track2) or \
               (self.root_node.track == track2 and self.left_node.track == track1) or \
               (self.root_node.track == track2 and self.right_node.track == track1)

    def get_node_from_turning_left(self, in_node: Node) -> Node:
        if in_node == self.node1:
            return self.node3
        elif in_node == self.node2:
            return self.node1
        elif in_node == self.node3:
            return self.node2
        return None

    def get_node_from_turning_right(self, in_node: Node) -> Node:
        if in_node == self.node1:
            return self.node2
        elif in_node == self.node2:
            return self.node3
        elif in_node == self.node3:
            return self.node1
        return None

    def get_node_from_going_forward(self, in_node: Node) -> Node:
        return None


class TransitionDiamondCrossing(Transition):
    def __init__(self,
                 x: float, y: float,
                 top_node: Node,
                 bottom_node: Node,
                 left_node: Node,
                 right_node: Node,
                 transition_next_id: int):
        """
        In a double crossing:
        - Trains coming from the top node will go to the bottom node and vice versa
        - Trains coming from the left node will go to the right node and vice versa
        - Trains cannot turn, but only go across
        :param x, y: Position of the transition
        :param top_node, bottom_node, left_node, right_node: Nodes involved in the transition.
        """
        super().__init__(x, y, transition_next_id)
        self.type = TransitionType.DIAMOND_CROSSING
        self.top_node = top_node
        self.bottom_node = bottom_node
        self.left_node = left_node
        self.right_node = right_node
        self.nodes = [top_node, bottom_node, left_node, right_node]

    def get_new_direction(self,
                          in_node: Node, train_action) -> (Track, Node):
        if in_node == self.top_node:
            out_node = self.bottom_node
        elif in_node == self.bottom_node:
            out_node = self.top_node
        elif in_node == self.right_node:
            out_node = self.left_node
        else:
            out_node = self.right_node
        new_track = out_node.track
        new_node = new_track.get_other_node(out_node)
        return new_track, new_node

    def is_valid_connection(self, track1: Track, track2: Track) -> bool:
        """
        :param track1: Track 1
        :param track2: Track 2
        :return: True if Track 1 and Track 2 are validly connected by the transition
        """
        return (self.top_node.track == track1 and self.bottom_node.track == track2) or \
               (self.top_node.track == track2 and self.bottom_node.track == track1) or \
               (self.left_node.track == track1 and self.right_node.track == track2) or \
               (self.left_node.track == track2 and self.right_node.track == track1)

    def get_node_from_turning_left(self, in_node: Node) -> Node:
        return None

    def get_node_from_turning_right(self, in_node: Node) -> Node:
        return None

    def get_node_from_going_forward(self, in_node: Node) -> Node:
        if in_node == self.left_node:
            return self.right_node
        elif in_node == self.right_node:
            return self.left_node
        if in_node == self.bottom_node:
            return self.top_node
        elif in_node == self.top_node:
            return self.bottom_node
        return None


class TransitionSingleSlipSwitch(Transition):
    def __init__(self,
                 x: float, y: float,
                 top_node: Node,
                 bottom_node: Node,
                 left_node: Node,
                 right_node: Node,
                 switch: [Node, Node],
                 transition_next_id: int):
        """
        In a single slip switch:
        - Trains coming from the top node will go to the bottom node and vice versa
        - Trains coming from the left node will go to the right node and vice versa
        - There is a single switch that can connect two adjacent nodes A and B, allow trains to from A to B and B to A.
        :param top_node, bottom_node, left_node, right_node: 4 nodes involved in the transition.
        :param switch: An array of two representing two connected nodes.
        """
        super().__init__(x, y, transition_next_id)
        self.type = TransitionType.SINGLE_SLIP_SWITCH
        self.top_node = top_node
        self.bottom_node = bottom_node
        self.left_node = left_node
        self.right_node = right_node
        self.nodes = [top_node, bottom_node, left_node, right_node]
        self.switch = switch

    def get_new_direction(self,
                          in_node: Node, train_action) -> (Track, Node):
        if in_node == self.top_node:
            if in_node not in self.switch:
                out_node = self.bottom_node
            elif self.left_node in self.switch:
                out_node = self.bottom_node if train_action == TRAIN_ACTION_TURN_LEFT else self.left_node
            elif self.right_node in self.switch:
                out_node = self.right_node if train_action == TRAIN_ACTION_TURN_LEFT else self.bottom_node
        elif in_node == self.bottom_node:
            if in_node not in self.switch:
                out_node = self.top_node
            elif self.left_node in self.switch:
                out_node = self.left_node if train_action == TRAIN_ACTION_TURN_LEFT else self.top_node
            elif self.right_node in self.switch:
                out_node = self.top_node if train_action == TRAIN_ACTION_TURN_LEFT else self.right_node
        elif in_node == self.right_node:
            if in_node not in self.switch:
                out_node = self.left_node
            elif self.top_node in self.switch:
                out_node = self.left_node if train_action == TRAIN_ACTION_TURN_LEFT else self.top_node
            elif self.bottom_node in self.switch:
                out_node = self.bottom_node if train_action == TRAIN_ACTION_TURN_LEFT else self.left_node
        else:
            if in_node not in self.switch:
                out_node = self.right_node
            elif self.top_node in self.switch:
                out_node = self.top_node if train_action == TRAIN_ACTION_TURN_LEFT else self.right_node
            elif self.bottom_node in self.switch:
                out_node = self.right_node if train_action == TRAIN_ACTION_TURN_LEFT else self.bottom_node
        new_track = out_node.track
        new_node = new_track.get_other_node(out_node)
        return new_track, new_node

    def is_valid_connection(self, track1: Track, track2: Track) -> bool:
        """
        :param track1: Track 1
        :param track2: Track 2
        :return: True if Track 1 and Track 2 are validly connected by the transition
        """
        return (self.top_node.track == track1 and self.bottom_node.track == track2) or \
               (self.top_node.track == track2 and self.bottom_node.track == track1) or \
               (self.left_node.track == track1 and self.right_node.track == track2) or \
               (self.left_node.track == track2 and self.right_node.track == track1) or \
               (self.switch[0].track == track1 and self.switch[1].track == track2) or \
               (self.switch[0].track == track2 and self.switch[1].track == track1)

    def get_node_from_going_forward(self, in_node: Node) -> Node:
        if in_node == self.left_node and in_node not in self.switch:
            return self.right_node
        elif in_node == self.right_node and in_node not in self.switch:
            return self.left_node
        if in_node == self.bottom_node and in_node not in self.switch:
            return self.top_node
        elif in_node == self.top_node and in_node not in self.switch:
            return self.bottom_node
        return None

    def get_node_from_turning_left(self, in_node: Node) -> Node:
        if self.left_node in self.switch and self.top_node in self.switch:
            if in_node == self.left_node:
                return self.top_node
            if in_node == self.top_node:
                return self.bottom_node
        if self.left_node in self.switch and self.bottom_node in self.switch:
            if in_node == self.bottom_node:
                return self.left_node
            if in_node == self.left_node:
                return self.right_node
        if self.right_node in self.switch and self.top_node in self.switch:
            if in_node == self.top_node:
                return self.right_node
            if in_node == self.right_node:
                return self.left_node
        if self.right_node in self.switch and self.bottom_node in self.switch:
            if in_node == self.right_node:
                return self.bottom_node
            if in_node == self.bottom_node:
                return self.top_node
        return None

    def get_node_from_turning_right(self, in_node: Node) -> Node:
        if self.left_node in self.switch and self.top_node in self.switch:
            if in_node == self.top_node:
                return self.left_node
            if in_node == self.left_node:
                return self.right_node
        if self.left_node in self.switch and self.bottom_node in self.switch:
            if in_node == self.left_node:
                return self.bottom_node
            if in_node == self.bottom_node:
                return self.top_node
        if self.right_node in self.switch and self.top_node in self.switch:
            if in_node == self.right_node:
                return self.top_node
            if in_node == self.top_node:
                return self.bottom_node
        if self.right_node in self.switch and self.bottom_node in self.switch:
            if in_node == self.bottom_node:
                return self.right_node
            if in_node == self.right_node:
                return self.left_node


class TransitionDoubleSlipSwitch(Transition):
    def __init__(self,
                 x: float, y: float,
                 top_node: Node,
                 bottom_node: Node,
                 left_node: Node,
                 right_node: Node,
                 switch1: [Node, Node],
                 switch2: [Node, Node],
                 transition_next_id: int):
        """
        In a single slip switch:
        - Trains coming from the top node will go to the bottom node and vice versa
        - Trains coming from the left node will go to the right node and vice versa
        - There are two opposite side switch one connect two adjacent nodes A and B, the other connecting remaining
          adjacent nodes C and D. Train can comes from A to B and vice versa, C to D an vice versa.
        :param top_node, bottom_node, left_node, right_node: 4 nodes involved in the transition.
        :param switch1, switch2: Each is an array of two representing two connected nodes.
        """
        super().__init__(x, y, transition_next_id)
        self.type = TransitionType.DOUBLE_SLIP_SWITCH
        self.top_node = top_node
        self.bottom_node = bottom_node
        self.left_node = left_node
        self.right_node = right_node
        self.nodes = [top_node, bottom_node, left_node, right_node]
        self.switch1 = switch1
        self.switch2 = switch2

    def get_new_direction(self,
                          in_node: Node, train_action) -> (Track, Node):
        connected_switch = self.switch1 if in_node in self.switch1 else self.switch2
        if in_node == self.top_node:
            if self.left_node in connected_switch:
                out_node = self.bottom_node if train_action == TRAIN_ACTION_TURN_LEFT else self.left_node
            elif self.right_node in connected_switch:
                out_node = self.right_node if train_action == TRAIN_ACTION_TURN_LEFT else self.bottom_node
        elif in_node == self.bottom_node:
            if self.left_node in connected_switch:
                out_node = self.left_node if train_action == TRAIN_ACTION_TURN_LEFT else self.top_node
            elif self.right_node in connected_switch:
                out_node = self.top_node if train_action == TRAIN_ACTION_TURN_LEFT else self.right_node
        elif in_node == self.right_node:
            if self.top_node in connected_switch:
                out_node = self.left_node if train_action == TRAIN_ACTION_TURN_LEFT else self.top_node
            elif self.bottom_node in connected_switch:
                out_node = self.bottom_node if train_action == TRAIN_ACTION_TURN_LEFT else self.left_node
        else:
            if self.top_node in connected_switch:
                out_node = self.top_node if train_action == TRAIN_ACTION_TURN_LEFT else self.right_node
            elif self.bottom_node in connected_switch:
                out_node = self.right_node if train_action == TRAIN_ACTION_TURN_LEFT else self.bottom_node
        new_track = out_node.track
        new_node = new_track.get_other_node(out_node)
        return new_track, new_node

    def is_valid_connection(self, track1: Track, track2: Track) -> bool:
        """
        :param track1: Track 1
        :param track2: Track 2
        :return: True if Track 1 and Track 2 are validly connected by the transition
        """
        return (self.top_node.track == track1 and self.bottom_node.track == track2) or \
               (self.top_node.track == track2 and self.bottom_node.track == track1) or \
               (self.left_node.track == track1 and self.right_node.track == track2) or \
               (self.left_node.track == track2 and self.right_node.track == track1) or \
               (self.switch1[0].track == track1 and self.switch1[1].track == track2) or \
               (self.switch1[0].track == track2 and self.switch1[1].track == track1) or \
               (self.switch2[0].track == track1 and self.switch2[1].track == track2) or \
               (self.switch2[0].track == track2 and self.switch2[1].track == track1)

    def get_node_from_going_forward(self, in_node: Node) -> Node:
        return None

    def get_node_from_turning_left(self, in_node: Node) -> Node:
        if (self.left_node in self.switch1 and self.top_node in self.switch1) or \
                (self.left_node in self.switch2 and self.top_node in self.switch2):
            if in_node == self.left_node:
                return self.top_node
            if in_node == self.top_node:
                return self.bottom_node
            if in_node == self.right_node:
                return self.bottom_node
            if in_node == self.bottom_node:
                return self.top_node
        else:
            if in_node == self.left_node:
                return self.right_node
            if in_node == self.top_node:
                return self.right_node
            if in_node == self.right_node:
                return self.left_node
            if in_node == self.bottom_node:
                return self.left_node

    def get_node_from_turning_right(self, in_node: Node) -> Node:
        if (self.left_node in self.switch1 and self.top_node in self.switch1) or \
                (self.left_node in self.switch2 and self.top_node in self.switch2):
            if in_node == self.left_node:
                return self.right_node
            if in_node == self.top_node:
                return self.left_node
            if in_node == self.right_node:
                return self.left_node
            if in_node == self.bottom_node:
                return self.right_node
        else:
            if in_node == self.left_node:
                return self.bottom_node
            if in_node == self.top_node:
                return self.bottom_node
            if in_node == self.right_node:
                return self.top_node
            if in_node == self.bottom_node:
                return self.top_node


class TransitionDeadEnd(Transition):
    def __init__(self,
                 x: float, y: float, in_node,
                 transition_next_id: int):
        super().__init__(x, y, transition_next_id)
        self.type = TransitionType.DEAD_END
        self.in_node = in_node
        self.nodes = [in_node]

    def get_new_direction(self,
                          in_node: Node, train_action) -> (Track, Node):
        return in_node.track, in_node.track.get_other_node(in_node)

    def is_valid_connection(self, track1: Track, track2: Track) -> bool:
        """
        Always return False because no connections are possibly valid with a dead end
        """
        return False

    def get_node_from_going_forward(self, in_node: Node) -> Node:
        return None

    def get_node_from_turning_left(self, in_node: Node) -> Node:
        return None

    def get_node_from_turning_right(self, in_node: Node) -> Node:
        return None