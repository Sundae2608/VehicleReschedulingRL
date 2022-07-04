from enum import Enum

class TransitionType(Enum):
    SIMPLE_SWITCH = 1
    DIAMOND_CROSSING = 2
    SINGLE_SLIP_SWITCH = 3
    DOUBLE_SLIP_SWITCH = 4
    DEAD_END = 5

class Transition:
    def __init__(self):
        self.type = None
        
class TransitionSimpleSwitch(Node):
    def __init__(self, root_node, left_node, right_node):
        """
        In a simple switch, training coming from the root can either turn left or right, but train coming from left or right will always go to root node.
        """
        self.root