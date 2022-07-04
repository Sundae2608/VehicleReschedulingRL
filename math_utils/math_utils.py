import math

from train_sim.transitions import *


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))


def get_pointing_angle(curr_x, curr_y, pt_x, pt_y):
    return math.atan2(pt_y - curr_y, pt_x - curr_x)


def get_node_direction(t: Transition,
                       n: Node):
    next_trans = n.track.get_other_node(n).transition
    return get_pointing_angle(
        t.x, t.y, next_trans.x, next_trans.y)