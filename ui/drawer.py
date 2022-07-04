import arcade

from train_sim.simulation import Simulation
from train_sim.train import Train, LongTrain
from train_sim.tracks import Station
from math_utils.math_utils import *

TRANSITION_RADIUS = 15
TRAIN_RADIUS = 8

COLOR_TEXT = arcade.color.BLACK
COLOR_RAIL = arcade.color.GRAY
COLOR_TRAIN = arcade.color.CG_RED
COLOR_TRANSITION_RAILS = arcade.color.BLACK
COLOR_TRANSITION = arcade.make_transparent_color(arcade.color.BLACK, 50)
COLOR_TRANSITION_OCCUPIED = arcade.make_transparent_color(arcade.color.RED_DEVIL, 128)

SIZE_TRACK = 3
SIZE_TRAIN = 8
SIZE_STATION = 20
SIZE_STATION_HALF = SIZE_STATION / 2

DRAW_TRANSITION_ID = False
DRAW_TRACK_ID = False

# Max arc angle. If angle is steeper than this value, simply draw the arc as a straight line
MAX_ARC_ANGLE = math.pi * 15 / 18


class Point:
    def __init__(self, x: float, y: float):
        self.x: float = x
        self.y: float = y


class Line:
    def __init__(self, m: float = None, b: float = None, x: float = None):
        """
        Representing a line
        :param m: Slope
        :param b: y-intersection
        :param x: Special case when the line is vertical
        """
        self.m: float = m
        self.b: float = b
        self.x: float = x


def angle_difference(angle1, angle2):
    d = angle2 - angle1
    if d > math.pi:
        return d - 2 * math.pi
    elif d < -math.pi:
        return d + 2 * math.pi
    return d


def get_degrees(rad):
    return rad * 180 / math.pi


def find_distance(p1: Point, p2: Point) -> float:
    """
    :return: Distance between two point
    """
    return math.sqrt((p2.y - p1.y) * (p2.y - p1.y) + (p2.x - p1.x) * (p2.x - p1.x))


def find_cutting_point(line1: Line, line2: Line) -> Point:
    """
    Find cutting point between two lines
    :return: None if lines are parallel, else return intersecting points
    """
    if (line1.x is not None and line2.x is not None) or line1.m == line2.m:
        # Parallel lines don't cut at a single point
        return None

    if line1.x is not None:
        y = line2.m * line1.x + line2.b
        return Point(line1.x, y)

    if line2.x is not None:
        y = line1.m * line2.x + line1.b
        return Point(line2.x, y)

    x = (line2.b - line1.b) / (line1.m - line2.m)
    y = line1.m * x + line1.b
    return Point(x, y)


def find_perpendicular_line(p1: Point, p2: Point, intersect: Point) -> Line:
    """
    Find the line perpendicular to line form by p1 and p2 at intersection point.
    It is given that intersection point resides within p1 and p2.
    """
    # Slope of the first line
    if abs(p2.y - p1.y) <= 1e-10:
        return Line(None, None, intersect.x)
    if abs(p2.x - p1.x) <= 1e-10:
        return Line(0, intersect.y, None)
    dy = p2.y - p1.y
    dx = p2.x - p1.x
    m = -1 / (dy / dx)
    b = -m * intersect.x + intersect.y
    return Line(m, b, None)


def draw_transition_center_to_dir(t, dir, color, size):
    """
    Draw a line from transition center to direction
    """
    # Intersection of those 2 perpendicular lines form the center of the new arc
    arcade.draw_line(t.x,
                     t.y,
                     t.x + math.cos(dir) * TRANSITION_RADIUS,
                     t.y + math.sin(dir) * TRANSITION_RADIUS,
                     color, size)


def draw_transition_connection(t, dir1, dir2, color, size):
    """
    Draw a line that represents connection between two nodes within a transition.
    """
    # Arc must be drawn from left to right
    d = angle_difference(dir1, dir2)
    if d > 0:
        dir2, dir1 = dir1, dir2
    # Find two perpendicular lines at direction.
    p_center = Point(t.x, t.y)
    p1 = Point(t.x + math.cos(dir1) * TRANSITION_RADIUS,
               t.y + math.sin(dir1) * TRANSITION_RADIUS)
    p2 = Point(t.x + math.cos(dir2) * TRANSITION_RADIUS,
               t.y + math.sin(dir2) * TRANSITION_RADIUS)
    line1 = find_perpendicular_line(p_center, p1, p1)
    line2 = find_perpendicular_line(p_center, p2, p2)
    arc_center = find_cutting_point(line1, line2)
    if arc_center is None or angle_difference(dir1, dir2) > MAX_ARC_ANGLE or \
            angle_difference(dir1, dir2) < - MAX_ARC_ANGLE:
        arcade.draw_line(t.x + math.cos(dir1) * TRANSITION_RADIUS,
                         t.y + math.sin(dir1) * TRANSITION_RADIUS,
                         t.x + math.cos(dir2) * TRANSITION_RADIUS,
                         t.y + math.sin(dir2) * TRANSITION_RADIUS,
                         color, size)
    else:
        arc_width = find_distance(arc_center, p1) * 2
        angle1 = math.atan2(p1.y - arc_center.y, p1.x - arc_center.x)
        angle2 = math.atan2(p2.y - arc_center.y, p2.x - arc_center.x)
        if angle1 <= 0:
            angle1 += math.pi * 2
        if angle2 <= 0:
            angle2 += math.pi * 2
        if angle1 > angle2:
            angle1 -= math.pi * 2
        arcade.draw_arc_outline(arc_center.x, arc_center.y,
                                arc_width, arc_width, color,
                                get_degrees(angle1),
                                get_degrees(angle2),
                                size * 2, 0, 100)


def draw_transition(t: Transition):

    # Draw transition ID
    if DRAW_TRANSITION_ID:
        arcade.draw_text(str(t.id),
                         t.x + 5, t.y + 10, COLOR_TEXT, 10)

    arcade.draw_circle_filled(t.x, t.y,
                              TRANSITION_RADIUS, COLOR_TRANSITION)
    if t.type == TransitionType.STRAIGHT:
        dir_node_1 = get_node_direction(t, t.node1)
        dir_node_2 = get_node_direction(t, t.node2)
        draw_transition_connection(t, dir_node_1, dir_node_2, COLOR_TRANSITION_RAILS, SIZE_TRACK)
    elif t.type == TransitionType.SIMPLE_SWITCH:
        dir_root = get_node_direction(t, t.root_node)
        dir_left = get_node_direction(t, t.left_node)
        dir_right = get_node_direction(t, t.right_node)
        draw_transition_connection(t, dir_root, dir_left, COLOR_TRANSITION_RAILS, SIZE_TRACK)
        draw_transition_connection(t, dir_root, dir_right, COLOR_TRANSITION_RAILS, SIZE_TRACK)
    elif t.type == TransitionType.THREE_WAY_SWITCH:
        dir_node_1 = get_node_direction(t, t.node1)
        dir_node_2 = get_node_direction(t, t.node2)
        dir_node_3 = get_node_direction(t, t.node3)
        draw_transition_connection(t, dir_node_1, dir_node_2, COLOR_TRANSITION_RAILS, SIZE_TRACK)
        draw_transition_connection(t, dir_node_1, dir_node_3, COLOR_TRANSITION_RAILS, SIZE_TRACK)
        draw_transition_connection(t, dir_node_2, dir_node_3, COLOR_TRANSITION_RAILS, SIZE_TRACK)
    elif t.type == TransitionType.DIAMOND_CROSSING:
        dir_top = get_node_direction(t, t.top_node)
        dir_bottom = get_node_direction(t, t.bottom_node)
        dir_left = get_node_direction(t, t.left_node)
        dir_right = get_node_direction(t, t.right_node)
        draw_transition_connection(t, dir_left, dir_right, COLOR_TRANSITION_RAILS, SIZE_TRACK)
        draw_transition_connection(t, dir_top, dir_bottom, COLOR_TRANSITION_RAILS, SIZE_TRACK)
    elif t.type == TransitionType.SINGLE_SLIP_SWITCH:
        dir_top = get_node_direction(t, t.top_node)
        dir_bottom = get_node_direction(t, t.bottom_node)
        dir_left = get_node_direction(t, t.left_node)
        dir_right = get_node_direction(t, t.right_node)
        draw_transition_connection(t, dir_left, dir_right, COLOR_TRANSITION_RAILS, SIZE_TRACK)
        draw_transition_connection(t, dir_top, dir_bottom, COLOR_TRANSITION_RAILS, SIZE_TRACK)
        node1, node2 = t.switch
        dir_node1, dir_node2 = get_node_direction(t, node1), get_node_direction(t, node2)
        draw_transition_connection(t, dir_node1, dir_node2, COLOR_TRANSITION_RAILS, SIZE_TRACK)
    elif t.type == TransitionType.DOUBLE_SLIP_SWITCH:
        dir_top = get_node_direction(t, t.top_node)
        dir_bottom = get_node_direction(t, t.bottom_node)
        dir_left = get_node_direction(t, t.left_node)
        dir_right = get_node_direction(t, t.right_node)
        draw_transition_connection(t, dir_left, dir_right, COLOR_TRANSITION_RAILS, SIZE_TRACK)
        draw_transition_connection(t, dir_top, dir_bottom, COLOR_TRANSITION_RAILS, SIZE_TRACK)
        node1, node2 = t.switch1
        dir_node1, dir_node2 = get_node_direction(t, node1), get_node_direction(t, node2)
        draw_transition_connection(t, dir_node1, dir_node2, COLOR_TRANSITION_RAILS, SIZE_TRACK)
        node3, node4 = t.switch2
        dir_node3, dir_node4 = get_node_direction(t, node3), get_node_direction(t, node4)
        draw_transition_connection(t, dir_node3, dir_node4, COLOR_TRANSITION_RAILS, SIZE_TRACK)
    elif t.type == TransitionType.DEAD_END:
        dir_node = get_node_direction(t, t.in_node)
        draw_transition_center_to_dir(t, dir_node, COLOR_TRANSITION_RAILS, SIZE_TRACK)


def draw_track(track: Track):

    # Offset by the space for the Transition:
    x1, y1 = track.node1.transition.x, track.node1.transition.y
    x2, y2 = track.node2.transition.x, track.node2.transition.y
    dist = euclidean_distance(x1, y1, x2, y2)
    if dist < 2 * TRANSITION_RADIUS:
        # Don't draw if there isn't enough space to draw track
        return

    # Draw Track ID
    if DRAW_TRACK_ID:
        arcade.draw_text(str(track.id),
                         (x1 + x2) / 2 + 5, (y1 + y2) / 2 + 10, COLOR_TEXT, 10)

    # Shorten each side of the line by TRANSITION_RADIUS:
    x1, x2 = x2 + (x1 - x2) * (dist - TRANSITION_RADIUS) / dist, x1 + (x2 - x1) * (dist - TRANSITION_RADIUS) / dist
    y1, y2 = y2 + (y1 - y2) * (dist - TRANSITION_RADIUS) / dist, y1 + (y2 - y1) * (dist - TRANSITION_RADIUS) / dist
    arcade.draw_line(x1, y1, x2, y2, COLOR_RAIL, 3.0)


def get_position(track, node_toward, track_position):
    """
    Back on the track, the node direction and the track position, returns real x, y value representing the graphical
    position.
    """
    # Get the position of both ends of the tracks
    x1, y1 = track.node1.transition.x, track.node1.transition.y
    x2, y2 = track.node2.transition.x, track.node2.transition.y

    # Offset each point closer to each other by the transition radius
    if node_toward == track.node1:
        x = x2 + (x1 - x2) * track_position / track.distance
        y = y2 + (y1 - y2) * track_position / track.distance
    else:
        x = x1 + (x2 - x1) * track_position / track.distance
        y = y1 + (y2 - y1) * track_position / track.distance
    return x, y


def draw_train_head(train: Train, train_color):
    # First, find the transition position:
    x, y = get_position(train.track, train.node_toward, train.track_position)
    arcade.draw_circle_filled(x, y, TRAIN_RADIUS, train_color)


def draw_point_train(train: Train, train_color):
    # Point train only needs the head
    draw_train_head(train, train_color)


def draw_long_train(train: LongTrain, train_color):
    # First draw the train head
    draw_train_head(train, train_color)

    # Then draw the remaining parts represented by the linked list
    curr = train.linked_train_head
    x1, y1 = get_position(train.track, train.node_toward, train.track_position)
    if train.linked_train_head is None:
        x2, y2 = get_position(train.track, train.node_toward, train.track_position - train.length)
    else:
        x2, y2 = get_position(train.track, train.node_toward, 0)
    arcade.draw_line(x1, y1, x2, y2, train_color, SIZE_TRAIN)
    while curr:
        node_toward = curr.track.node2 if curr.track.node1 in curr.transition.get_nodes() else curr.track.node1
        x1, y1 = get_position(curr.track, node_toward, 0)
        if curr.next:
            x2, y2 = get_position(curr.track, node_toward, curr.track.distance)
        else:
            x2, y2 = get_position(curr.track, node_toward, curr.distance_filled)
        arcade.draw_line(x1, y1, x2, y2, train_color, SIZE_TRAIN)
        curr = curr.next


def draw_track_map(simulation: Simulation):
    track_map = simulation.track_map
    for track in track_map.tracks:
        draw_track(track)

    for transition in track_map.transitions:
        draw_transition(transition)


def draw_station(station: Station, color, station_visited):
    # Modify color based on whether the station is visited
    modified_color = arcade.get_four_byte_color([color[0], color[1], color[2], 128]) if not station_visited else color
    x, y = get_position(station.track, station.track.node2, station.position)
    arcade.draw_rectangle_filled(x, y, SIZE_STATION, SIZE_STATION, modified_color)
