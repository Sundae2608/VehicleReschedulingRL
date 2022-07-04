import numpy as np


class Node:
    def __init__(self, node_id: int):
        """
        A node connects a track to a transition
        """
        self.track = None
        self.transition = None
        self.id = node_id


class Track:
    def __init__(self, node1, node2, distance, track_id: int):
        """
        A track connect two nodes together and is un interrupted.
        :param node1:
        :param node2:
        """
        self.node1 = node1
        self.node2 = node2
        self.distance = distance
        self.id = track_id
        self.stations = []

    def get_other_node(self, node):
        if node != self.node1 and node != self.node2:
            raise RuntimeError("Node does not exist in this track")
        if node == self.node1:
            return self.node2
        else:
            return self.node1

    def add_station(self, position):
        """
        Add a station to the track
        """
        station = Station(self, position)
        self.stations.append(station)
        return station

    def get_nodes(self):
        return [self.node1, self.node2]

    def get_distance_from_node1(self, distance_covered: float, node_toward: Node):
        """
        Get the position from node1. Since a track connects two nodes, we select position from node1 as canonical
        representation of the position inside the track.
        :param distance_covered: Distance covered
        :param node_toward: Node that towards
        :return: The position as measured from node 1.
        """
        return distance_covered if node_toward == self.node2 else self.distance - distance_covered

    def __repr__(self):
        return "Track " + str(self.id)


class Station:
    def __init__(self,
                 track: Track,
                 position: float):
        """
        A station on the track
        :param track: The track that the station resides
        :param position on the track, represented as the position from node1 of the track
        """
        self.track = track
        self.position = position
        pass

