from math_utils import math_utils, list_utils
from train_sim.transitions import Transition, TransitionType
from train_sim.track_map import SwitchType, DoubleSwitchType, TrackMap

import networkx as nx
import numpy as np


class TrackMapGenerator:

    MAX_SEED = 2**31 - 1
    MAX_INT = 2**31 - 1

    def __init__(self,
                 average_section_length: float = 500,
                 num_transitions: int = 20,
                 rng_seed: int = 50000):
        self.average_section_length = average_section_length
        self.num_transitions = num_transitions
        self.rng = np.random.default_rng(rng_seed)
        self.graph_retry = 10
        self.map_retry = 100

    def _check_legal_map(self, track_map: TrackMap):
        """
        Check whether the map is legal, meaning that it is possible for trains to go its designated goal
        Return True for now.
        TODO: Check to make sure map generated is legal
        :param track_map: Track map
        :return: True
        """
        return True

    def generate_next_map(self):
        """
        Generate the next map.
        :return:
        """
        # First, generate the list of transitions. The transitions must be eligible
        type_degrees = {
            TransitionType.DEAD_END: 1,
            TransitionType.STRAIGHT: 2,
            TransitionType.SIMPLE_SWITCH: 3,
            TransitionType.THREE_WAY_SWITCH: 3,
            TransitionType.DIAMOND_CROSSING: 4,
            TransitionType.SINGLE_SLIP_SWITCH: 4,
            TransitionType.DOUBLE_SLIP_SWITCH: 4
        }
        type_choices = [
            TransitionType.DEAD_END,
            TransitionType.STRAIGHT,
            TransitionType.SIMPLE_SWITCH,
            TransitionType.THREE_WAY_SWITCH,
            TransitionType.DIAMOND_CROSSING,
            TransitionType.SINGLE_SLIP_SWITCH,
            TransitionType.DOUBLE_SLIP_SWITCH
        ]
        choice_probability = [0, 0.15, 0.15, 0.25, 0.15, 0.15, 0.15]
        graph_generated_successfully = False
        while not graph_generated_successfully:
            while True:
                trans_sequence = list(
                    self.rng.choice(type_choices, self.num_transitions, replace=True, p=choice_probability))
                deg_sequence = [type_degrees[trans_type] for trans_type in trans_sequence]
                if sum(deg_sequence) % 2 == 0:
                    break

            # Secondly, generate the graph
            graph_seed = int(self.rng.integers(0, self.MAX_SEED))
            position_seed = int(self.rng.integers(0, self.MAX_SEED))
            for _ in range(self.graph_retry):
                G = nx.configuration_model(deg_sequence=deg_sequence, seed=graph_seed)
                if not nx.is_connected(G):
                    graph_seed += 1
                else:
                    graph_generated_successfully = True
                    break
        edges = list(G.edges)
        unique_edge_set = set()
        nx.set_edge_attributes(G, 1, "weight")
        for e in edges:
            # Self loop:
            if e[0] == e[1]:
                G.remove_edge(*e)
                new_node_id = len(G.nodes)
                G.add_edge(e[0], new_node_id, weight=2)
                trans_sequence.append(TransitionType.DEAD_END)
                if G.degree(e[0]) == 3:
                    trans_sequence[e[0]] = self.rng.choice(
                        [TransitionType.SIMPLE_SWITCH, TransitionType.THREE_WAY_SWITCH])
                elif G.degree(e[0]) == 2:
                    trans_sequence[e[0]] = TransitionType.STRAIGHT

            # Parallel edges:
            if (e[0], e[1]) in unique_edge_set:
                G.remove_edge(*e)
                new_node_id = len(G.nodes)
                G.add_edge(e[0], new_node_id, weight=3)
                G.add_edge(e[1], new_node_id, weight=3)
                trans_sequence.append(TransitionType.STRAIGHT)
            else:
                unique_edge_set.add((e[0], e[1]))

        # Scale the position of each node so that the average distance of each section is the specified average section
        # length.
        # Seed layout for reproducibility
        node_positions = nx.spring_layout(G, seed=position_seed, iterations=100, weight="weight")
        total_track_length = 0
        num_edges = 0
        for e in G.edges:
            total_track_length += math_utils.euclidean_distance(
                node_positions[e[0]][0], node_positions[e[0]][1],
                node_positions[e[1]][0], node_positions[e[1]][1])
            num_edges += 1
        scaling_factor = self.average_section_length / (total_track_length / num_edges)
        for n in node_positions:
            node_positions[n][0] = (node_positions[n][0] + 1) / 2 * scaling_factor
            node_positions[n][1] = (node_positions[n][1] + 1) / 2 * scaling_factor

        # Now, create an adjacency set and sort each node in clock-wise order, this will help with assigning transitions
        adjacency_set = {}
        for e in G.edges:
            if e[0] not in adjacency_set:
                adjacency_set[e[0]] = [e[1]]
            else:
                adjacency_set[e[0]].append(e[1])
            if e[1] not in adjacency_set:
                adjacency_set[e[1]] = [e[0]]
            else:
                adjacency_set[e[1]].append(e[0])

        # Sort each node in the adjacency list by direction
        for n in adjacency_set:
            adjacency_set[n].sort(key=lambda other_n: math_utils.get_pointing_angle(
                node_positions[n][0], node_positions[n][1],
                node_positions[other_n][0], node_positions[other_n][1]
            ))

        # We keep creating the map until it is legal
        for _ in range(self.map_retry):
            new_map = TrackMap()
            trans_list = []
            for i in range(len(trans_sequence)):
                trans_type = trans_sequence[i]
                x, y = node_positions[i]
                trans: Transition = None
                if trans_type == TransitionType.STRAIGHT:
                    trans = new_map.create_transition_straight(x, y)
                elif trans_type == TransitionType.SIMPLE_SWITCH:
                    trans = new_map.create_transition_simple_switch(x, y)
                elif trans_type == TransitionType.THREE_WAY_SWITCH:
                    trans = new_map.create_transition_three_way_switch(x, y)
                elif trans_type == TransitionType.DIAMOND_CROSSING:
                    trans = new_map.create_transition_diamond_crossing(x, y)
                elif trans_type == TransitionType.SINGLE_SLIP_SWITCH:
                    switch_type = self.rng.choice(SwitchType.get_all_types(SwitchType))
                    trans = new_map.create_transition_single_slip_switch(x, y, switch_type)
                elif trans_type == TransitionType.DOUBLE_SLIP_SWITCH:
                    double_switch_type = self.rng.choice(DoubleSwitchType.get_all_types(DoubleSwitchType))
                    trans = new_map.create_transition_double_slip_switch(x, y, double_switch_type)
                elif trans_type == TransitionType.DEAD_END:
                    trans = new_map.create_transition_dead_end(x, y)
                trans_list.append(trans)

            # Randomly rotate the direction in the adjacency set. This will help with randomization of transition
            # direction
            for n in adjacency_set:
                list_utils.rotate_list(adjacency_set[n], int(self.rng.integers(0, self.MAX_INT)))

            # Connect the node using the track
            for e in G.edges:
                trans_index_1, trans_index_2, _ = e
                trans1 = trans_list[trans_index_1][0]
                trans2 = trans_list[trans_index_2][0]
                node_index_1 = adjacency_set[trans_index_1].index(trans_index_2) + 1
                node_index_2 = adjacency_set[trans_index_2].index(trans_index_1) + 1
                new_map.create_track(
                    trans_list[trans_index_1][node_index_1],
                    trans_list[trans_index_2][node_index_2],
                    math_utils.euclidean_distance(
                        trans1.x, trans1.y, trans2.x, trans2.y
                    ))

            # Check map legality. A map is only legal under certain condition.
            if self._check_legal_map(new_map):
                return new_map, G, node_positions

        raise RuntimeError("Fail to create viable map with current configs")


if __name__ == '__main__':
    tmg = TrackMapGenerator(
        average_section_length=200,
        num_transitions=12,
        rng_seed=53)
    tmg.generate_next_map()
