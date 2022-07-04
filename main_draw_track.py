"""
Drawing a sample train track
Author: Son Pham
"""

import arcade
import ui.drawer as drawer

from rl.agent import Agent
from rl.observation import TreeObserver
from train_sim.train import TrainType
from train_sim.track_map import *
from train_sim.simulation import Simulation, SimulationParams

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
SCREEN_TITLE = "Starting Template"

CAMERA_SPEED = 5

TRAIN_COLORS = [
    arcade.color.AIR_FORCE_BLUE,
    arcade.color.BOSTON_UNIVERSITY_RED,
    arcade.color.DARK_GREEN,
    arcade.color.ALLOY_ORANGE,
    arcade.color.AMAZON,
    arcade.color.AMARANTH_PINK
]


class TrainSimulation(arcade.Window):
    """
    Main application class.

    NOTE: Go ahead and delete the methods you don't need.
    If you do need a method, delete the 'pass' and replace it
    with your own code. Don't leave 'pass' in this program.
    """

    def __init__(self, width, height, title):
        super().__init__(width, height, title)
        arcade.set_background_color(arcade.color.LIGHT_GRAY)
        self.camera = None
        self.camera_x = 0
        self.camera_dx = 0
        self.camera_y = 0
        self.camera_dy = 0

        # Game controller
        self.pause = False

        # Create a sample back end map
        track_map = TrackMap()
        t1, node1_r, node1_d = track_map.create_transition_straight(0, 0)
        t2, node2_l, node2_d = track_map.create_transition_straight(300, 0)
        t3, node3_u, node3_r = track_map.create_transition_straight(0, 180)
        t4, node4_l, node4_r, node4_d = track_map.create_transition_simple_switch(150, 180)
        t5, node5_u, node5_r, node5_l = track_map.create_transition_simple_switch(300, 180)
        t6, node6_u, node6_r = track_map.create_transition_straight(150, 360)
        t7, node7_u, node7_d, node7_l, node7_r = track_map.create_transition_double_slip_switch(
            480, 180, DoubleSwitchType.TOP_LEFT_BOTTOM_RIGHT
        )
        t8, node8_d, node8_r = track_map.create_transition_straight(480, 45)
        t9, node9_l, node9_d = track_map.create_transition_straight(630, 45)
        t10, node10_l, node10_u, node10_d = track_map.create_transition_simple_switch(630, 180)
        t11, node11_u, node11_d, node11_l, node11_r = track_map.create_transition_single_slip_switch(
            480, 360, SwitchType.BOTTOM_LEFT)
        t12, node12_d, node12_l, node12_u = track_map.create_transition_simple_switch(630, 360)
        t13, node13_u, node13_r = track_map.create_transition_straight(480, 480)
        t14, node14_l, node14_u = track_map.create_transition_straight(630, 480)
        t15, node15_r, node15_d, node15_l = track_map.create_transition_simple_switch(300, 360)
        t16, node16_u = track_map.create_transition_dead_end(300, 480)
        t17, node17_d, node17_u, node17_r = track_map.create_transition_simple_switch(630, 300)
        t18, node18_l = track_map.create_transition_dead_end(810, 300)

        tracks = track_map.create_tracks([
            (node1_d, node3_u, 180),
            (node1_r, node2_l, 300),
            (node3_r, node4_l, 150),
            (node2_d, node5_u, 150),
            (node4_r, node5_l, 150),
            (node4_d, node6_u, 180),
            (node6_r, node15_l, 150),
            (node15_d, node16_u, 120),
            (node15_r, node11_l, 180),
            (node5_r, node7_l, 180),
            (node7_u, node8_d, 135),
            (node8_r, node9_l, 150),
            (node9_d, node10_u, 135),
            (node7_r, node10_l, 150),
            (node7_d, node11_u, 180),
            (node10_d, node17_u, 120),
            (node17_r, node18_l, 180),
            (node17_d, node12_u, 60),
            (node11_r, node12_l, 150),
            (node11_d, node13_u, 120),
            (node13_r, node14_l, 150),
            (node12_d, node14_u, 120),
        ])

        sim_params = SimulationParams(
            num_trains=0, train_length_ratio=1.0, train_speed=100, train_breakdown_timeout=1.0,
            train_breakdown_probability_per_unit_distance=0.0, time_limit=120, default_interval=0.02
        )
        self.simulation = Simulation(track_map, sim_params)
        train1 = self.simulation.add_long_train(100, 170, tracks[0], node1_d, 175, 0.0, 0.0, [])
        train2 = self.simulation.add_long_train(100, 170, tracks[5], node6_u, 175, 0.0, 0.0, [])
        train3 = self.simulation.add_long_train(100, 100, tracks[7], node16_u, 105, 0.0, 0.0, [])
        train4 = self.simulation.add_long_train(100, 130, tracks[12], node10_u, 131, 0.0, 0.0, [])
        self.agent1 = Agent(train1, TreeObserver(train1, self.simulation, 3), self.simulation.rng)

        self.train_colors = {
            train1: TRAIN_COLORS[0],
            train2: TRAIN_COLORS[1],
            train3: TRAIN_COLORS[2],
            train4: TRAIN_COLORS[3],
        }

    def setup(self):
        """ Set up the game variables. Call to re-start the game. """
        # Create your sprites and sprite lists here
        self.camera = arcade.Camera(self.width, self.height)
        pass

    def on_draw(self):
        """
        Render the screen.
        """

        # This command should happen before we start drawing. It will clear
        # the screen to the background color, and erase what we drew last frame.
        self.clear()

        # Activate our Camera
        self.camera.use()
        drawer.draw_track_map(self.simulation)

        for train in self.simulation.train_fleet:
            if train.train_type == TrainType.LONG_TRAIN:
                drawer.draw_long_train(train, self.train_colors[train])

            elif train.train_type == TrainType.POINT_TRAIN:
                drawer.draw_point_train(train, self.train_colors[train])

    def on_update(self, delta_time):
        """
        All the logic to move, and the game logic goes here.
        Normally, you'll call update() on the sprite lists that
        need it.
        """
        if not self.pause:
            self.simulation.step_interval(delta_time)

        # Update the camera position
        self.camera_x += self.camera_dx
        self.camera_y += self.camera_dy
        self.camera.move((self.camera_x, self.camera_y))

    def on_key_press(self, key, key_modifiers):
        """
        Called whenever a key on the keyboard is pressed.

        For a full list of keys, see:
        https://api.arcade.academy/en/latest/arcade.key.html
        """
        if key == arcade.key.W:
            self.camera_dy = CAMERA_SPEED
        if key == arcade.key.S:
            self.camera_dy = -CAMERA_SPEED
        if key == arcade.key.A:
            self.camera_dx = -CAMERA_SPEED
        if key == arcade.key.D:
            self.camera_dx = CAMERA_SPEED
        self.camera.move((self.camera_x, self.camera_y))
        pass

    def on_key_release(self, key, key_modifiers):
        """
        Called whenever the user lets off a previously pressed key.
        """
        if key == arcade.key.W or key == arcade.key.S:
            self.camera_dy = 0
        if key == arcade.key.A or key == arcade.key.D:
            self.camera_dx = 0
        if key == arcade.key.SPACE:
            self.pause = not self.pause
        pass

    def on_mouse_motion(self, x, y, delta_x, delta_y):
        """
        Called whenever the mouse moves.
        """
        pass

    def on_mouse_press(self, x, y, button, key_modifiers):
        """
        Called when the user presses a mouse button.
        """
        pass

    def on_mouse_release(self, x, y, button, key_modifiers):
        """
        Called when a user releases a mouse button.
        """
        pass


def main():
    """ Main function """
    game = TrainSimulation(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    game.setup()
    arcade.run()


if __name__ == "__main__":
    main()