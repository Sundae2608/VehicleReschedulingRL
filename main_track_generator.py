"""
Generate a random track based on the graph algorithm, and run a quick simulation on said track.
Author: Son Pham
"""
import arcade
import ui.drawer as drawer

from rl.agent import DefaultAgent, RandomAgent
from rl.observation import TreeObserver
from train_sim.simulation import Simulation, SimulationParams
from train_sim.train import TrainType
from train_sim.track_map_utils import *

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

    def __init__(self, width, height, title, average_track_length, num_transitions, num_trains):
        super().__init__(width, height, title)
        arcade.set_background_color(arcade.color.WHITE)
        self.camera = None
        self.camera_x = 0
        self.camera_dx = 0
        self.camera_y = 0
        self.camera_dy = 0

        # Game controller
        self.pause = False

        # Create a sample back end map
        self.tmg = TrackMapGenerator(average_track_length, num_transitions)
        self.sim_params = SimulationParams(10.0, 0.02, num_trains, 0.70, 300, 0.4, 0.0001, 10000, 100)
        self.simulation = Simulation(self.tmg.generate_next_map()[0], self.sim_params)

        # Assign AI to the trains
        for i, train in enumerate(self.simulation.train_fleet):
            train.set_agent(DefaultAgent(train, TreeObserver(train, self.simulation, 2), self.simulation.rng))

        # Assign color to the trains
        self.train_colors = {}
        for i, train in enumerate(self.simulation.train_fleet):
            self.train_colors[train] = TRAIN_COLORS[i]

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

        for train in self.simulation.train_goals:
            goal_station = self.simulation.train_goals[train]
            drawer.draw_station(goal_station, self.train_colors[train], self.simulation.train_goals_reached[train])

    def on_update(self, delta_time):
        """
        All the logic to move, and the game logic goes here.
        Normally, you'll call update() on the sprite lists that
        need it.
        """
        if not self.pause:
            self.simulation.step_interval(delta_time)

        if self.simulation.sim_ended:
            self.simulation = Simulation(self.tmg.generate_next_map()[0], self.sim_params)

            # Assign AI to the trains
            self.train_colors = {}
            for i, train in enumerate(self.simulation.train_fleet):
                train.set_agent(DefaultAgent(train, TreeObserver(train, self.simulation, 2), self.simulation.rng))

            # Assign color to the trains
            self.train_colors = {}
            for i, train in enumerate(self.simulation.train_fleet):
                self.train_colors[train] = TRAIN_COLORS[i]

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
    np.random.seed(801)
    game = TrainSimulation(
        1280, 720, "Track generator", average_track_length=300, num_transitions=20, num_trains=6)
    game.setup()
    arcade.run()


if __name__ == "__main__":
    main()
