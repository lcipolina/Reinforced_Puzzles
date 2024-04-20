import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import gymnasium as gym
from gym import spaces

class Piece:
    '''represents individual puzzle pieces. Each piece has an ID, a list of sides (each side having some value),
    and a list to track which sides are still available for connection.'''

    def __init__(self, id, sides):
        self.id = id
        self.sides = sides  # List of side values
        self.available_sides = [True] * len(sides)  # All sides are initially available for connection

    def rotate(self, degrees):
         # Allows the piece to be rotated in 90-degree increments.  This method updates the sides and available_sides
        if degrees % 90 != 0:
            return
        num_rotations = degrees // 90
        self.sides = self.sides[-num_rotations:] + self.sides[:-num_rotations]
        self.available_sides = self.available_sides[-num_rotations:] + self.available_sides[:-num_rotations]

    def connect_side(self, side_index):
        # Mark a side as no longer available once it's connected. Returns True if the side was available
        if self.available_sides[side_index]:
            self.available_sides[side_index] = False
            return True
        return False

    def copy(self):
        # Create a new Piece instance with the same id and sides, but independent available_sides
        # This is useful for maintaining the original piece while updating the available sides. Used when resetting the environment.
        new_piece = Piece(self.id, self.sides.copy())
        new_piece.available_sides = self.available_sides.copy()  # Ensure the availability status is also copied
        return new_piece

    @classmethod
    def _generate_pieces(cls, sides):
        pieces = []
        for i in range(len(sides)):
            random.shuffle(sides)
            piece = cls(i + 1, sides.copy())
            pieces.append(piece)
        return pieces


class PuzzleEnvironment:
    def __init__(self, config=None):
        self.sides = config.get("sides", [1, 2, 3, 4])  # Default sides if not provided
        self.pieces = Piece._generate_pieces(self.sides)
        self.target_puzzle = nx.Graph()  # Target configuration as a graph.
        self.current_puzzle = nx.Graph()  # Current state of the puzzle.
        self._setup_target_puzzle()
        self.reset()

    def check_completion(self):
         # Check if the current puzzle is a complete match with the target puzzle
        return nx.is_isomorphic(self.current_puzzle, self.target_puzzle)

    def _setup_target_puzzle(self):
        # Defines the target puzzle structure.
        for piece in self.pieces:
            self.target_puzzle.add_node(piece.id, sides=piece.sides)
        # Assuming edges are predefined (this should be adjusted based on actual puzzle rules)
        for i in range(len(self.pieces) - 1):
            self.target_puzzle.add_edge(self.pieces[i].id, self.pieces[i + 1].id)
        # Add an edge between the first and last piece to close the loop
        self.target_puzzle.add_edge(self.pieces[0].id, self.pieces[-1].id)


    def reset(self):
        ''' Puzzle starts blank with all pieces available and no connections made. Returns the initial observation.'''
        self.current_puzzle.clear()  # Clear the current puzzle state
        self.available_pieces = [piece.copy() for piece in self.pieces]   # Reset available pieces to a deep copy of all pieces, simulating all pieces scattered on the table
        self.available_connections = []  # Initialize empty list
        self.update_available_connections()  # Initialize connections  based on available pieces and their sides (this will be updated after each action)
        return self._get_observation()


    def update_available_connections(self):
        # This method updates the list of available connections based on the current state of the puzzle.
        self.available_connections = []  # Clear existing connections
        for node_id, attrs in self.current_puzzle.nodes(data=True):
            piece = attrs['piece']
            for side_index, is_available in enumerate(piece.available_sides):
                if is_available:
                    self.available_connections.append([node_id, side_index])
        self.available_connections = np.array(self.available_connections, dtype=np.uint8)
        return self.available_connections


    def _get_observation(self):
        # Returns the current state of the environment as an observation. This includes the current puzzle graph, available pieces, and available connections.
        graph_data = nx.to_numpy_array(self.current_puzzle, dtype=np.uint8) # Converts current puzzle graph as an adjacency matrix (for gym env)
        # Available pieces and their details (e.g., sides, orientation)
        available_pieces_data = np.array(
            [[piece.id] + piece.sides for piece in self.available_pieces],
            dtype=np.uint8
        )

        observation = {
            "current_puzzle": graph_data,
            "available_pieces": available_pieces_data,
            "available_connections": self.available_connections  # Additional information about available connections (e.g., piece ID, side index)
        }
        return observation

    def process_action(self, action):
        piece_id, target_id, side_index, target_side_index = action         # Extract the components of the action, which are expected to be the IDs and side indices of the two pieces to connect.
        piece = next((p for p in self.available_pieces if p.id == piece_id), None)    # Attempt to find the piece with the given `piece_id` among the currently available pieces.
        target_piece = self.current_puzzle.nodes.get(target_id, {}).get('piece')         # Attempt to get the target piece from the nodes in the current puzzle graph. If `target_id` is not found, defaults to None.
        # Check if both pieces are found and the specified sides match according to the puzzle's rules.
        if piece and target_piece and self.sides_match(piece, side_index, target_piece, target_side_index):
            self.current_puzzle.add_node(piece.id, piece=piece)  # Add the active piece to the current puzzle graph, signifying its placement in the puzzle.
            self.current_puzzle.add_edge(piece_id, target_id)    # Connect the active piece with the target piece in the graph, effectively linking their respective sides.
            piece.connect_side(side_index)                       # Mark the used sides of both pieces as connected, which updates their availability for future connections.
            target_piece.connect_side(target_side_index)
            self.available_pieces.remove(piece)                  # Remove the active piece from the list of available pieces, as it's now part of the puzzle structure.
            return True                                          # Return True to indicate a valid and successful action that has modified the puzzle's state.
        return False                                             # Return False if the action is invalid (e.g., the pieces cannot be connected, one of the pieces wasn't found, or sides don't match).


    def step(self, action):
        valid_action = self.process_action(action)  # The action is valid if the piece is available, the target piece exists, and the sides match
        if valid_action:
            # Update connections since the puzzle structure has changed
            self.available_connections = self.update_available_connections()
            reward = self.calculate_reward()
            is_done = self.check_completion()
        else:
            reward = -1        # Penalize invalid actions without updating the state
            is_done = False

        obs = self._get_observation()  # Always return the latest state
        return obs, reward, is_done, {}


    # REWARD MECHANISM
    def calculate_reward(self):
        incremental_reward = 1  # Base reward for valid placement
        config_reward = self.overall_configuration_reward()
        completion_reward = self.completion_bonus()  # Calculate completion bonus based on the puzzle state
        return incremental_reward + config_reward + completion_reward

    def completion_bonus(self):
        # Calculate completion bonus based on whether the puzzle is completely solved
        if self.check_completion():
            return 20  # Significant bonus for completing the puzzle
        return 0

    def overall_configuration_reward(self):
        # Reward based on the number of correctly connected edges, scaled to enhance impact
        correct_connections = sum(1 for u, v in self.current_puzzle.edges() if self.target_puzzle.has_edge(u, v))
        total_possible_connections = self.target_puzzle.number_of_edges()
        if total_possible_connections > 0:
            # Scale the reward to make it more significant
            return (correct_connections / total_possible_connections) * 10
        return 0

    def sides_match(self, piece, side_index, target_piece, target_side_index):
        # Validate matching criteria for sides - checks whether two pieces can be legally connected at specified sides.
        # This should be updated to more complexity could mean that the sides have complementary shapes, colors, numbers, or any other criteria that define a correct connection in the puzzle.
        return piece.sides[side_index] == target_piece.sides[target_side_index]


    # VISUALIZATION
    def visualize_puzzle(self):
        plt.figure(figsize=(12, 6))

        # Check if the current puzzle graph is not empty
        if self.current_puzzle.number_of_nodes() > 0:
            plt.subplot(121)
            pos = nx.circular_layout(self.current_puzzle)  # Circular layout for clear visualization
            nx.draw(self.current_puzzle, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_color='darkblue')
            plt.title("Current Puzzle")
        else:
            plt.subplot(121)
            plt.text(0.5, 0.5, 'No pieces placed yet', horizontalalignment='center', verticalalignment='center')
            plt.title("Current Puzzle")

        # Check if the target puzzle graph is not empty
        if self.target_puzzle.number_of_nodes() > 0:
            plt.subplot(122)
            pos = nx.circular_layout(self.target_puzzle)  # Consistent layout with the current puzzle
            nx.draw(self.target_puzzle, pos, with_labels=True, node_color='lightgreen', node_size=500, font_size=10, font_color='darkgreen')
            plt.title("Target Puzzle")
        else:
            plt.subplot(122)
            plt.text(0.5, 0.5, 'Target puzzle is undefined', horizontalalignment='center', verticalalignment='center')
            plt.title("Target Puzzle")

        plt.show()


# GYM ENV #######################

class PuzzleGymEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config=None):
        super(PuzzleGymEnv, self).__init__()
        if config is None:
            config = {'sides': [1, 2, 3, 4]}  # Default configuration
        self.env = PuzzleEnvironment(config)

        # Define the action space and observation space based on the configuration
        num_pieces = len(self.env.pieces)
        num_sides  = len(config['sides'])
        self.action_space = spaces.MultiDiscrete([num_pieces, num_pieces, num_sides, num_sides])
        self.observation_space = spaces.Dict({
            'current_puzzle': spaces.Box(low=0, high=1, shape=(num_pieces, num_pieces), dtype=np.uint8),
            'available_pieces': spaces.Box(low=0, high=1, shape=(num_pieces, num_sides + 1), dtype=np.uint8),
            'available_connections': spaces.Box(low=0, high=1, shape=(len(self.env.update_available_connections()),), dtype=np.uint8)
        })

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self, mode='human'):
        if mode == 'human':
            self.env.visualize_puzzle()

    def close(self):
        pass

####################################################################################################
# EXAMPLE USAGE

if __name__ == "__main__":
    # Initialize the puzzle environment
    config = {'sides': [1, 2, 3, 4]}  # Assuming pieces have four sides numbered for simplicity

    env = PuzzleGymEnv(config)
    obs = env.reset()

    num_steps = 10

    for _ in range(num_steps):
        action = env.action_space.sample()   # action is a tuple of 4 values: (piece_id, target_id, side_index, target_side_index)
        obs, reward, done, _ = env.step(action)  # Apply the action to the environment
        print(f"Action: {action}, Reward: {reward}, Done: {done}")

       # env.render()  # Visualize the state of the environment

        if done:
            print("The puzzle has been solved or the episode is over!")
            break

    env.close()  # Properly close the environment
