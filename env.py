
import matplotlib.pyplot as plt
import matplotlib.patches as patches  #create shapes
import networkx as nx
import random
import numpy as np
import gymnasium as gym
from gym import spaces

class Piece:
    '''represents individual puzzle pieces. Each piece has an ID, a list of sides (each side having some value),
    and a list to track which sides are still available for connection.'''

    def __init__(self, id, sides):
        self.id = id
        self.sides_lst = sides  # List of side values
        self.available_sides = [True] * len(sides)  # All sides are initially available for connection

    def rotate(self, degrees):
         # Allows the piece to be rotated in 90-degree increments.  This method updates the sides and available_sides
        if degrees % 90 != 0:
            return
        num_rotations = degrees // 90
        self.sides_lst = self.sides_lst[-num_rotations:] + self.sides_lst[:-num_rotations]
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
        new_piece = Piece(self.id, self.sides_lst.copy())
        new_piece.available_sides = self.available_sides.copy()  # Ensure the availability status is also copied
        return new_piece

    @classmethod
    def _generate_pieces(cls, num_sides=4, num_pieces=4):
        # Each side should have a unique number on each piece
        sides = [1, 2, 3, 4]
        # Generate pieces with sides in different orders to ensure variability
        pieces = [cls(i, sides[:]) for i in range(1, num_pieces + 1)]
        # Shuffle the sides for each piece to simulate a real puzzle - TODO: at the beginning, do not shuffle the sides (easier to visualize the puzzle)
        # for piece in pieces:
        #    random.shuffle(piece.sides_lst)

        return pieces

class PuzzleEnvironment:
    def __init__(self, config=None):
        self.sides = config.get("sides", [1, 2, 3, 4])  # Default sides if not provided
        self.pieces_lst = Piece._generate_pieces(num_sides=len(self.sides),num_pieces=len(self.sides)) # Generate puzzle pieces with matching sides
        self.target_puzzle = nx.Graph()  # Target configuration as a graph.
        self.current_puzzle = nx.Graph()  # Current state of the puzzle.
        self.available_pieces = self.pieces_lst.copy() # Initially, all pieces are available to be placed in the puzzle
        self._setup_target_puzzle()
        self.reset()

    def check_completion(self):
         # Check if the current puzzle is a complete match with the target puzzle
        return nx.is_isomorphic(self.current_puzzle, self.target_puzzle)

    def _setup_target_puzzle(self):
            # This method will now correctly match sides based on the layout you provided.
            # The connections should be hardcoded based on the puzzle's logic.

            # Add nodes
            for piece in self.pieces_lst:
                self.target_puzzle.add_node(piece.id, piece=piece.copy())

            # Add edges based on the matching side numbers, the order is important here
            # Assuming the ids of the pieces are 1, 2, 3, 4 and they are arranged as follows:
            # 1 2
            # 3 4

            # Graph.add_edge(node1, node2, attribute_name=value)
            # Connect top left (1) and top right (2) on side '1'
            self.target_puzzle.add_edge(1, 2, side_match=(self.pieces_lst[0].sides_lst.index(1), self.pieces_lst[1].sides_lst.index(1)))
            # Connect top right (2) and bottom right (4) on side '2'
            self.target_puzzle.add_edge(2, 4, side_match=(self.pieces_lst[1].sides_lst.index(2), self.pieces_lst[3].sides_lst.index(2)))
            # Connect bottom right (4) and bottom left (3) on side '3'
            self.target_puzzle.add_edge(4, 3, side_match=(self.pieces_lst[3].sides_lst.index(3), self.pieces_lst[2].sides_lst.index(3)))
            # Connect bottom left (3) and top left (1) on side '4'
            self.target_puzzle.add_edge(3, 1, side_match=(self.pieces_lst[2].sides_lst.index(4), self.pieces_lst[0].sides_lst.index(4)))


    def update_available_connections(self, piece_id, side_index):
        # This method updates the list of available connections based on the current state of the puzzle.
        if piece_id in self.available_connections:
            self.available_connections[piece_id][side_index] = 0  # Mark side as no longer available

    def initialize_available_connections(self):
        # Initializes available connections for all pieces with all sides available
        connections = {piece.id: [1] * len(self.sides) for piece in self.available_pieces}
        return connections

    def reset(self):
        # Puzzle starts with a single node and no connections.
        self.current_puzzle.clear() # Clear the current puzzle state
        start_piece = random.choice(self.pieces_lst)  # Select a random piece to start with or specify the starting piece
        self.current_puzzle.add_node(start_piece.id, piece=start_piece.copy())
        self.available_pieces = [piece.copy() for piece in self.pieces_lst if piece.id != start_piece.id] # Reset available pieces to all pieces except the starting piece
        self.available_connections = self.initialize_available_connections()          # Initialize available connections based on the starting piece's available sides
        return self._get_observation()


    def _get_observation(self):
        # Returns the current state of the environment as an observation. This includes the current puzzle graph, available pieces, and available connections.
        graph_data = nx.to_numpy_array(self.current_puzzle, dtype=np.uint8) # Converts current puzzle graph as an adjacency matrix (for gym env)
        available_pieces_data = np.array([[piece.id] + piece.sides_lst for piece in self.available_pieces],dtype=np.uint8 )  # Available pieces and their details (e.g., sides, orientation)
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
            self.update_available_connections(piece_id, side_index)
            self.update_available_connections(target_id, target_side_index)
            self.available_pieces.remove(piece)                  # Remove the active piece from the list of available pieces, as it's now part of the puzzle structure.
            return True                                          # Return True to indicate a valid and successful action that has modified the puzzle's state.
        return False                                             # Return False if the action is invalid (e.g., the pieces cannot be connected, one of the pieces wasn't found, or sides don't match).


    def step(self, action):
        valid_action = self.process_action(action)  # Check validity and update connections since the puzzle structure has changed
        if valid_action:
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
        return piece.sides_lst[side_index] == target_piece.sides_lst[target_side_index]


   # VISUALIZATION

    #TODO: rotate the pieces to align the sides correctly
    def visualize_puzzle(self):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Create a figure with two subplots side by side, one for the current puzzle and one for the target puzzle
        axs[0].set_title('Current Puzzle')    # Set titles for the subplots for easy identification
        axs[1].set_title('Target Puzzle')
        # Define the positions for each puzzle piece, where the keys are the piece IDs and the values are the coordinates
        positions = {1: (0, 0), 2: (1, 0), 3: (0, -1), 4: (1, -1)}
        label_offsets = {0: (0, 0.05), 1: (0.05, 0), 2: (0, -0.05), 3: (-0.05, 0)}  # Define offsets for drawing the side labels of each puzzle piece, relative to the center of the piece
        self.draw_puzzle(self.current_puzzle, axs[0], positions, label_offsets)     # Draw the current puzzle state on the first subplot
        self.draw_puzzle(self.target_puzzle, axs[1], positions, label_offsets)      # Draw the target puzzle configuration on the second subplot
        plt.show()                                                                  # Display the figure with the two subplots

    def draw_puzzle(self, puzzle, ax, positions, label_offsets):
        half_side = 0.1  # Set a variable for half the length of a puzzle piece's side to assist with centering the squares

        # Loop through each piece in the puzzle
        for node_id, attrs in puzzle.nodes(data=True):
            piece = attrs['piece']           # Extract the piece object, which contains information about the piece
            x, y = positions[node_id]         # Retrieve the (x, y) position for the piece based on its ID
            # Create a rectangle to represent the puzzle piece, centered at (x, y) with width and height of 0.2 units
            square = patches.Rectangle((x - half_side, y - half_side), 0.2, 0.2, linewidth=1, edgecolor='black', facecolor='lightblue')
            ax.add_patch(square)  # Add the square to the subplot
            ax.text(x, y, str(node_id), ha='center', va='center', fontsize=9, color='red') # Place the piece's ID at the center of the square
            for side_idx, side_val in enumerate(piece.sides_lst): # Label each side of the piece with its respective value from the sides list
                ox, oy = label_offsets[side_idx] # Calculate the offset for each side label based on predefined offsets
                ax.text(x + ox, y + oy, f'{side_val}', ha='center', va='center', color='blue', fontsize=8)  # Place the label for each side next to the square

        # Draw lines between connected pieces to represent the edges of the puzzle
        for u, v, data in puzzle.edges(data=True):
            # Check if the edge has a 'side_match' attribute, which indicates which sides of the pieces are connected
            if 'side_match' in data:
                side_u, side_v = data['side_match']  # Extract the side indices from the 'side_match' attribute
                u_pos = positions[u] # Get the positions for the connected pieces
                v_pos = positions[v]
                u_offset = label_offsets[side_u] # Calculate the starting and ending points for the line based on side offsets
                v_offset = label_offsets[side_v]
                start_point = (u_pos[0] + u_offset[0], u_pos[1] + u_offset[1]) # Determine the exact points for the line based on positions and offsets
                end_point = (v_pos[0] + v_offset[0], v_pos[1] + v_offset[1])
                ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'green', linewidth=2)  # Draw the line connecting the sides of the pieces



# GYM ENV #######################
class PuzzleGymEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config=None):
        super(PuzzleGymEnv, self).__init__()
        if config is None:
            config = {'sides': [1, 2, 3, 4]}  # Default configuration
        self.env = PuzzleEnvironment(config)

        # Define the action space and observation space based on the configuration
        num_pieces = len(self.env.pieces_lst)
        num_sides  = len(config['sides'])
        num_connections  = num_pieces * num_sides #TODO: delete this
        self.action_space = spaces.MultiDiscrete([num_pieces, num_pieces, num_sides, num_sides])
        self.observation_space = spaces.Dict({
            'current_puzzle': spaces.Box(low=0, high=1, shape=(num_pieces, num_pieces), dtype=np.uint8),
            'available_pieces': spaces.Box(low=0, high=1, shape=(num_pieces, num_sides + 1), dtype=np.uint8),
            'available_connections': spaces.Dict({
                str(i): spaces.MultiBinary(num_sides) for i in range(1, num_pieces + 1)
            })
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

        env.render()  # Visualize the state of the environment

        if done:
            print("The puzzle has been solved or the episode is over!")
            break

    env.close()  # Properly close the environment
