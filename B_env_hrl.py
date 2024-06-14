''' Only works on Ray 2.7 or later as they had a bug on the MultiDiscrete '''




import matplotlib.pyplot as plt
import matplotlib.patches as patches  #create shapes
import networkx as nx
import random
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete, Dict
from ray.rllib.env import MultiAgentEnv


from Z_utils import my_print


class Piece:
    '''represents individual puzzle pieces. Each piece has an ID, a list of sides (each side having some value),
    and a list to track which sides are still available for connection.'''

    def __init__(self, id, sides):
        self.id = id
        self.sides_lst = sides  # List of side values
        self.available_sides = [True] * len(sides)  # All sides are initially available for connection

    def rotate(self, degrees):
        num_sides = len(self.sides_lst)
        if degrees % (360 // num_sides) != 0:
            return
        num_rotations = degrees // (360 // num_sides)
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

    @classmethod # Alternative class constructor to generate a list of puzzle pieces with unique sides
    def _generate_pieces(cls, sides_lst):
        '''Generates a list of puzzle pieces with unique sides. Each piece has an ID and a list of sides.
           Arguments:
           'cls' is used to reference the class itself (and not instances of the class), allowing us to create new instances of the class.
           'sides_lst' is a list of lists, where each inner list represents the sides of a piece.
        '''
        pieces = []
        for i, sides in enumerate(sides_lst):
            pieces.append(cls(i , sides)) # Create a new piece with a unique ID and sides
        return pieces


class PuzzleEnvironment:
    ''' A puzzle environment where agents must connect pieces together to form a complete puzzle.'''
    def __init__(self, config=None):
        self.sides          = config.get("sides", [[5, 6, 7, 8]])                                          # List of Lists -  Sides are labeled to be different from the keynumbers: "1" for available, etc.
        self.num_pieces     = config.get("num_pieces", 4)                                                  # Number of pieces in the puzzle
        self.num_sides      = len(self.sides)
        self.DEBUG          = config.get("DEBUG", False)                                                   # Whether to print or not
        self.grid_size      = config.get("grid_size", 10)                                                  # height and width # int(np.sqrt(self.num_pieces))  # Generates 4 pieces with 4 sides
        self.pieces_lst     = Piece._generate_pieces(sides_lst =self.sides)     # Generate pieces, sides and availability

        # Define the puzzle grid dimensions (Ex: 2x2 for 4 pieces)
        # Current puzzle is an array (then converted to graph for comparisons), target puzzle is a graph
        self.current_puzzle = np.full((self.grid_size, self.grid_size), -1, dtype=np.int8)              # (grid_size)x(grid_size) grid for a N-piece puzzle , "-1"represents an empty cell in the puzzle grid
        self.available_pieces_sides = np.full((self.num_pieces, self.num_sides + 1), 1, dtype=np.int8)  # Availability of pieces and sides.  "1" represents available - where each row represents a piece, and the last element in each row indicates the availability of the piece.
        self.available_connections  = np.full((self.num_pieces * self.num_sides,), 1, dtype=np.int8)    # "1" represents available  of connections as a flat array, where each element corresponds to a specific connection.

        self.target_puzzle = nx.Graph()                                                                 # Target configuration as a graph.
        self._setup_target_puzzle()                                                                     # Defines target puzzle as a graph based on pre-defined pieces and connections
        self.reset()


    def _get_observation(self, agent_id):
        '''Get the current observation of the environment.'''

        #TODO: think about the mask
        mask_piece_id, mask_target_side_index = self._get_action_mask()  # Mask for: Valid active pieces, target pieces, and the target pieces' sides.

        if agent_id == "high_level_agent":
            observation = {
                "current_puzzle":         self.current_puzzle,                # Current state of the puzzle grid, -1 means empty
                "available_connections":  self.available_connections,         # List of available connections between pieces, -1 means unavailable
                "mask_target_side_index": mask_target_side_index              # Mask for selecting the target piece and the side (Desideratum 4 and 5)
            }
        else:  # Obs for the low_level_agent
            observation = {
                "current_puzzle":         self.current_puzzle,                # Current state of the puzzle grid, -1 means empty
                "available_pieces_sides": self.available_pieces_sides,        # List of available pieces and their sides, -1 means unavailable, 1 means available
                "available_connections":  self.available_connections,         # List of available connections between pieces, -1 means unavailable
                "mask_piece_id":          mask_piece_id,                      # Mask for selecting the active piece (Desideratum 2) - Only available pieces can be selected as current piece
                "selected_target_piece":  self.selected_target_piece,         # The target piece selected by the high-level agent
                "selected_target_side":   self.selected_target_side           # The side of the target piece selected by the high-level agent
            }
        return observation


    def _high_level_step(self, action):

        self.selected_target_piece, self.selected_target_side = action

        obs = {"low_level_agent": self._get_observation("low_level_agent")}         # Obs for the next agent
        rew = {"high_level_agent": 0}                                               #TODO: this might need to be enhanced later
        done = truncated = {"__all__": False}        #TODO: check if this is OK

        my_print(f"Target piece {self.selected_target_piece} and side:{self.selected_target_side}, Reward:{rew}", self.DEBUG)

        return obs, rew, done, truncated, {}


    def _low_level_step(self, action):

        self.selected_active_piece, self.selected_active_side = action






    def reset(self, seed=None, options=None):
        '''High level agent selects a target node and a side to connect to.
           Initializes the puzzle grid and available pieces and sides.
        '''
        if seed is not None: random.seed(seed)   # as per new gymnasium

        # Reset current puzzle mark all pieces and connections as available
        self.current_puzzle.fill(-1)             # Reset the puzzle grid to all -1, indicating empty cells
        self.update_pieces_sides()               # Mark all pieces and sides as available (1)
        self.available_connections.fill(1)       # Mark all connections as available (1)
        self.selected_target_piece = -1
        self.selected_target_side = -1

        # Create a list of piece IDs from the already existing pieces_lst
        piece_ids = [piece.id for piece in self.pieces_lst]
        # TODO: np.random.shuffle(piece_ids)

        # Need to start with a placed piece, otherwise there is no available target piece and the action mask fails
        start_piece_id = piece_ids[0]                                 # Start with the first piece in the list
        middle_position = (self.grid_size // 2, self.grid_size // 2)  # Place the first piece in the middle of the grid
        self.current_puzzle[middle_position] = start_piece_id
        self.update_pieces_sides(start_piece_id)                      # Mark the starting piece as unavailable (it's already placed)

        my_print(f"Starting piece: {start_piece_id} placed in puzzle grid at {middle_position}", self.DEBUG)

        # First playing agent is the hihg level agent


        return {"high_level_agent": self._get_observation("high_level_agent")} , {}


    def step(self, action_dict):


        if "high_level_agent" in action_dict:
            return self._high_level_step(action_dict["high_level_agent"])
        else:
            return self._low_level_step(action_dict["low_level_agent"])  #list(action_dict.values())[0]




# ========================================================================================================
# GYM ENV
# ========================================================================================================
class PuzzleGymEnv(MultiAgentEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, config=None):
        super(PuzzleGymEnv, self).__init__()

        if config is None:  # Default configuration
            config = {'sides': [[5, 6, 7, 8]],    # Sides are labeled to be different from the keynumbers: "1" for available, etc.
                      'num_pieces': 4}

        self.env = PuzzleEnvironment(config)

        #  The action is defined as a tuple of 3 values: (piece_id, target_id, side_index * target_side_index)
        self.action_space = MultiDiscrete([
            self.env.num_pieces,                       # active_piece_id
            self.env.num_sides,                        # active_piece_side_index. The INDEX of the side, not the actual value.
            self.env.num_pieces * self.env.num_sides   # combined dimension for target_piece_id and side_target_piece_index
        ])

        # Include a masking on the policy to dynamically indicate which actions are valid at any given state of the environment.
        self.observation_space = Dict({
                    'current_puzzle'         : Box(low=-1, high=np.inf, shape=(self.env.grid_size, self.env.grid_size)),         # 2x2 grid for a 4-piece puzzle
                    'available_pieces_sides' : Box(low=-1, high=np.inf, shape=(self.env.num_pieces, self.env.num_sides + 1), dtype=np.int8),    # Availability of pieces and sides. Columns are the side values, and the last column indicates the availability of the piece
                    'available_connections'  : Box(low=-1, high=1, shape=(self.env.num_pieces * self.env.num_sides,), dtype=np.int8),           # Availability of connections
                    'mask_piece_id'          : Box(low=0, high=1, shape=(self.env.num_pieces,), dtype=np.uint8),                                # Mask for selecting the active piece - Only Available Pieces Can be Selected as Current Piece
                   'mask_target_side_index'  : Box(low=0, high=1, shape=(self.env.num_pieces * self.env.num_sides, ), dtype=np.uint8),          # Mask for selecting the target piece and side in 2D - If the piece is placed and has at least one available side
                })


    def step(self, action):
        return self.env.step(action)

    def reset(self, seed=None, options=None):
        return self.env.reset()

    def render(self, mode='human'):
        ''' Simple visualization in ASCII'''
        if mode == 'human':
            self.env.visualize_puzzle()

    def close(self):
        pass

####################################################################################################
# EXAMPLE USAGE

if __name__ == "__main__":
    # Initialize the puzzle environment
    sides_list = [
        [5, 6, 7, 8],
        [7, 8, 5, 6],
        [5, 6, 7, 8],
        [7, 8, 5, 6]
        ]
    config = {
        'sides': sides_list         ,  # Sides are labeled to be different from the keynumbers: "1" for available, etc.
        'num_pieces': len(sides_list),
        'grid_size': 10,
        'DEBUG': True
        }

    env = PuzzleGymEnv(config) # Initialize the env (including reset)

    num_steps = 10

    for _ in range(num_steps):
        action = env.action_space.sample()       # action is to connect available pieces and sides with the target side
        piece_id, target_id, target_side_idx = action
        print(f"Action: (active_piece: {piece_id}, target_piece: {target_id}, target_side: {target_side_idx})")
        obs, reward, terminated, truncated, info = env.step(action)  # the observation is the current state of the puzzle and available pieces
        print(f"Reward: {reward}, Done: {terminated}")

        env.render()                                     # Convert current puzzle into string for visualization

        if terminated:
            print("The puzzle has been solved or the episode is over!")
            break

    env.close()  # Properly close the environment
