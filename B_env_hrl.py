''' Only works on Ray 2.7 or later as they had a bug on the MultiDiscrete '''

import matplotlib.pyplot as plt
import matplotlib.patches as patches  #create shapes
import networkx as nx
import random
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete, Dict, Discrete
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


    #TODO: ver esto
    def _setup_target_puzzle(self):
        '''Defines target puzzle configuration by building a graph where nodes are pieces, and edges represent connections between pieces.
        The target puzzle is a 2x2 grid with 4 pieces, and each piece has 4 sides. The pieces are connected based on their positions in the grid.

        I believe it is doing something like this but I am not sure:

         # Add edges based on the matching side numbers; the order is crucial here
        # Assuming the ids of the pieces are 0, 1, 2, 3 and they are arranged as follows:
        # 0 1
        # 2 3
        # Syntax: Graph.add_edge(node1, node2, attribute_name=value)

        '''
        # Add nodes to the target graph using the pieces with zero-based IDs
        for piece in self.pieces_lst:
            self.target_puzzle.add_node(piece.id, piece=piece.copy())

        # Helper function to determine side index
        def get_side_index(piece_pos, neighbor_pos):
            """Return the side index that connects to the given neighbor."""
            if piece_pos[0] == neighbor_pos[0]:  # Same row
                if piece_pos[1] < neighbor_pos[1]:
                    return 1  # Right side
                else:
                    return 3  # Left side
            else:  # Same column
                if piece_pos[0] < neighbor_pos[0]:
                    return 2  # Bottom side
                else:
                    return 0  # Top side

        # Arrange pieces in a grid and connect them
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                piece_id = i * self.grid_size + j
                if j < self.grid_size - 1:  # Connect to the right piece
                    right_piece_id = piece_id + 1
                    self.target_puzzle.add_edge(
                        piece_id, right_piece_id,
                        side_match=(get_side_index((i, j), (i, j + 1)), get_side_index((i, j + 1), (i, j)))
                    )
                if i < self.grid_size - 1:  # Connect to the bottom piece
                    bottom_piece_id = piece_id + self.grid_size
                    self.target_puzzle.add_edge(
                        piece_id, bottom_piece_id,
                        side_match=(get_side_index((i, j), (i + 1, j)), get_side_index((i + 1, j), (i, j)))
                    )

    def update_pieces_sides(self,current_piece_id = None, target_piece_id = None, side_idx = None, target_side_idx = None):
        """ Used when a piece is placed to mark a piece and connected sides as unavailable (meaning it's placed).
        """
        # Start of the game - No current_pice provided - Reset the available pieces and sides
        if current_piece_id is None:
            for i, piece in enumerate(self.pieces_lst):
                self.available_pieces_sides[i, :-1] = np.array(piece.sides_lst)  # Optionally set specific side values if needed
            self.available_pieces_sides[:, -1] = 1                               # Set the last column to all pieces available (1)

        # Placement of a piece has happened - Update the available pieces and sides
        else:  # current_piece provided - Mark as unavailable (-1) (placed) - last column of the array
            self.available_pieces_sides[current_piece_id, -1] = -1

            # Connection between pieces' sides - Mark the sides as unavailable
            if side_idx and target_side_idx is not None:  # Sides involved in a connection are no longer available
                self.pieces_lst[current_piece_id].connect_side(side_idx)
                self.pieces_lst[target_piece_id].connect_side(target_side_idx)

            # Connections array updated
            if target_piece_id is not None:
                # Since we are connecting 2 pieces, we need to update the available connections for both pieces - meaning that the sides that were connected are no longer available.
                self.update_available_connections_n_sides(current_piece_id, side_idx)         # Update the available connections and sides based on the newly connected piece's side
                self.update_available_connections_n_sides(target_piece_id, target_side_idx)   # Update the available connections and sides for the target piece as well

    def _get_observation(self, agent_id):
        '''Get the current observation of the environment.'''

        #TODO: think about the mask - #TODO: check if the mask is correct
        mask_piece_id, mask_target_side_index = 1, 1 #self._get_action_mask()  # Mask for: Valid active pieces, target pieces, and the target pieces' sides.

        if agent_id == "high_level_agent":  # Selects the target piece and side
            observation = {
                "current_puzzle":         self.current_puzzle,                # Current state of the puzzle grid, -1 means empty
                "available_pieces_sides": self.available_pieces_sides,        # List of available pieces and their sides, -1 means unavailable, 1 means available
                "available_connections":  self.available_connections,         # List of available connections between pieces, -1 means unavailable
                "mask_target_side_index": mask_target_side_index              # Mask for selecting the target piece and the side (Desideratum 4 and 5)
            }
        else:  # Obs for the low_level_agent - Selects the active piece and side
            observation = {
                "current_puzzle":         self.current_puzzle,                # Current state of the puzzle grid, -1 means empty
                "available_pieces_sides": self.available_pieces_sides,        # List of available pieces and their sides, -1 means unavailable, 1 means available
                "available_connections":  self.available_connections,         # List of available connections between pieces, -1 means unavailable
                "mask_piece_id":          mask_piece_id,                      # Mask for selecting the active piece (Desideratum 2) - Only available pieces can be selected as current piece
                "selected_target_piece":  self.target_piece,         # The target piece selected by the high-level agent
                "selected_target_side":   self.target_side           # The side of the target piece selected by the high-level agent
            }
        return observation

    def check_completion(self):
        ''' Completion if no more available pieces'''
        if np.all(self.available_pieces_sides[:, -1] == -1):  # Check if all pieces are unavailable
            return True

        # TODO: ver esto
        '''
        """Check if the current puzzle is a complete match with the target puzzle."""
        current_graph = self.convert_array_to_graph()  # Convert the current puzzle grid to a graph for comparison
        return nx.is_isomorphic(current_graph, self.target_puzzle)  # Compare current graph structure with target graph
        '''

    def convert_array_to_graph(self):
        '''Used to compare against two graphs to check if they are isomorphic. For the reward.
        This method converts the current puzzle array to a graph.'''
        graph = nx.Graph()
        for idx, row in enumerate(self.current_puzzle):
            for jdx, piece_id in enumerate(row):
                if piece_id != -1:
                    graph.add_node(piece_id, pos=(idx, jdx))
                    # Add edges for adjacent pieces if they exist
                    if idx > 0 and self.current_puzzle[idx-1, jdx] != -1:
                        graph.add_edge(piece_id, self.current_puzzle[idx-1, jdx])
                    if jdx > 0 and self.current_puzzle[idx, jdx-1] != -1:
                        graph.add_edge(piece_id, self.current_puzzle[idx, jdx-1])
        return graph

    def process_action(self, action):
        '''Process the action for the Low-level agent. Connect the chosen active piece to the target piece and side.
            Checks if the action is valid  - if the two pieces can be connected based on the puzzle's rules.
            If valid action: method returns True  - and the step method assigns rewards. If invalid action: method returns False.
            If valid action: updates the puzzle state, the side availability and the available connections.
            Update the available connections based on the newly connected piece's side and remove the active piece from the list of available pieces.
            Agent will then receive a reward based on whether the action was valid - meaning that the pieces could be connected and the puzzle state was updated accordingly.
        '''
        #OBS: the action returns the side_idx in (0-4) numbers, but the sides are labeled as 5,6,7,8
        current_piece_id, side_idx = action

        my_print(f"Processing action: Connect selected piece {current_piece_id} at side {side_idx} to target piece {self.target_piece} at side {self.target_side}",self.DEBUG)

        # Check if the selected current piece is still available to be played (i.e., not already placed)
        if self.available_pieces_sides[current_piece_id, -1] == -1:
            my_print("Selected current piece is not available.",self.DEBUG)
            return False

        # Check if the current piece is available and not already placed
        if self.available_pieces_sides[current_piece_id, -1] == 1:
            # Check if the target piece is already placed in the puzzle and has at least one available side to connect
            if self.available_pieces_sides[self.target_piece, -1] == -1 and self.available_pieces_sides[self.target_piece, self.target_side] != -1:
                # Check if the selected sides on the active and target pieces can legally connect
                if self.sides_match(self.pieces_lst[current_piece_id], side_idx, self.pieces_lst[target_piece_id], target_side_idx):
                        # If placement is successful, update the active piece and target_piece, sides and connections as no longer available
                        self.update_current_puzzle(current_piece_id, target_piece_id, side_idx, target_side_idx) # Update self.current_puzzle
                        self.update_pieces_sides(current_piece_id,target_piece_id, side_idx, target_side_idx)
                        my_print(f"Connected piece {current_piece_id} side {side_idx} to piece {target_piece_id} side {target_side_idx}",self.DEBUG)

                        return True                                                             # Return True to indicate a valid and successful action that has modified the puzzle's state.
                else:
                        my_print(f"Sides unmatched for piece {current_piece_id} side {side_idx} and piece {target_piece_id} side {target_side_idx}",self.DEBUG)
            else:
                    my_print(f"Cannot connect piece {current_piece_id} to (renumerated) side {side_idx} with piece's {target_piece_id}  (renumerated) side {target_side_idx}",self.DEBUG)
        else:
                my_print(f"Target piece {target_piece_id} or side {target_side_idx} not available.", self.DEBUG)

        return False                                                                            # Return False if any condition fails and the pieces cannot be connected as intended

    #------------------------------------------------------------------------------------------------
    # Reward mechanism
    #------------------------------------------------------------------------------------------------
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
        # Convert the current puzzle array to a graph
        current_graph = self.convert_array_to_graph()
        # Reward based on the number of correctly connected edges, scaled to enhance impact
        correct_connections = sum(1 for u, v in current_graph.edges() if self.target_puzzle.has_edge(u, v))
        total_possible_connections = self.target_puzzle.number_of_edges()
        if total_possible_connections > 0:
            # Scale the reward to make it more significant
            return (correct_connections / total_possible_connections) * 10
        return 0

    #------------------------------------------------------------------------------------------------
    # High-level and low-level step
    #------------------------------------------------------------------------------------------------
    def _high_level_step(self, action):
        target_piece_n_side = action
        # Decode the combined_target_index to get target_piece_id and target_side_index
        # Sides are renumerated from their original lables to 1, 2, ... TODO: think if this is correct
        self.target_piece = target_piece_n_side // self.num_sides  # Calculate target piece ID
        self.target_side  = target_piece_n_side % self.num_sides   # Calculate side index of the target piece

        obs = {"low_level_agent": self._get_observation("low_level_agent")}         # Obs for the next agent
        rew = {"high_level_agent": 0}                                               # TODO: this might need to be enhanced later
        done = truncated = {"__all__": False}                                       # High level agent never terminates a game
        my_print(f"Target piece {self.target_piece} and side:{self.target_side}, Reward:{rew}", self.DEBUG)
        return obs, rew, done, truncated, {}

    def _low_level_step(self, action):
        '''Low-level agent connects the active piece to the target piece and side.'''
        obs, rew = {}, {}
        valid_action = self.process_action(action)     # Check validity and update connections if valid action
        if valid_action:
           reward = self.calculate_reward()
           terminated = self.check_completion()
        else:
           reward = -1                               # Penalize invalid actions without updating the state
           terminated = False                        # The environment only ever terminates when we reach the goal state.
        obs = {"high_level_agent": self._get_observation("high_level_agent")}         # Obs for the next agent
        rew["high_level_agent"] = 1  #TODO: TRAIN AND TEST to see if I can assign reward to agent without observation
        rew["low_level_agent"] = 5
        done = truncated = self.check_completion()
        return obs, rew, done, truncated, {}


    #------------------------------------------------------------------------------------------------
    # reset
    #------------------------------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        '''High level agent selects a target node and a side to connect to.
           Initializes the puzzle grid and available pieces and sides.
        '''
        if seed is not None: random.seed(seed)   # as per new gymnasium

        # Reset current puzzle mark all pieces and connections as available
        self.current_puzzle.fill(-1)             # Reset the puzzle grid to all -1, indicating empty cells
        self.update_pieces_sides()               # Mark all pieces and sides as available (1)
        self.available_connections.fill(1)       # Mark all connections as available (1)
        self.target_piece = -1
        self.target_side = -1

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

    #------------------------------------------------------------------------------------------------
    # step
    #------------------------------------------------------------------------------------------------
    def step(self, action_dict):

        if "high_level_agent" in action_dict:
            return self._high_level_step(action_dict["high_level_agent"])
        else:
            return self._low_level_step(action_dict["low_level_agent"])


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

        #TODO: revise this! - CHECK: The number of sides has to be the max nbr of sides
        #  High level - select target piece and side: (piece_id, side_index)
        self.high_level_action_space = Discrete(
            self.env.num_pieces * self.env.num_sides)  # combined dimension for target_piece_id and side_target_piece_index

        #  Low level - select active piece and side to connect to (side mask depends on the target piece)
        self.low_level_action_space = MultiDiscrete([
            self.env.num_pieces,                       # active_piece_id
            self.env.num_sides,                        # side_index. The INDEX of the side, not the actual value.
        ])
        # High level - select target piece and side: (piece_id, side_index)
        self.high_level_obs_space = Dict({
                   'current_puzzle'         : Box(low=-1, high=np.inf, shape=(self.env.grid_size, self.env.grid_size)),                        # 2x2 grid for a 4-piece puzzle
                   'available_pieces_sides' : Box(low=-1, high=np.inf, shape=(self.env.num_pieces, self.env.num_sides + 1), dtype=np.int8),    # Availability of pieces and sides. Columns are the side values, and the last column indicates the availability of the piece
                   'available_connections'  : Box(low=-1, high=1, shape=(self.env.num_pieces * self.env.num_sides,), dtype=np.int8),           # Availability of connections
                   'mask_target_side_index' : Box(low=0, high=1, shape=(self.env.num_pieces * self.env.num_sides, ), dtype=np.uint8),          # Mask for selecting the target piece and side in 2D - If the piece is placed and has at least one available side
                })
        #  Low level - select active piece and side given the target piece and side
        self.low_level_obs_space = Dict({
                    'current_puzzle'         : Box(low=-1, high=np.inf, shape=(self.env.grid_size, self.env.grid_size)),                        # 2x2 grid for a 4-piece puzzle
                    'available_pieces_sides' : Box(low=-1, high=np.inf, shape=(self.env.num_pieces, self.env.num_sides + 1), dtype=np.int8),    # Availability of pieces and sides. Columns are the side values, and the last column indicates the availability of the piece
                    'available_connections'  : Box(low=-1, high=1, shape=(self.env.num_pieces * self.env.num_sides,), dtype=np.int8),           # Availability of connections
                    'mask_piece_id'          : Box(low=0, high=1, shape=(self.env.num_pieces,), dtype=np.uint8),                                # Mask for selecting the active piece - Only Available Pieces Can be Selected as Current Piece
                    'target_piece'           : Discrete(self.env.num_pieces),  # target piece selected by the high-level agent
                    'target_side'            : Discrete(self.env.num_sides),   # side of the target piece selected by the high-level agent
                })
        self.action_space = Dict({
                "high_level_agent": self.high_level_action_space,
                "low_level_agent": self.low_level_action_space
            })
        self.observation_space = Dict({
                "high_level_agent": self.high_level_obs_space,
                "low_level_agent": self.low_level_obs_space
            })

    def step(self, action):
        return self.env.step(action)

    def reset(self, seed=None, options=None):
        return self.env.reset()

    def render(self, mode='human'):
        ''' Simple visualization in ASCII'''
        if mode == 'human':
            print("Visualization method not implemented yet")
        #    self.env.visualize_puzzle()

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
    num_sides = len(sides_list[0])

    for _ in range(num_steps):
        action = env.action_space.sample()       # action is to connect available pieces and sides with the target side
        target_piece_n_side  = action['high_level_agent']
        target_piece = target_piece_n_side // num_sides  # Calculate target piece ID
        target_side  = target_piece_n_side % num_sides   # Calculate side index of the target piece
        active_piece, active_side = action['low_level_agent']
        print(f"Action: active_piece: {active_piece}, side:{active_side}, target_piece: {target_piece}, side:{target_side}")

        obs, reward, terminated, truncated, info = env.step(action)  # the observation is the current state of the puzzle and available pieces
        print(f"Reward: {reward}, Done: {terminated}")

        env.render()                                     # Convert current puzzle into string for visualization

        if terminated:
            print("The puzzle has been solved or the episode is over!")
            break

    env.close()  # Properly close the environment
