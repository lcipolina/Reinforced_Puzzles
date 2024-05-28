''' Only works on Ray 2.7 and after as they had a bug on the MultiDiscrete



- TODO: Add visualization

REPRESENTATIONS:

1- Target puzzle is represented as a graph
2- Current_puzzle is represented as a matrix:

[ piece 0, piece 1 ]
[ piece 2, piece 3 ]


3- 'available_pieces' is a list of lists representing the piece sides and availability.
When a piece is placed, its availability status is updated to -1, making it unavailable for further placement.

# matrix representation of each piece's sides and availability
Each row represents a piece, with the last element indicating the availability of the piece.
Each piece has 4 sides, with the last column indicating the availability of the piece.

available_pieces = np.array([
    [0, 1, 2, 3, 1],  # Piece 0: [sides 0, 1, 2, 3], available (1)
    [1, 2, 3, 0, 1],  # Piece 1: [sides 1, 2, 3, 0], available (1)
    [2, 3, 0, 1, 1],  # Piece 2: [sides 2, 3, 0, 1], available (1)
    [-1, -1, -1, -1, -1]  # Piece 3: Not available
], dtype=np.int8)

4- 'available_connections' is a flat array representing the available connections between pieces.
unconnected sides:

Piece 0: [ -1, 2, 3, -1 ]  # North side (1) is already connected, East side (2) is available, South side (3) is available, West side (4) is connected
Piece 1: [ -1, -1, 3, 4 ]  # North side (1) is already connected, East side (2) is connected, South side (3) is available, West side (4) is available
Piece 2: [ 1, 2, -1, -1 ]  # North side (1) is available, East side (2) is available, South side (3) is connected, West side (4) is connected
Piece 3: [ 1, 2, 3, 4 ]    # Piece 3 is still available, so all sides are unconnected and numbered (1 to 4)

'''

import matplotlib.pyplot as plt
import matplotlib.patches as patches  #create shapes
import networkx as nx
import random
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete, Dict


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

    @classmethod # Alternative class constructor to generate a list of puzzle pieces with unique sides
    def _generate_pieces(cls,sides_lst = [1,2,3,4], num_pieces=4):
        '''Generates a list of puzzle pieces with unique sides. Each piece has an ID and a list of sides.
        '''
        # Generate a list of puzzle pieces, assigning an ID to each piece and 'sides_lst[:]' creates a shallow copy of the sides list so each piece has the sides in the same initial order
        pieces = [cls(i, sides_lst[:]) for i in range(num_pieces)]  # IDs now start at 0
        # Generate pieces with sides in different orders to ensure variability
        '''
        # TODO: Shuffling the sides of each piece is disabled at first to make the initial state of the puzzle easier to visualize
        for piece in pieces:
           random.shuffle(piece.sides_lst) # Shuffle the sides for each piece to simulate a real puzzle
        '''
        return pieces  # List of 4 'Piece' objects with their initial states [Piece(id=1, sides_lst=[5,6,7,8], available_sides=[True, True, True, True]),....]

class PuzzleEnvironment:
    def __init__(self, config=None):
        self.sides          = config.get("sides", [5, 6, 7, 8])                                            # Sides are labeled to be different from the keynumbers: "1" for available, etc.
        self.num_sides      = len(self.sides)
        self.pieces_lst     = Piece._generate_pieces(sides_lst =self.sides,num_pieces=len(self.sides)) # Generate puzzle pieces with matching sides

        # Define the puzzle grid dimensions (2x2 for 4 pieces)
        # Current puzzle is an array (then converted to graph for comparisons), target puzzle is a graph
        self.num_pieces     = config.get("num_pieces", 4)
        self.grid_size      = int(np.sqrt(self.num_pieces))                                             # Generates 4 pieces with 4 sides
        self.current_puzzle = np.full((self.grid_size, self.grid_size), -1, dtype=np.int8)              # 2x2 grid for a 4-piece puzzle , "-1"represents an empty cell in the puzzle grid
        self.available_pieces      = np.full((self.num_pieces, self.num_sides + 1), 1, dtype=np.int8)   # Availability of pieces and sides.  "1" represents available - where each row represents a piece, and the last element in each row indicates the availability of the piece.
        self.available_connections = np.full((self.num_pieces * self.num_sides,), 1, dtype=np.int8)     # "1" represents available  of connections as a flat array, where each element corresponds to a specific connection.
        self.available_connections_flat_size = len(self.available_connections)

        self.target_puzzle = nx.Graph()                                                                 # Target configuration as a graph.
        self._setup_target_puzzle()                                                                     # Defines target puzzle as a graph based on pre-defined pieces and connections
        self.reset()

    def convert_array_to_graph(self):
        '''Used to compare against two graphs to check if they are isomorphic. This method converts the current puzzle array to a graph.'''
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

    def check_completion(self):
        """Check if the current puzzle is a complete match with the target puzzle."""
        current_graph = self.convert_array_to_graph()  # Convert the current puzzle grid to a graph for comparison
        return nx.is_isomorphic(current_graph, self.target_puzzle)  # Compare current graph structure with target graph

    def _setup_target_puzzle_old(self):
        '''Defines target puzzle configuration by building a graph where nodes are pieces, and edges represent connections between pieces,
        with matching sides hardcoded based on the puzzle's logic.
        '''
        # Add nodes to the target graph using the pieces with zero-based IDs
        for piece in self.pieces_lst:
            self.target_puzzle.add_node(piece.id, piece=piece.copy())

        # Add edges based on the matching side numbers; the order is crucial here
        # Assuming the ids of the pieces are 0, 1, 2, 3 and they are arranged as follows:
        # 0 1
        # 2 3
        # Syntax: Graph.add_edge(node1, node2, attribute_name=value)

        # Connect top-left (0) and top-right (1) on side '1'
        self.target_puzzle.add_edge(0, 1, side_match=(self.pieces_lst[0].sides_lst.index(1), self.pieces_lst[1].sides_lst.index(1)))
        # Connect top-right (1) and bottom-right (3) on side '2'
        self.target_puzzle.add_edge(1, 3, side_match=(self.pieces_lst[1].sides_lst.index(2), self.pieces_lst[3].sides_lst.index(2)))
        # Connect bottom-right (3) and bottom-left (2) on side '3'
        self.target_puzzle.add_edge(3, 2, side_match=(self.pieces_lst[3].sides_lst.index(3), self.pieces_lst[2].sides_lst.index(3)))
        # Connect bottom-left (2) and top-left (0) on side '4'
        self.target_puzzle.add_edge(2, 0, side_match=(self.pieces_lst[2].sides_lst.index(4), self.pieces_lst[0].sides_lst.index(4)))


    def _setup_target_puzzle(self):
        '''Defines target puzzle configuration by building a graph where nodes are pieces, and edges represent connections between pieces.'''
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


    def _get_action_mask(self):
        """Returns a mask with 1 for valid actions and 0 for invalid ones.

        The mask is a 4D array with dimensions: (active_piece_id, target_piece_id, active_side_index, target_side_index).
        In Multidiscrete, the action space is a tuple of 4 values: (piece_id, target_id, side_index, target_side_index)
        The total number of actions the policy can take is 4*4*4*4 = 256, which is the size of the action space.
        The policy outputs parameters for the categorical distribution of each dimension separately
        (there is no joint distribution over all dimensions- only the shared representation on the feed forward).
        The mask should be applied to each of these 256 actions to determine which are valid at any given state.

        Each element (i, j, k, l) represents the validity of connecting piece i's side k to piece j's side l.
        active_piece_id: is masked if the piece is not available (i.e. already placed in the puzzle)
        target_piece_id: is masked if the piece is not available (i.e. all sides are already connected) - or same as active piece
        active_side_index: is not masked as sides of the active piece are always available
        target_side_index: is masked if the side is not available

        The iterations are nested because for each active piece and its side,
          we need to check the validity of connecting it to every possible target piece and its sides.
        This ensures that every combination of actions is considered and correctly marked as valid or invalid.
        """
        num_pieces = len(self.pieces_lst)
        num_sides  = len(self.sides)

        # Initialize a mask to all zeros - for invalid actions
        action_mask = np.zeros((num_pieces, num_pieces, num_sides, num_sides), dtype=np.uint8) #(active_piece, target_piece, active_piece_side, target_piece_side)

        # Extract availability information from available_pieces
        piece_availability = self.available_pieces[:, -1] # 1D array indicating whether each piece is available (1) or not (-1).
        side_availability = self.available_pieces[:, :-1] # 2D array indicating whether each side of each piece is available (1) or not (-1).

        # Determine valid active pieces and sides
        valid_active_pieces = [piece_idx for piece_idx in range(num_pieces) if piece_availability[piece_idx] == 1]

        # Determine valid target pieces and sides - Loop through each possible action combination
        # Loop through each valid active piece (avoid placing an active piece)
        for active_piece_idx in valid_active_pieces:
            # Loop through each possible target piece
            for target_piece_idx in range(num_pieces):
                # Ensure the target piece is not the same as the active piece
                if active_piece_idx != target_piece_idx:                                       # Check 1: active_piece_idx must not equal target_piece_idx
                    # Check if the target piece is already placed on the board
                    if piece_availability[target_piece_idx] == -1:                             # Check 2: target_piece must be placed (piece_availability == -1)
                        # Check if the target piece has at least one available side
                        if any(self.pieces_lst[target_piece_idx].available_sides):             # Check 3: target_piece must have at least one available side
                            # Loop through each side of the target piece
                            for target_side_idx in range(num_sides):
                                # Check if the current side of the target piece is available
                                if side_availability[target_piece_idx, target_side_idx] == 1:  # Check 4: target_side_idx must be available
                                    # Mark all actions involving the *active* piece's sides as valid
                                    for active_side_idx in range(num_sides):                   # Since the active piece is not yet placed, all its sides are available for connection.
                                        # Mark this action as valid in the action mask
                                        action_mask[active_piece_idx, target_piece_idx, active_side_idx, target_side_idx] = 1  # Mark as valid

        # Flatten the mask to a 1D array for use in the RLLIB Net
        self.current_mask = action_mask.flatten()

        return self.current_mask

    def mark_piece_unavailable(self, piece_index):
        """Marks a piece as unavailable but keeps its sides available."""
        # Update the corresponding row in available_pieces to reflect it's not available
        self.available_pieces[piece_index, -1] = -1 # "-1" means unavaiable

    def update_available_connections(self, piece_id, side_index):
        '''Updates the list of available connections based on the current state of the puzzle.'''
        # Assume connections are indexed by piece_id and side_index
        connection_index = (piece_id * self.num_sides) + side_index # Calculate the index of the connection in the flattened array
        self.available_connections[connection_index] = -1           # Mark as unavailable

    def flatten_available_connections(self,connections_dict):
        """Flattens a dictionary of lists into a 1D array."""
        return np.concatenate([np.array(v) for v in connections_dict.values()])


    def reset(self, seed=None, options=None):
        ''' Puzzle starts with a single node and no connections.'''
        if seed is not None: random.seed(seed) # as per new gymnasium

        # Reset current puzzle mark all pieces and connections as available
        self.current_puzzle.fill(-1)             # Reset the puzzle grid to all -1, indicating empty cells
        self.available_pieces.fill(1)            # Mark all pieces and sides as available (1)
        self.available_connections.fill(1)       # Mark all connections as available (1)

        # Create a list of piece IDs from the already existing pieces_lst
        piece_ids = [piece.id for piece in self.pieces_lst]
        # TODO: np.random.shuffle(piece_ids)

        # Need to start with a placed piece, otherwise there is no available target piece and the action mask fails
        start_piece_id = piece_ids[0]                         # Start with the first piece in the list
        self.current_puzzle[0, 0] = start_piece_id            # Place the first piece in the top-left corner
        self.mark_piece_unavailable(start_piece_id)           # Mark the starting piece as unavailable

        print(f"Starting piece: {start_piece_id} placed in puzzle grid")

        # TODO: go throught the action mask and review it thoroughly

        return self._get_observation(), {}

    def _get_observation(self):
        '''Get the current observation of the environment.
        '''
        self.current_mask = self._get_action_mask()                     # Calculate/update the current action mask to determine which actions are valid at this state
        observation = {
                "current_puzzle"       : self.current_puzzle,           # Current state of the puzzle grid, -1 means empty
                "available_pieces"     : self.available_pieces,         # List of available pieces and their sides, -1 means unavailable, 1 means available
                "available_connections": self.available_connections,    # List of available connections between pieces, -1 means unavailable
                "action_mask"          : self.current_mask
            }
        return observation


    def sides_match(self, piece, side_index, target_piece, target_side_index):
        # Validate matching criteria for sides - checks whether two pieces can be legally connected at specified sides.
        # This should be updated to more complexity could mean that the sides have complementary shapes, colors, numbers, or any other criteria that define a correct connection in the puzzle.
        return piece.sides_lst[side_index] == target_piece.sides_lst[target_side_index]

    def process_action(self, action):
        '''Process the action to connect two pieces if the rules permit.
            First checks if the action is valid  - if the two pieces can be connected based on the puzzle's rules.
            If valid action: method returns True  - and next method assigns rewards. If invalid action: method returns False.
            If valid action: updates the puzzle state and available connections.
            Update the available connections based on the newly connected piece's side and remove the active piece from the list of available pieces.
            Agent will then receive a reward based on whether the action was valid
            meaning that the pieces could be connected and the puzzle state was updated accordingly.
        '''

        current_piece_id, target_id, side_index, target_side_index = action

        # Find the active piece among the available pieces
        piece_idx_lst = np.where(self.available_pieces[:, -1] == 1)[0]  # Lst of idx of available pieces
        piece_row     = next((idx for idx in piece_idx_lst if self.available_pieces[idx, 0] == current_piece_id), None) # Bring the first piece that matches with the given `current_piece_id` among the currently available pieces.

        # Retrieve the target piece from the puzzle grid (assuming it is already placed)
        target_position = np.where(self.current_puzzle == target_id)               # Get the target piece from the nodes in the current puzzle graph.
        if target_position[0].size == 0:
            print(f"ISSUE: Selected target piece {target_id} not found in the current puzzle")
            return False  # Target not found

        # Check if the specified sides match according to the puzzle's rules.  # If they do, add the active piece to the current puzzle graph as a new node and connect it with the target piece.
        if piece_row is not None and self.sides_match(
            piece=self.pieces_lst[current_piece_id],
            side_index=side_index,
            target_piece=self.pieces_lst[target_id],
            target_side_index=target_side_index
        ):
            print(f"Piece {current_piece_id} and target piece {target_id} can be connected")

            # 1 - Update current_puzzle: Place the active piece on the grid at the target position
            self.current_puzzle[target_position] = current_piece_id  # Place the active piece on the grid and update the graph state

            # 2 - Update available_pieces: Mark the piece as unavailable ("-1")
            self.available_pieces[piece_row, -1] = -1       # Remove the active piece from the list of available pieces, as it's now part of the puzzle structure.

            # 3 - Mark the sides as connected using connect_side
            self.pieces_lst[current_piece_id].connect_side(side_index)
            self.pieces_lst[target_id].connect_side(target_side_index)

            # 3 - Update available_connections: Mark the specific connections (sides) involved in the action as unavailable
            # Since we are connecting 2 pieces, we need to update the available connections for both pieces - meaning that the sides that were connected are no longer available.
            self.update_available_connections(current_piece_id, side_index)         # Update the available connections based on the newly connected piece's side
            self.update_available_connections(target_id, target_side_index)         # Update the available connections for the target piece as well

            return True                                                             # Return True to indicate a valid and successful action that has modified the puzzle's state.

        print(f"ISSUE: No matching sides - current piece side: {side_index} and target piece {target_side_index}")
        return False                                                                # Return False if the action is invalid (e.g., the pieces cannot be connected, one of the pieces wasn't found, or sides don't match).

    def step(self, action):
        valid_action = self.process_action(action)     # Check validity and update connections if valid action
        if valid_action:
            reward = self.calculate_reward()
            terminated = self.check_completion()
        else:
            reward = -1                               # Penalize invalid actions without updating the state
            terminated = False                        # The environment only ever terminates when we reach the goal state.

        obs = self._get_observation()
        truncated = False  # Whether the episode was truncated due to the time limit

        return obs, reward, terminated, truncated, {}


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
        # Convert the current puzzle array to a graph
        current_graph = self.convert_array_to_graph()
        # Reward based on the number of correctly connected edges, scaled to enhance impact
        correct_connections = sum(1 for u, v in current_graph.edges() if self.target_puzzle.has_edge(u, v))
        total_possible_connections = self.target_puzzle.number_of_edges()
        if total_possible_connections > 0:
            # Scale the reward to make it more significant
            return (correct_connections / total_possible_connections) * 10
        return 0


    # ------------------------------------------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------------------------------------------
    # TODO: rotate the pieces to align the sides correctly
    # TODO: add graphing


# ========================================================================================================
# GYM ENV
# ========================================================================================================
class PuzzleGymEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config=None):
        super(PuzzleGymEnv, self).__init__()

        if config is None:  # Default configuration
            config = {'sides': [5, 6, 7, 8],    # Sides are labeled to be different from the keynumbers: "1" for available, etc.
                      'num_pieces': 4}
        self.env = PuzzleEnvironment(config)

        # Define the action space and observation space based on the configuration
        num_pieces = len(self.env.pieces_lst)
        num_sides  = len(config['sides'])
        #  The action is defined as a tuple of 4 values: (piece_id, target_id, side_index, target_side_index)
        self.action_space = MultiDiscrete([num_pieces, num_pieces, num_sides, num_sides])

        # Include a masking on the policy to dynamically indicate which actions are valid at any given state of the environment.

        available_connections_flat_size = num_pieces * num_sides # Calculate the flattened size of available_connections
        self.observation_space =Dict({
            'current_puzzle': Box(low=0, high=1, shape=(num_pieces, num_pieces), dtype=np.uint8),
            'available_pieces': Box(low=0, high=1, shape=(num_pieces, num_sides + 1), dtype=np.uint8),
            'available_connections': Box(low=0, high=1, shape=(available_connections_flat_size,), dtype=np.uint8),
            'action_mask': Box(low=0, high=1, shape=(num_pieces * num_pieces * num_sides * num_sides,), dtype=np.float32)  # Switch to float32
        })
        '''NOT SURE IF STILL NEEDED
        # Method to sample a single observation
        def observation_space_sample(self):
            return self.observation_space.sample()

        # Method to sample a single action
        def action_space_sample(self):
            return self.action_space.sample()

        # Check if an action is valid in the environment's action space
        def action_space_contains(self, action):
            return self.action_space.contains(action)

        # Check if an observation is valid in the environment's observation space
        def observation_space_contains(self, observation):
            return self.observation_space.contains(observation)
        '''

    def step(self, action):
        return self.env.step(action)

    def reset(self, seed=None, options=None):
        return self.env.reset()

    '''TODO: fix the visualization'
    def render(self, mode='human'):
        if mode == 'human':
            self.env.visualize_puzzle()
    '''

    def close(self):
        pass

####################################################################################################
# EXAMPLE USAGE

if __name__ == "__main__":
    # Initialize the puzzle environment
    config = {
        'sides': [5, 6, 7, 8],  # Sides are labeled to be different from the keynumbers: "1" for available, etc.
        'num_pieces': 4,
        }

    env = PuzzleGymEnv(config) # Initialize the env (including reset)

    num_steps = 10

    for _ in range(num_steps):
        action = env.action_space.sample()       # action is to connect available pieces and sides with the target side
        piece_id, target_id, side_index, target_side_idx = action
        print(f"Action: (active_piece: {piece_id}, target_piece: {target_id}, active_side: {side_index}, target_side: {target_side_idx})")
        obs, reward, terminated, truncated, info = env.step(action)  # the observation is the current state of the puzzle and available pieces
        print(f"Reward: {reward}, Done: {terminated}")

       # env.render()  # Visualize the state of the environment #TODO: fix the visualization

        if terminated:
            print("The puzzle has been solved or the episode is over!")
            break

    env.close()  # Properly close the environment
