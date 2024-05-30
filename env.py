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
        #self.available_connections_flat_size = len(self.available_connections)

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
        ''' Completion if no more available pieces'''
        if np.all(self.available_pieces[:, -1] == -1):  # Check if all pieces are unavailable
            return True

        # TODO: ver esto
        '''
        """Check if the current puzzle is a complete match with the target puzzle."""
        current_graph = self.convert_array_to_graph()  # Convert the current puzzle grid to a graph for comparison
        return nx.is_isomorphic(current_graph, self.target_puzzle)  # Compare current graph structure with target graph
        '''
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

    #TODO: ver esto
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


    def mark_piece_unavailable(self, piece_index):
        """Marks a piece as unavailable (meaning it's placed) but keeps its sides available."""
        # Update the corresponding row in available_pieces to reflect it's not available
        self.available_pieces[piece_index, -1] = -1 # "-1" means unavaiable

    def update_available_connections(self, piece_id, side_index):
        '''Updates the list of available connections between pieces based on the current state of the puzzle.
        '''
        # Assume connections are indexed by piece_id and side_index
        connection_index = (piece_id * self.num_sides) + side_index # Calculate the index of the connection in the flattened array
        self.available_connections[connection_index] = -1           # Mark as unavailable

    def flatten_available_connections(self,connections_dict):
        """Flattens a dictionary of lists into a 1D array."""
        return np.concatenate([np.array(v) for v in connections_dict.values()])

    def _get_action_mask(self):
        """Returns separate masks for each dimension in the MultiDiscrete action space.
           Masking is to ensure that the policy only selects valid actions at each step.
           Basically is a way to narrow down the combinatorial explosion of possible actions. Policy should coverge faster.

           # A mask value of 0 means the action is invalid, and a value of 1 means the action is valid.

           DESIDERATA:
           1- Selected Current Piece (piece_id) Cannot be a Target Piece (target_id).
                This is inherently addressed by ensuring that the active piece is available and the target piece is already placed.
           2- Only Available Pieces Can be Selected as Current Piece (piece_id).
                This is implemented by checking piece_availability for available pieces (piece_availability == 1).
           4- Only Pieces in the Puzzle (i.e. "unavailable pieces" - where unavailable means "already_placed") Can be Considered Target Pieces (target_id).
              This is ensured by checking piece_availability for placed pieces (piece_availability == -1).
           5- Target Pieces Should Have At Least One Side Available (target_side_index).
              This is checked by ensuring that there is at least one available side (np.any(side_availability == 1, axis=1)).
        """
        num_pieces = len(self.pieces_lst)
        num_sides = len(self.sides)

        # Initialize masks
        mask_piece_id = np.zeros(num_pieces, dtype=np.uint8)         # (Desideratum 2) Only Available Pieces Can be Selected as Current Piece
        mask_target_id = np.zeros(num_pieces, dtype=np.uint8)        # Mask for selecting the target piece (Desideratum 4 and 5)
        mask_target_side_index = np.zeros((num_pieces, num_sides), dtype=np.uint8)  # Initialize as 2D array  # To be set based on target piece's available sides

        # Extract availability information
        piece_availability = self.available_pieces[:, -1]             # Last column indicates piece availability (Desideratum 2, 4, 5)
        side_availability = self.available_pieces[:, :-1]             # All columns except the last indicate side availability (Desideratum 5)

        # Determine valid active pieces (Desideratum 2)
        valid_active_pieces = np.where(piece_availability == 1)[0]
        mask_piece_id[valid_active_pieces] = 1  # Mark available pieces

        # Determine valid target pieces (Desideratum 4 and 5) -  Check if the piece is placed and has at least one available side
        valid_target_pieces = np.where((piece_availability == -1) & (np.any(side_availability == 1, axis=1)))[0]

        print(f"Valid active pieces: {valid_active_pieces}")
        print(f"Valid target pieces: {valid_target_pieces}")

        # Determine valid target sides for each target piece (Desideratum 5)
        for target_piece_idx in valid_target_pieces:
            mask_target_id[target_piece_idx] = 1  # Mark piece as a valid target
            mask_target_side_index[target_piece_idx] = (side_availability[target_piece_idx] == 1)

        # Flatten the masks to ensure compatibility with RLlib's expectations (flattening for convenience in use)
        mask_piece_id = mask_piece_id.flatten()                      # Mask showing available pieces to select as the current piece
        mask_target_id = mask_target_id.flatten()                    # Mask showing available target pieces to select as the target piece
        mask_target_side_index = mask_target_side_index.flatten()

        return mask_piece_id, mask_target_side_index

    def _get_observation(self):
        '''Get the current observation of the environment.'''
        mask_piece_id, mask_target_side_index = self._get_action_mask()  # Update the action masks before getting the observation

        observation = {
            "current_puzzle": self.current_puzzle,                # Current state of the puzzle grid, -1 means empty
            "available_pieces": self.available_pieces,            # List of available pieces and their sides, -1 means unavailable, 1 means available
            "available_connections": self.available_connections,  # List of available connections between pieces, -1 means unavailable
            "mask_piece_id": mask_piece_id,                       # Mask for selecting the active piece (Desideratum 2) - Only Available Pieces Can be Selected as Current Piece
            "mask_target_side_index": mask_target_side_index      # Mask for selecting the target piece and the side (Desideratum 4 and 5)
        }

        return observation

    def sides_match(self, piece, side_index, target_piece, target_side_index):
        # Validate matching criteria for sides - checks whether two pieces can be legally connected at specified sides.
        # This should be updated to more complexity could mean that the sides have complementary shapes, colors, numbers, or any other criteria that define a correct connection in the puzzle.
        return piece.sides_lst[side_index] == target_piece.sides_lst[target_side_index]


    #TODO: recheck this method
    def place_piece(self, current_piece_id, side_index, target_position):
        """
        Place a piece adjacent to the target piece based on the connecting side.

        This function determines the correct new position for the current piece relative to the target piece,
        using the side_index to infer the correct adjacency direction. It ensures that the placement
        is within the boundaries of the puzzle grid and that the calculated position is not already occupied.

        Parameters:
        - current_piece_id (int): The ID of the current piece to be placed.
        - side_index (int): The index representing the side of the target piece where the current piece will connect.
        - target_position (tuple): The current position (row, column) of the target piece in the grid.

        Returns:
        - bool: True if the piece was successfully placed, False otherwise.
        """

        # Mapping from side indices to their corresponding row and column offsets in the grid.
        # This map translates the side connection into grid coordinates:
        # 0 - Top: place the current piece above the target piece.
        # 1 - Right: place the current piece to the right of the target piece.
        # 2 - Bottom: place the current piece below the target piece.
        # 3 - Left: place the current piece to the left of the target piece.
        side_to_position = {
            0: (-1, 0),  # Mapping side '0' (top) to move current piece above the target
            1: (0, 1),   # Mapping side '1' (right) to move current piece to the right of the target
            2: (1, 0),   # Mapping side '2' (bottom) to move current piece below the target
            3: (0, -1)   # Mapping side '3' (left) to move current piece to the left of the target
        }

        # Validate that the side_index provided is one that we have mapped to a position.
        if side_index in side_to_position:
            # Extract the row and column offset based on the side index.
            row_offset, col_offset = side_to_position[side_index]
            # Calculate the new position for the current piece based on the target's position and the offsets.
            new_position = (target_position[0][0] + row_offset, target_position[1][0] + col_offset)

            # Check if the new position is within the grid bounds and the specified cell is empty (not occupied).
            if (0 <= new_position[0] < self.grid_size) and (0 <= new_position[1] < self.grid_size) and (self.current_puzzle[new_position[0], new_position[1]] == -1):
                # If the position is valid and empty, place the current piece at this new position on the grid.
                self.current_puzzle[new_position] = current_piece_id
                return True  # Indicate successful placement.

        # If the side index was not valid, or the position was out of bounds or occupied, return False.
        return False


    def process_action(self, action):
        '''Process the action to connect two pieces if the rules permit.
            First checks if the action is valid  - if the two pieces can be connected based on the puzzle's rules.
            If valid action: method returns True  - and next method assigns rewards. If invalid action: method returns False.
            If valid action: updates the puzzle state and available connections.
            Update the available connections based on the newly connected piece's side and remove the active piece from the list of available pieces.
            Agent will then receive a reward based on whether the action was valid
            meaning that the pieces could be connected and the puzzle state was updated accordingly.
        '''

        current_piece_id, side_index, combined_target_index = action

        # Decode the combined_target_index to get target_piece_id and target_side_index
        # This calculation determines which piece and which side of that piece is being targeted
        target_piece_id = combined_target_index // self.num_sides  # Calculate target piece ID
        target_side_index = combined_target_index % self.num_sides # Calculate side index of the target piece

        print(f"Processing action: Connect selected piece {current_piece_id} at side {side_index} to target piece {target_piece_id} at side {target_side_index}")

        # Check if the selected current piece is still available to be played (i.e., not already placed)
        if self.available_pieces[current_piece_id, -1] == -1:
            print("Selected current piece is not available.")
            return False

        # Check if the current piece is available and not already placed
        if self.available_pieces[current_piece_id, -1] == 1:
            # Check if the target piece is already placed in the puzzle and has at least one available side to connect
            if self.available_pieces[target_piece_id, -1] == -1 and self.available_pieces[target_piece_id, target_side_index] == 1:
                # Check if the selected sides on the active and target pieces can legally connect
                if self.sides_match(self.pieces_lst[current_piece_id], side_index, self.pieces_lst[target_piece_id], target_side_index):
                        # If placement is successful, mark the active piece as no longer available
                        self.mark_piece_unavailable(current_piece_id)
                        # Register that the sides involved in the connection are no longer available
                        self.pieces_lst[current_piece_id].connect_side(side_index)
                        self.pieces_lst[target_piece_id].connect_side(target_side_index)
                        # Update the available connections array to reflect the new state
                        # Since we are connecting 2 pieces, we need to update the available connections for both pieces - meaning that the sides that were connected are no longer available.
                        self.update_available_connections(current_piece_id, side_index)         # Update the available connections based on the newly connected piece's side
                        self.update_available_connections(target_piece_id, target_side_index)   # Update the available connections for the target piece as well

                        print(f"Connected piece {current_piece_id} side {side_index} to piece {target_piece_id} side {target_side_index}")

                        return True                                                             # Return True to indicate a valid and successful action that has modified the puzzle's state.
                else:
                        print(f"Sides unmatched for piece {current_piece_id} side {side_index} and piece {target_piece_id} side {target_side_index}")
            else:
                    print(f"Cannot connect piece {current_piece_id} side {side_index} with piece {target_piece_id} side {target_side_index}")
        else:
                print(f"Target piece {target_piece_id} or side {target_side_index} not available.")

        return False                                                                            # Return False if any condition fails and the pieces cannot be connected as intended



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
        self.mark_piece_unavailable(start_piece_id)           # Mark the starting piece as unavailable (it's already placed)

        print(f"Starting piece: {start_piece_id} placed in puzzle grid")

        # TODO: go throught the action mask and review it thoroughly

        return self._get_observation(), {}


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

        print(f"Reward: {reward}")
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
    def visualize_puzzle(self, mode='human'):
        '''Visualize the current state of the puzzle in ASCII format.
        '''
        if mode == 'human':
            # Fetch the current puzzle state from the environment
            current_puzzle = self.current_puzzle
            output = ""
            # Iterate through each row in the puzzle grid
            for row in current_puzzle:
                line = ""
                # Iterate through each cell in the row
                for piece_id in row:
                    if piece_id == -1:
                        # If the cell is empty (-1), represent it with an empty placeholder
                        line += "[   ] "
                    else:
                        # Otherwise, display the piece ID, formatted to be 3 characters wide
                        line += f"[{piece_id:3d}] "
                # Add this row to the output, with a new line at the end
                output += line + "\n"
            print(output)



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

        #  The action is defined as a tuple of 3 values: (piece_id, target_id, side_index * target_side_index)
        self.action_space = MultiDiscrete([
            self.env.num_pieces,  # active_piece_id
            self.env.num_sides,   # active_piece_side_index
            self.env.num_pieces * self.env.num_sides  # combined dimension for target_piece_id and side_target_piece_index
        ])

        # Include a masking on the policy to dynamically indicate which actions are valid at any given state of the environment.
        self.observation_space = Dict({
                    'current_puzzle': Box(low=-1, high=1, shape=(self.env.grid_size, self.env.grid_size)),         # 2x2 grid for a 4-piece puzzle
                    'available_pieces': Box(low=-1, high=1, shape=(self.env.num_pieces, self.env.num_sides + 1), dtype=np.int8),    # Availability of pieces and sides
                    'available_connections': Box(low=-1, high=1, shape=(self.env.num_pieces * self.env.num_sides,), dtype=np.int8), # Availability of connections
                    'mask_piece_id': Box(low=0, high=1, shape=(self.env.num_pieces,), dtype=np.uint8),                              # Mask for selecting the active piece - Only Available Pieces Can be Selected as Current Piece
                   'mask_target_side_index': Box(low=0, high=1, shape=(self.env.num_pieces * self.env.num_sides, ), dtype=np.uint8),   # Mask for selecting the target piece and side in 2D - If the piece is placed and has at least one available side
                })

        '''
        #NOT SURE IF STILL NEEDED
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
    config = {
        'sides': [5, 6, 7, 8],  # Sides are labeled to be different from the keynumbers: "1" for available, etc.
        'num_pieces': 4,
        }

    env = PuzzleGymEnv(config) # Initialize the env (including reset)

    num_steps = 10

    for _ in range(num_steps):
        action = env.action_space.sample()       # action is to connect available pieces and sides with the target side
        piece_id, target_id, target_side_idx = action
        print(f"Action: (active_piece: {piece_id}, target_piece: {target_id}, target_side: {target_side_idx})")
        obs, reward, terminated, truncated, info = env.step(action)  # the observation is the current state of the puzzle and available pieces
        print(f"Reward: {reward}, Done: {terminated}")

        env.render()  # Visualize the state of the environment #TODO: fix the visualization

        if terminated:
            print("The puzzle has been solved or the episode is over!")
            break

    env.close()  # Properly close the environment
