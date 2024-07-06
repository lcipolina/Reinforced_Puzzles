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
        self.agents         = {"high_level_policy", "low_level_policy"}          # Agent IDs - Needed by the Policy dicts
        self._agent_ids     = set(self.agents)                                   # Needed by Policy and by terminateds_dict
        self.sides          = config.get("sides", [[5, 6, 7, 8]])                # List of Lists -  Sides are labeled to be different from the keynumbers: "1" for available, etc.
        self.num_pieces     = config.get("num_pieces", 4)                        # Number of pieces in the puzzle
        self.num_sides      = len(self.sides)
        self.DEBUG          = config.get("DEBUG", False)                         # Whether to print or not
        self.grid_size      = config.get("grid_size", 10)                        # height and width # int(np.sqrt(self.num_pieces))  # Generates 4 pieces with 4 sides
        self.pieces_lst     = Piece._generate_pieces(sides_lst =self.sides)      # Generate pieces, sides and availability

        # Define the puzzle grid dimensions (Ex: 2x2 for 4 pieces)
        # Current puzzle is an array (then converted to graph for comparisons), target puzzle is a graph
        self.current_puzzle = np.full((self.grid_size, self.grid_size), -1, dtype=np.int8)              # (grid_size)x(grid_size) grid for a N-piece puzzle , "-1"represents an empty cell in the puzzle grid
        self.available_pieces_sides = np.full((self.num_pieces, self.num_sides + 1), 1, dtype=np.int8)  # Availability of pieces and sides.  "1" represents available - where each row represents a piece, and the last element in each row indicates the availability of the piece.
        self.available_connections  = np.full((self.num_pieces * self.num_sides,), 1, dtype=np.int8)    # "1" represents available  of connections as a flat array, where each element corresponds to a specific connection.
        self.target_puzzle = nx.Graph()                                                                 # Target configuration as a graph.
        self._setup_target_puzzle()                                                                     # Defines target puzzle as a graph based on pre-defined pieces and connections
        self.reset()


    #------------------------------------------------------------------------------------------------
    # Puzzle handling
    #------------------------------------------------------------------------------------------------

    # TODO: need to fix this with the correct side number
    def place_piece(self, rotated_piece, side_lbl, target_position, target_side_lbl):
        '''Place a piece adjacent to the target piece based on the connecting sides.'''

        # Mapping of side indices to position offsets based on the connection logic
        # TODO: connection and matching is based on side's lables THIS WILL NOT WORK - URGENT CHANGE
        connection_map = {
            (1, 1): (-1, 0),  # Current top side connects to target top side
            (2, 2): (0, 1),   # Current right side connects to target right side
            (3, 3): (1, 0),   # Current bottom side connects to target bottom side
            (4, 4): (0, -1)   # Current left side connects to target left side
        }

        # Calculate the new position based on the target piece's position and the connecting sides
        if (side_lbl, target_side_lbl) in connection_map:
            row_offset, col_offset = connection_map[(side_lbl, target_side_lbl)]
            new_position = (target_position[0] + row_offset, target_position[1] + col_offset)

            # Ensure the new position is within bounds and not already occupied
            #TODO: change this
            if (0 <= new_position[0] < self.grid_size) and (0 <= new_position[1] < self.grid_size):
                if self.current_puzzle[new_position[0], new_position[1]] == -1:  # Check if the new position is empty
                    self.current_puzzle[new_position[0], new_position[1]] = rotated_piece.id
                    self.pieces_lst[rotated_piece.id] = rotated_piece  # Update the piece in the list with its new rotation
                    return True
                else:
                    # TODO: review this
                    my_print(f"Position {new_position} is already occupied.", self.DEBUG)

        my_print(f"Invalid placement for piece {rotated_piece.id} at side {side_lbl} adjacent to target position {target_position}", self.DEBUG)
        return False

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

    #TODO: revisit this, For now I am coding something like a domino. This might not be needed
    def update_current_puzzle(self, current_piece_id, target_piece_id, current_side_id, target_side_id):
        '''Update the current puzzle grid by placing the current piece adjacent to the target piece based on the connecting side.
           Finds the matching (labeled) sides and places the current piece in the correct position.
        '''
        target_position = np.argwhere(self.current_puzzle == target_piece_id)  # Find the position of the target piece in the puzzle grid
        if target_position.size > 0:
            target_position = tuple(target_position[0])                  # Convert the position to a tuple
            for rotation in range(4):                                    # Try all 4 possible orientations
                rotated_piece = self.pieces_lst[current_piece_id].copy() # Create a copy of the current piece
                rotated_piece.rotate(rotation * 90)                      # Use existing rotate method
                # Calculate the corresponding side index of the rotated piece
                adjusted_side_idx = (current_side_id + rotation) % 4     # This side will be connected to the target piece
                # Check if side lables match after the rotation
                if rotated_piece.sides_lst[adjusted_side_idx] == self.pieces_lst[target_piece_id].sides_lst[target_side_id]:
                   target_side_lbl = self.pieces_lst[target_piece_id].sides_lst[target_side_id]  # Convert target side idx to label
                   adjusted_side_lbl =  rotated_piece.sides_lst[adjusted_side_idx]
                   # Place piece in target position and connect adjusted side to target side based on matching side LBLs
                   if self.place_piece(rotated_piece, adjusted_side_lbl, target_position, target_side_lbl):
                        return True
        return False

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

    def update_available_connections_n_sides(self, piece_id, side_idx):
        '''Updates the list of available connections between pieces based on the current state of the puzzle.
        '''
        # Assume connections are indexed by piece_id and side_index
        connection_idx = (piece_id * self.num_sides) + side_idx  # Calculate the index of the connection in the flattened array
        self.available_connections[connection_idx] = -1          # Mark connection as unavailable
        self.available_pieces_sides[piece_id, side_idx]  = -1    # Mark side as unavailable

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

    def sides_match(self, current_piece_id, current_side_lbl, target_piece_id, target_side_lbl):
        ''' Validate matching criteria for sides - checks whether two pieces can be legally connected at specified sides.
            OBS: Policy selects pieces and sides idx but the checking is on the sides labels!
        '''
        # TODO: This should be updated to more complexity could mean that the sides have complementary shapes, colors, numbers, or any other criteria that define a correct connection in the puzzle.
        return current_side_lbl ==  target_side_lbl


    #------------------------------------------------------------------------------------------------
    # Observation
    #------------------------------------------------------------------------------------------------
    def _get_action_mask(self):
        """Returns separate masks for each dimension in the MultiDiscrete action space.

           Mask for: Valid active pieces, target pieces, and the target pieces' sides.

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
        num_sides  = len(self.sides)

        # Initialize masks
        mask_active_piece = np.zeros(num_pieces, dtype=np.uint8)                        # (Desideratum 2) Only available pieces can be selected as current piece
        mask_target_piece_n_side = np.zeros((num_pieces, num_sides), dtype=np.uint8)  # 2D mask - available target piece and corresp side of the target piece  (Desideratum 4 and 5)

        # Extract availability information
        piece_availability = self.available_pieces_sides[:, -1]             # List marking whether pieces are available or not - Last column indicates piece availability (Desideratum 2, 4, 5)
        side_availability = self.available_pieces_sides[:, :-1]             # All columns except the last indicate side availability (Desideratum 5)

        # Determine valid active pieces (Desideratum 2)
        valid_active_pieces = np.where(piece_availability == 1)[0]          # Pieces candidate to be selected as the current piece
        mask_active_piece[valid_active_pieces] = 1                              # Mark available pieces

        # Determine valid target pieces (Desideratum 4 and 5) -  Check if the piece is placed and has at least one available side
        valid_target_pieces = np.where((piece_availability == -1) & (np.any(side_availability != -1, axis=1)))[0]

        my_print(f"Valid active pieces: {valid_active_pieces}",self.DEBUG)
        my_print(f"Valid target pieces: {valid_target_pieces}",self.DEBUG)

        # Determine valid target piece and sides (Desideratum 4 and 5)
        for target_piece_idx in valid_target_pieces:
            mask_target_piece_n_side[target_piece_idx] = (side_availability[target_piece_idx] != -1)

        # Flatten the masks to ensure compatibility with the defined observation space (for convenience)
        mask_active_piece = mask_active_piece.flatten()               # Mask for available pieces to select as the current piece
        mask_target_piece_n_side = mask_target_piece_n_side.flatten()    # Mask for available target piece and sides

        return mask_active_piece, mask_target_piece_n_side

    def _get_observation(self, agent_id):
        '''Get the current observation of the environment.'''

        #TODO: think about the mask - #TODO: check if the mask is correct
        mask_active_piece, mask_target_piece_n_side = self._get_action_mask()  # Mask for: Valid active pieces, target pieces, and the target pieces' sides.

        #print("self.current_puzzle:",self.current_puzzle)
        #print("self.available_pieces_sides:",self.available_pieces_sides)

        if agent_id == "high_level_policy":  # Selects the target piece and side
            observation = {
                "current_puzzle":         self.current_puzzle,             # Current state of the puzzle grid, -1 means empty
                "available_pieces_sides": self.available_pieces_sides,     # List of available pieces and their sides, -1 means unavailable, 1 means available
                "available_connections":  self.available_connections,      # List of available connections between pieces, -1 means unavailable
                "mask_target_side_index": mask_target_piece_n_side          # Mask for selecting the target piece and the side (Desideratum 4 and 5)
            }
        else:  # Obs for the low_level_agent - Selects the active piece and side
            observation = {
                "current_puzzle":         self.current_puzzle,             # Current state of the puzzle grid, -1 means empty
                "available_pieces_sides": self.available_pieces_sides,     # List of available pieces and their sides, -1 means unavailable, 1 means available
                "available_connections":  self.available_connections,      # List of available connections between pieces, -1 means unavailable
                "target_piece":           self.target_piece_id,            # The IDX of target piece selected by the high-level agent
                "target_side":            self.target_side_lbl,            # The label side of the target piece selected by the high-level agent
                "mask_piece_id":          mask_active_piece,               # Mask for selecting the active piece- Only available pieces can be selected as current piece
            }
        return observation


    #------------------------------------------------------------------------------------------------
    # Action
    #------------------------------------------------------------------------------------------------
    def process_action(self, action):
        '''Process the action for the Low-level agent. Connect the chosen active piece to the target piece and side.
            Checks if the action is valid  - if the two pieces can be connected based on the puzzle's rules.
            If valid action: method returns True  - and the step method assigns rewards. If invalid action: method returns False.
            If valid action: updates the puzzle state, the side availability and the available connections.
            Update the available connections based on the newly connected piece's side and remove the active piece from the list of available pieces.
            Agent will then receive a reward based on whether the action was valid - meaning that the pieces could be connected and the puzzle state was updated accordingly.
        '''
        # Convert from side Idx to side label. Rationale: Policy selects idx (0,..,3) but sides have lables that need to match. Ex: 5,6,7,8)
        current_piece_id, current_side_id = action
        current_piece_obj = next((piece for piece in self.pieces_lst if piece.id == current_piece_id), None)   # From piece_idx to piece_object
        current_side_lbl  = current_piece_obj.sides_lst[current_side_id]                                       # This is the one used for comparison

        my_print(f"Processing action: Connect selected piece {current_piece_id} at side labled {current_side_lbl} to target piece {self.target_piece_id} at side labled {self.target_side_lbl}",self.DEBUG)

        # Check if the selected current piece is still available to be played (i.e., not already placed)
        if self.available_pieces_sides[current_piece_id, -1] == -1:
            my_print("Selected current piece is not available.",self.DEBUG)
            return False

        # Check if the current piece is available and not already placed
        if self.available_pieces_sides[current_piece_id, -1] == 1:
            # Check if the target piece is already placed in the puzzle and has at least one available side to connect
            if self.available_pieces_sides[self.target_piece_id, -1] == -1 and self.available_pieces_sides[self.target_piece_id, self.target_side_id] != -1:
                # Check if the selected sides on the active and target pieces can legally connect
                if self.sides_match(current_piece_id, current_side_lbl, self.target_piece_id, self.target_side_lbl):
                        # If placement is successful, update the active piece and target_piece, sides and connections as no longer available
                        self.update_current_puzzle(current_piece_id, self.target_piece_id, current_side_id, self.target_side_id) # Update self.current_puzzle
                        self.update_pieces_sides(current_piece_id,self.target_piece_id, current_side_id, self.target_side_id)
                        my_print(f"Connected piece {current_piece_id} to side labled {current_side_lbl} to piece {self.target_piece_id} to side labled {self.target_side_lbl}",self.DEBUG)
                        return True                                                             # Return True to indicate a valid and successful action that has modified the puzzle's state.
                else:
                        my_print(f"Sides unmatched for piece {current_piece_id} to side labled {current_side_lbl} and piece {self.target_piece_id} to side labled {self.target_side_lbl}",self.DEBUG)
            else:
                    my_print(f"Cannot connect piece {current_piece_id} to side labled {current_side_lbl} with piece's {self.target_piece_id} to side labled {self.target_side_lbl}",self.DEBUG)
        else:
                my_print(f"Target piece {self.target_piece_id} or side labled {self.target_side_lbl} not available.", self.DEBUG)

        return False                                                                            # Return False if any condition fails and the pieces cannot be connected as intended

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
        '''Process action: converts idx to lables and returns the observation for the next agent.'''
        target_piece_n_side = action

#TODO: this won't work for different number of sides
        # Convert from side Idx to side label. Rationale: Policy selects idx (0,..,3) but sides have lables that need to match. Ex: 5,6,7,8)
        self.target_piece_id = target_piece_n_side // self.num_sides            # Calculate target piece ID
        target_piece_obj = next((piece for piece in self.pieces_lst if piece.id == self.target_piece_id), None)   # From piece_idx to piece_object
        self.target_side_id = target_piece_n_side % self.num_sides              # Calculate side index of the target piece - used for the piece placement in the puzzle - The 'reminder' is a trick to cycle through a *fixed* range of numbers from 0 to (num_sides -1)
        self.target_side_lbl  = target_piece_obj.sides_lst[self.target_side_id] # This is the one used for matching checks
        obs = {"low_level_policy": self._get_observation("low_level_policy")}   # Observation for the next agent
        rew = {"high_level_policy": 0}                                          # TODO: this might need to be enhanced later
        self.truncated_dict = {"__all__": False}                                # High-level agent never terminates a game

        my_print(f"Target piece id: {self.target_piece_id},  side_lbl: {self.target_side_lbl}, Reward:{rew}", self.DEBUG)
        return obs, rew, self.terminateds_dict, self.truncated_dict, {}

    def _low_level_step(self, action):
        '''Low-level agent connects the active piece to the target piece and side.'''
        obs, rew = {}, {}
        valid_action = self.process_action(action)                               # Process idx to lbl and Check validity and update connections if valid action
        if valid_action:
           reward = self.calculate_reward()
           terminated = self.check_completion()
        else:
           reward = -1                                                           # Penalize invalid actions without updating the state
           terminated = False                                                    # The environment only ever terminates when we reach the goal state.
        obs = {"high_level_policy": self._get_observation("high_level_policy")}  # Obs for the next agent
        rew["high_level_policy"], rew["low_level_policy"]  = reward, reward
        self.terminateds_dict["__all__"]= terminated
        self.truncated_dict = self.terminateds_dict
        return obs, rew, self.terminateds_dict, self.truncated_dict, {}

    #------------------------------------------------------------------------------------------------
    # reset
    #------------------------------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        '''High level agent selects a target node and a side to connect to.
           Initializes the puzzle grid and available pieces and sides.
        '''
        if seed is not None: random.seed(seed)   # as per new gymnasium
        self.terminateds_dict = {agent: False for agent in self._agent_ids}
        self.terminateds_dict['__all__']   = False
        self.truncated_dict = {agent: False for agent in self._agent_ids}
        self.truncated_dict['__all__']   = False

        # Reset current puzzle mark all pieces and connections as available
        self.current_puzzle.fill(-1)             # Reset the puzzle grid to all -1, indicating empty cells
        self.update_pieces_sides()               # Mark all pieces and sides as available (1)
        self.available_connections.fill(1)       # Mark all connections as available (1)
        self.target_piece_id = -1
        self.target_side_id = 0
        self.target_side_lbl = -1  # TODO: not sure if this is needed and the right value!
        piece_ids         = [piece.id for piece in self.pieces_lst]  # Take each piece ID and Create a list of piece IDs from the already existing pieces_lst
        # TODO: np.random.shuffle(piece_ids)

        # Need to start with a placed piece, otherwise there is no available target piece and the action mask fails
        start_piece_id  = piece_ids[0]                                 # Start with the first piece in the list
        middle_position = (self.grid_size // 2, self.grid_size // 2)   # Place the first piece in the middle of the grid
        self.current_puzzle[middle_position] = start_piece_id
        self.update_pieces_sides(start_piece_id)                       # Mark the starting piece as unavailable (it's already placed)

        my_print(f"Starting piece: {start_piece_id} placed in puzzle grid at {middle_position}", self.DEBUG)

        # First playing agent is the hihg level agent
        return {"high_level_policy": self._get_observation("high_level_policy")} , {}

    #------------------------------------------------------------------------------------------------
    # step
    #------------------------------------------------------------------------------------------------
    def step(self, action_dict):
        ''' Each Pol converts from selected idx to labels and returns the observation for the next agent.
            The matching is done at the label level.
        '''

        if "high_level_policy" in action_dict:
          my_print(f"High-level action: {action_dict}", self.DEBUG)
          return self._high_level_step(action_dict["high_level_policy"])
        else:
          my_print(f"Low-level action: {action_dict}", self.DEBUG)
          return self._low_level_step(action_dict["low_level_policy"])



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
        #  High level - select target piece and side: (piece_id, side_idx)
        self.high_level_action_space = Discrete(
            self.env.num_pieces * self.env.num_sides)  # combined dimension for target_piece_id and side_target_piece_index

        #  Low level - select active piece and side to connect to (side mask depends on the target piece)
        self.low_level_action_space = MultiDiscrete([
            self.env.num_pieces,                       # active_piece_id
            self.env.num_sides                         # side_index. The INDEX of the side, not the actual value.
        ])
        # High level - select target piece and side: (piece_id, side_index)
        self.high_level_obs_space = Dict({
                   'current_puzzle'         : Box(low=-1, high=100, shape=(self.env.grid_size, self.env.grid_size)),                        # 2x2 grid for a 4-piece puzzle
                   'available_pieces_sides' : Box(low=-1, high=100, shape=(self.env.num_pieces, self.env.num_sides + 1), dtype=np.int8),    # Availability of pieces and sides. Columns are the side values, and the last column indicates the availability of the piece
                   'available_connections'  : Box(low=-1, high=1, shape=(self.env.num_pieces * self.env.num_sides,), dtype=np.int8),           # Availability of connections
                   'mask_target_side_index' : Box(low=0, high=1, shape=(self.env.num_pieces * self.env.num_sides, ), dtype=np.uint8),          # Mask for selecting the target piece and side in 2D - If the piece is placed and has at least one available side
                })
        #  Low level - select active piece and side given the target piece and side
        self.low_level_obs_space = Dict({
                    'current_puzzle'         : Box(low=-1, high=np.inf, shape=(self.env.grid_size, self.env.grid_size)),                     # 2x2 grid for a 4-piece puzzle
                    'available_pieces_sides' : Box(low=-1, high=np.inf, shape=(self.env.num_pieces, self.env.num_sides +1), dtype=np.int8),     # Availability of pieces and sides. Columns are the side values, and the last column indicates the availability of the piece
                    'available_connections'  : Box(low=-1, high=1, shape=(self.env.num_pieces * self.env.num_sides,), dtype=np.int8),        # Availability of connections
                    'mask_piece_id'          : Box(low=0, high=1, shape=(self.env.num_pieces,), dtype=np.uint8),                             # Mask for selecting the active piece - Only Available Pieces Can be Selected as Current Piece
                    'target_piece'           : Discrete(self.env.num_pieces),  # target piece selected by the high-level agent
                    'target_side'            : Discrete(self.env.num_sides),   # side of the target piece selected by the high-level agent
                })
        # To map the action space to the right policy.
        self._action_space_in_preferred_format = True # This means that the action space is a dictionary with the agent_id as key
        self.action_space = Dict({
                "high_level_policy": self.high_level_action_space,
                "low_level_policy": self.low_level_action_space
            })
        self._obs_space_in_preferred_format = True # This means that the observation space is a dictionary with the agent_id as key
        self.observation_space = Dict({
                "high_level_policy": self.high_level_obs_space,
                "low_level_policy": self.low_level_obs_space
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
    # Piece ID is taken from the index of the list of pieces - 0, 1, 2, 3 - TODO: need to change and add an ID dimension
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
        target_piece_n_side  = action['high_level_policy']
        target_piece = target_piece_n_side // num_sides  # Calculate target piece ID
        target_side  = target_piece_n_side % num_sides   # Calculate side index of the target piece
        active_piece, active_side = action['low_level_policy']
        print(f"Action: active_piece: {active_piece}, side:{active_side}, target_piece: {target_piece}, side:{target_side}")

        obs, reward, terminated, truncated, info = env.step(action)  # the observation is the current state of the puzzle and available pieces
        print(f"Reward: {reward}, Done: {terminated}")

        env.render()                                     # Convert current puzzle into string for visualization

        if terminated:
            print("The puzzle has been solved or the episode is over!")
            break

    env.close()  # Properly close the environment
