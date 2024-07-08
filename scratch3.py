import numpy as np
import gymnasium as gym
from gymnasium import spaces


''' NEW OBSERVATION SPACE AND A GOOD DISPLAY FUNCTION:

    Piece Sides: The number of sides each piece has.
    Connections: The connections between the pieces, indicating which sides are connected.
    Available Sides: Which sides are still available for connection.
    Piece Identifiers: Identifiers for each piece to help the agent distinguish between them.

    piece_sides: Helps the agent understand the structure of the pieces.
    connections: Provides information about the current state of the connections.
    available_sides: Indicates which sides are available for making new connections.
    piece_ids: Helps in identifying and differentiating between pieces.

'''


class Piece:
    def __init__(self, id, sides):
        self.id = id
        self.sides_lst = sides
        self.available_sides = [True] * len(sides)

    def rotate(self, degrees):
        num_sides = len(self.sides_lst)
        if degrees % (360 // num_sides) != 0:
            return
        num_rotations = degrees // (360 // num_sides)
        self.sides_lst = self.sides_lst[-num_rotations:] + self.sides_lst[:-num_rotations]
        self.available_sides = self.available_sides[-num_rotations:] + self.available_sides[:-num_rotations]

    def connect_side(self, side_index):
        if self.available_sides[side_index]:
            self.available_sides[side_index] = False
            return True
        return False

    def copy(self):
        new_piece = Piece(self.id, self.sides_lst.copy())
        new_piece.available_sides = self.available_sides.copy()
        return new_piece

class Puzzle(gym.Env):
    def __init__(self):
        super(Puzzle, self).__init__()
        self.pieces = self.initialize_pieces()
        self.connections = np.full((len(self.pieces), max(len(p.sides_lst) for p in self.pieces.values()), 2), -1, dtype=int)

        max_sides = max(len(piece.sides_lst) for piece in self.pieces.values())
        num_pieces = len(self.pieces)

        self.observation_space = spaces.Dict({
            "piece_sides": spaces.Box(low=0, high=max_sides, shape=(num_pieces,), dtype=np.int32),
            "connections": spaces.Box(low=-1, high=num_pieces, shape=(num_pieces, max_sides, 2), dtype=np.int32),
            "available_sides": spaces.MultiBinary((num_pieces, max_sides)),
            "piece_ids": spaces.Box(low=0, high=num_pieces-1, shape=(num_pieces,), dtype=np.int32)
        })
        self.action_space = spaces.Discrete(num_pieces)

    def initialize_pieces(self):
        pieces = {
            0: Piece(0, [0, 1, 2, 3]),
            1: Piece(1, [0, 1, 2, 3]),
            2: Piece(2, [0, 1, 2, 3]),
            3: Piece(3, [0, 1, 2, 3])
        }
        return pieces

    def reset(self):
        self.pieces = self.initialize_pieces()
        self.connections.fill(-1)  # Reset connections
        return self.get_observation()

    def get_observation(self):
        piece_sides = [len(piece.sides_lst) for piece in self.pieces.values()]
        available_sides = [piece.available_sides for piece in self.pieces.values()]
        piece_ids = [piece.id for piece in self.pieces.values()]
        return {
            "piece_sides": np.array(piece_sides),  # [4, 4, 4, 4]
            "connections": self.connections,       # 4-D matrix representing connections for each piece and side (see below)
            "available_sides": np.array(available_sides), # list of list of bools representing available sides for each piece
            "piece_ids": np.array(piece_ids)  # [0, 1, 2, 3]
        }

    def step(self, action):
        observation = self.get_observation()
        reward = 0
        done = False
        return observation, reward, done, {}

    def connect_pieces(self, piece_a_id, side_a, piece_b_id, side_b):
        '''Connect two pieces if the sides are available and compatible.'''
        if self.pieces[piece_a_id].connect_side(side_a) and self.pieces[piece_b_id].connect_side(side_b):
            self.connections[piece_a_id, side_a] = [piece_b_id, side_b]
            self.connections[piece_b_id, side_b] = [piece_a_id, side_a]
            return True
        return False

    def display_connections(self):
        num_pieces = len(self.pieces)
        max_sides = max(len(piece.sides_lst) for piece in self.pieces.values())

        display = []
        for piece_id in range(num_pieces):
            for side_id in range(max_sides):
                connected_piece, connected_side = self.connections[piece_id, side_id]
                if connected_piece != -1:
                    display.append(f"Piece {piece_id} side {side_id} is connected to Piece {connected_piece} side {connected_side}")

        if not display:
            display.append("No connections")

        return "\n".join(display)

# Example usage
env = Puzzle()
obs = env.reset()

# Connect pieces as specified
'''Side 1 of Piece 0 is connected to Side 1 of Piece 1.
Side 2 of Piece 0 is connected to Side 2 of Piece 2.
Side 3 of Piece 0 is connected to Side 3 of Piece 3.'''

env.connect_pieces(0, 1, 1, 1) # (piece_a_id, side_a, piece_b_id, side_b)
env.connect_pieces(0, 2, 2, 2) # (piece 0 side 2) connected to (piece 2 side 2)
env.connect_pieces(0, 3, 3, 3)

obs = env.get_observation()

print("Piece Sides:", obs["piece_sides"])
print("Connections:\n", env.display_connections())
print("Available Sides:\n", obs["available_sides"])
print("Piece IDs:\n", obs["piece_ids"])

'''
Visualization and Explanation of the 'connections' array:

    Piece Sides: [4, 4, 4, 4]
        Each piece has 4 sides.

    Connections Array:
        For piece 0:
            side 0: [-1, -1] (not connected)
            side 1: [1, 1] (connected to side 1 of piece 1)
            side 2: [2, 2] (connected to side 2 of piece 2)
            side 3: [3, 3] (connected to side 3 of piece 3)
        For piece 1:
            side 0: [-1, -1] (not connected)
            side 1: [0, 1] (connected to side 1 of piece 0)
            side 2: [-1, -1] (not connected)
            side 3: [-1, -1] (not connected)
        For piece 2:
            side 0: [-1, -1] (not connected)
            side 1: [-1, -1] (not connected)
            side 2: [0, 2] (connected to side 2 of piece 0)
            side 3: [-1, -1] (not connected)
        For piece 3:
            side 0: [-1, -1] (not connected)
            side 1: [-1, -1] (not connected)
            side 2: [-1, -1] (not connected)
            side 3: [0, 3] (connected to side 3 of piece 0)





'''