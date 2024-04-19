import networkx as nx
import matplotlib.pyplot as plt
import random

class Piece:
    def __init__(self, id, sides):
        self.id = id
        self.sides = sides  # sides is a list of side values

    def rotate(self, degrees):
        if degrees % 90 != 0:
            return
        num_rotations = degrees // 90
        self.sides = self.sides[-num_rotations:] + self.sides[:-num_rotations]

    def copy(self):
        # Create a new instance of Piece with the same id and a copy of the sides
        return Piece(self.id, self.sides.copy())

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
        self.target_puzzle = nx.Graph()
        self.current_puzzle = nx.Graph()
        self._setup_target_puzzle()
        self.reset()

    def _setup_target_puzzle(self):
        # Set up the target puzzle graph
        for piece in self.pieces:
            self.target_puzzle.add_node(piece.id, sides=piece.sides)
        # Assuming edges are predefined (this should be adjusted based on actual puzzle rules)
        for i in range(len(self.pieces) - 1):
            self.target_puzzle.add_edge(self.pieces[i].id, self.pieces[i + 1].id)
        # Add an edge between the first and last piece to close the loop
        self.target_puzzle.add_edge(self.pieces[0].id, self.pieces[-1].id)

    def reset(self):
        self.current_puzzle.clear()
        # Reset with fresh pieces
        self.available_pieces = [piece.copy() for piece in self.pieces]
        return self._get_observation()  # Return initial observation

    def _get_observation(self):
        graph_data = nx.to_dict_of_dicts(self.current_puzzle)
        available_pieces_data = [{'id': piece.id, 'sides': piece.sides} for piece in self.available_pieces]
        observation = {
            "current_puzzle": graph_data,
            "available_pieces": available_pieces_data
        }
        return observation

    def step(self, action):
        total_reward = self.calculate_reward(action)
        is_done = self.check_completion()
        return total_reward, is_done

    def calculate_reward(self, action):
        piece_id, target_id, rotate_degree = action
        piece = next(p for p in self.available_pieces if p.id == piece_id)
        piece.rotate(rotate_degree)
        if self.target_puzzle.has_edge(piece_id, target_id):
            self.current_puzzle.add_node(piece.id, sides=piece.sides)
            if target_id in self.current_puzzle.nodes:
                self.current_puzzle.add_edge(piece_id, target_id)
                incremental_reward = 1
            else:
                incremental_reward = -1
        else:
            incremental_reward = -1
        config_reward = self.overall_configuration_reward()
        completion_reward = self.completion_bonus()
        total_reward = incremental_reward + config_reward + completion_reward
        return total_reward

    def overall_configuration_reward(self):
        correct_edges_count = sum(1 for (u, v) in self.current_puzzle.edges if self.target_puzzle.has_edge(u, v))
        possible_correct_edges = self.target_puzzle.number_of_edges()
        return correct_edges_count / possible_correct_edges

    def completion_bonus(self):
        if self.check_completion():
            return 100.0
        return 0

    def check_completion(self):
        return nx.is_isomorphic(self.current_puzzle, self.target_puzzle)

    def visualize_puzzle(self):
        plt.figure(figsize=(12, 6))
        if len(self.target_puzzle.nodes) > 0:
            plt.subplot(122)
            pos = nx.circular_layout(self.target_puzzle)
            nx.draw(self.target_puzzle, pos, with_labels=True, node_color='lightgreen')
            plt.title("Target Puzzle")

        if len(self.current_puzzle.nodes) > 0:
            plt.subplot(121)
            pos = nx.circular_layout(self.current_puzzle)
            nx.draw(self.current_puzzle, pos, with_labels=True, node_color='lightblue')
            plt.title("Current Puzzle")
        plt.show()

# Example usage:
config = {'sides': [1, 2, 3, 4]}
my_puzzle = PuzzleEnvironment(config=config)
initial_observation = my_puzzle.reset()
print("Initial observation:", initial_observation)
my_puzzle.visualize_puzzle()

# Example of taking an action
# Assuming the action format is (piece_id, target_id, rotate_degree)
action = (1, 2, 90)  # Rotate piece 1 by 90 degrees and attempt to place it next to piece 2
reward, done = my_puzzle.step(action)
print("Reward received:", reward)
print("Puzzle solved:", done)
my_puzzle.visualize_puzzle()

