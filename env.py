import networkx as nx
import matplotlib.pyplot as plt
import random

class Piece:
    def __init__(self, id, sides):
        self.id = id
        self.sides = sides  # List of side values
        self.available_sides = [True] * len(sides)  # All sides are initially available for connection

    def rotate(self, degrees):
        if degrees % 90 != 0:
            return
        num_rotations = degrees // 90
        self.sides = self.sides[-num_rotations:] + self.sides[:-num_rotations]
        self.available_sides = self.available_sides[-num_rotations:] + self.available_sides[:-num_rotations]

    def connect_side(self, side_index):
        if self.available_sides[side_index]:
            self.available_sides[side_index] = False
            return True
        return False

    def copy(self):
        # Create a new Piece instance with the same id and sides, but independent available_sides
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
        self.target_puzzle = nx.Graph()
        self.current_puzzle = nx.Graph()
        self._setup_target_puzzle()
        self.reset()

    def check_completion(self):
        return nx.is_isomorphic(self.current_puzzle, self.target_puzzle)

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
        self.available_pieces = [piece.copy() for piece in self.pieces]  # Deep copy of pieces
        self.update_available_connections()  # Reset available connections
        return self._get_observation()  # Return initial observation

    def update_available_connections(self):
        # This method updates the list of available connections
        available_connections = []
        for node_id, attrs in self.current_puzzle.nodes(data=True):
            piece = attrs['piece']
            for side_index, is_available in enumerate(piece.available_sides):
                if is_available:
                    available_connections.append((node_id, side_index))
        return available_connections

    def _get_observation(self):
        graph_data = nx.to_dict_of_dicts(self.current_puzzle)
        available_pieces_data = [{'id': piece.id, 'sides': piece.sides} for piece in self.available_pieces]
        available_connections = self.update_available_connections()
        observation = {
            "current_puzzle": graph_data,
            "available_pieces": available_pieces_data,
            "available_connections": available_connections
        }
        return observation

    def check_completion(self):
        # Check if the current puzzle is a complete match with the target puzzle
        return nx.is_isomorphic(self.current_puzzle, self.target_puzzle)


    def step(self, action):
        piece_id, target_id, side_index, target_side_index = action
        piece = next((p for p in self.available_pieces if p.id == piece_id), None)
        target_piece = self.current_puzzle.nodes.get(target_id, {}).get('piece')

        if piece and target_piece and self.sides_match(piece, side_index, target_piece, target_side_index):
            self.current_puzzle.add_node(piece.id, piece=piece)  # Add piece to the puzzle graph
            self.current_puzzle.add_edge(piece_id, target_id)  # Connect the pieces
            self.available_pieces.remove(piece)  # Remove the piece from available pieces

            # Update the reward and completion status after the action
            reward = self.calculate_reward()
            is_done = self.check_completion()
        else:
            reward = -1  # Penalize invalid or incorrect actions
            is_done = False

        return reward, is_done


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
        # Validate matching criteria for sides
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


####################################################################################################
# EXAMPLE USAGE

# Initialize the puzzle environment
config = {'sides': [1, 2, 3, 4]}  # Assuming pieces have four sides numbered for simplicity
environment = PuzzleEnvironment(config=config)

# Manually add a piece to start the puzzle
starting_piece = Piece(1, [1, 2, 3, 4])
environment.current_puzzle.add_node(starting_piece.id, piece=starting_piece)

# Simulate adding a second piece that matches one side of the starting piece
# Let's assume the second piece has a side that matches one side of the starting piece
second_piece = Piece(2, [4, 1, 2, 3])  # This piece can match the first piece on side '1'
environment.available_pieces.append(second_piece)

# Define a valid action:
# (piece_id, target_id, side_index of piece, side_index of target_piece)
# Matching side '1' of the second piece with side '4' of the first piece
action = (2, 1, 0, 0)  # Connect side 0 of piece 2 to side 0 of piece 1

# Execute the step
reward, is_done = environment.step(action)

# Print results to see what happened
print("Reward:", reward)
print("Puzzle Completed:", is_done)
environment.visualize_puzzle()