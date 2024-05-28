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
