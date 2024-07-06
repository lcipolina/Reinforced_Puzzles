def find_piece_and_side(sequence_number, sides_per_piece):
    """
    Find the puzzle piece and side ID given a sequence number and a list
    of the number of sides per piece.

    :param sequence_number: The global sequence number across all sides of all pieces.
    :param sides_per_piece: A list where each element is the number of sides for the corresponding piece.
    :return: A tuple (piece_index, side_id) indicating the piece and the side within that piece.
    """
    total_sides = sum(sides_per_piece)
    if sequence_number >= total_sides:
        raise ValueError("Sequence number is out of range.")

    cumulative_sides = 0
    for piece_index, num_sides in enumerate(sides_per_piece):
        if cumulative_sides + num_sides > sequence_number:
            side_id = sequence_number - cumulative_sides
            return piece_index, side_id
        cumulative_sides += num_sides

# Example usage
sides_per_piece = [4, 3, 5]  # Example: First piece has 4 sides, second has 3 sides, third has 5 sides
sequence_number = 7  # The global sequence number we're interested in

piece_index, side_id = find_piece_and_side(sequence_number, sides_per_piece)
print(f"Sequence Number {sequence_number} is Side {side_id} of Piece {piece_index}")