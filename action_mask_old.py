
    def _get_action_mask_old(self):
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