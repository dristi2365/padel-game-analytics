# Store the previous position of each player between frames
# This lets us calculate how much they moved since the last frame
previous_positions = {}

def classify_shot(player, center_x, center_y, width, height):
    """
    Classify shot type based on player movement between frames.
    
    Instead of using fixed court positions, we track how the player
    moved since the last frame and use that movement to guess the shot.
    
    Movement logic:
    - Moving upward fast = Smash (charging toward net)
    - Moving right = Forehand (swinging right)
    - Moving left = Backhand (swinging left)
    - Barely moving = Neutral (no shot happening)
    """
    
    global previous_positions
    
    # If we haven't seen this player before, just store position and return Neutral
    if player not in previous_positions:
        previous_positions[player] = (center_x, center_y)
        return "Neutral"
    
    # Get where this player was in the previous frame
    prev_x, prev_y = previous_positions[player]
    
    # Calculate how much they moved in x and y directions
    # Positive dx = moved right, Negative dx = moved left
    # Positive dy = moved down, Negative dy = moved up
    dx = center_x - prev_x
    dy = center_y - prev_y
    
    # Update position for next frame
    previous_positions[player] = (center_x, center_y)
    
    # Calculate total movement magnitude
    # This tells us how much they moved overall
    total_movement = (dx**2 + dy**2) ** 0.5
    
    # If player barely moved, no shot is happening
    if total_movement < 8:
        return "Neutral"
    
    # Moving upward quickly = Smash
    # dy is negative when moving up in the frame
    if dy < -15:
        return "Smash"
    
    # Moving right = Forehand
    elif dx > 10:
        return "Forehand"
    
    # Moving left = Backhand
    elif dx < -10:
        return "Backhand"
    
    # Some movement but doesn't fit clear pattern
    else:
        return "Neutral"

def get_player_label(center_x, center_y, width, height):
    """
    Assign a player label based on court position.
    The court is divided into 4 quadrants to identify each player.

    Layout:
    Player 1 (top-left)  | Player 2 (top-right)
    ---------------------|---------------------
    Player 3 (bot-left)  | Player 4 (bot-right)
    """
    
    # Top half of the court (far side from camera)
    if center_y < height * 0.5:
        if center_x < width * 0.5:
            return "Player 1"
        else:
            return "Player 2"
    
    # Bottom half of the court (near side to camera)
    else:
        if center_x < width * 0.5:
            return "Player 3"
        else:
            return "Player 4"
