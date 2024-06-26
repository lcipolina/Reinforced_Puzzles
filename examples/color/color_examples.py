from PIL import Image, ImageDraw


################################################################################
# Draw a square with the specified colors for each side
################################################################################

def draw_colored_square(sides_colors, image_size=400):
    '''Draw a square with the specified colors for each side.'''
    if len(sides_colors) != 4:
        raise ValueError("sides_colors must have exactly 4 elements.")

    # Create a new image with a white background
    img = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(img)

    # Define the coordinates for the square's sides
    top_left = (0, 0)
    top_right = (image_size, 0)
    bottom_right = (image_size, image_size)
    bottom_left = (0, image_size)

    # Draw each side with the respective color
    draw.line([top_left, top_right], fill=sides_colors[0], width=image_size//10)  # Top side
    draw.line([top_right, bottom_right], fill=sides_colors[1], width=image_size//10)  # Right side
    draw.line([bottom_right, bottom_left], fill=sides_colors[2], width=image_size//10)  # Bottom side
    draw.line([bottom_left, top_left], fill=sides_colors[3], width=image_size//10)  # Left side

    return img

# Example usage
sides_colors = ["#00FF00", "#FF0000", "#0000FF", "#FFFF00"]
img = draw_colored_square(sides_colors)
img.save("colored_square.png")  # Save the image as a PNG file
print("Image saved as colored_square.png")



################################################################################
# Read an image and guess the color of each side
################################################################################
from PIL import Image
from collections import Counter

def get_dominant_color(colors):
    """
    Get the most common color from the list of colors.
    """
    color_counts = Counter(colors)
    dominant_color = color_counts.most_common(1)[0][0]
    return dominant_color

def guess_side_colors(image_path):
    """
    Guess the color of each side of the square image.
    """
    img = Image.open(image_path)
    width, height = img.size

    if width != height:
        raise ValueError("Image is not a square.")

    # Sampling points
    num_samples = 100  # Number of pixels to sample along each side
    step = width // num_samples

    # Collect colors for each side
    top_colors = [img.getpixel((i, 0)) for i in range(0, width, step)]
    right_colors = [img.getpixel((width - 1, i)) for i in range(0, height, step)]
    bottom_colors = [img.getpixel((i, height - 1)) for i in range(0, width, step)]
    left_colors = [img.getpixel((0, i)) for i in range(0, height, step)]

    # Guess the dominant color for each side
    top_color = get_dominant_color(top_colors)
    right_color = get_dominant_color(right_colors)
    bottom_color = get_dominant_color(bottom_colors)
    left_color = get_dominant_color(left_colors)

    # Convert RGB tuples to HEX
    top_color_hex = '#{:02x}{:02x}{:02x}'.format(*top_color)
    right_color_hex = '#{:02x}{:02x}{:02x}'.format(*right_color)
    bottom_color_hex = '#{:02x}{:02x}{:02x}'.format(*bottom_color)
    left_color_hex = '#{:02x}{:02x}{:02x}'.format(*left_color)

    return {
        "top_color": top_color_hex,
        "right_color": right_color_hex,
        "bottom_color": bottom_color_hex,
        "left_color": left_color_hex,
    }

# Example usage
image_path = "colored_square.png"  # Replace with your image path
side_colors = guess_side_colors(image_path)
print("Guessed side colors:", side_colors)


################################################################################
# HEX to English color names
################################################################################

from PIL import ImageColor

# Define a dictionary for common colors
HEX_TO_COLOR_NAME = {
    "#FFFFFF": "white",
    "#C0C0C0": "silver",
    "#808080": "gray",
    "#000000": "black",
    "#FF0000": "red",
    "#800000": "maroon",
    "#FFFF00": "yellow",
    "#808000": "olive",
    "#00FF00": "lime",
    "#008000": "green",
    "#00FFFF": "aqua",
    "#008080": "teal",
    "#0000FF": "blue",
    "#000080": "navy",
    "#FF00FF": "fuchsia",
    "#800080": "purple",
    # Add more colors as needed
}

def hex_to_color_name(hex_code):
    """
    Convert a HEX color code to its English color name.
    """
    hex_code = hex_code.upper()
    return HEX_TO_COLOR_NAME.get(hex_code, "Unknown color")

# Example usage
hex_code = "#00FF00"
color_name = hex_to_color_name(hex_code)
print(f"The color name for {hex_code} is {color_name}.")
# The color name for #00FF00 is lime.