import matplotlib.pyplot as plt
import numpy as np

'''Plot shapes and connects matching sides'''

def plot_polygon(ax, sides, radius=1, center=(0, 0), rotation=0):
    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False) + np.radians(rotation)
    points = np.c_[np.cos(angles), np.sin(angles)] * radius + np.array(center)
    points = np.vstack([points, points[0]])  # Close the polygon

    ax.plot(points[:, 0], points[:, 1], 'o-')

    # Label the sides
    for i in range(sides):
        mid_point = (points[i] + points[i + 1]) / 2
        ax.text(mid_point[0], mid_point[1], f'{i+1}', ha='center', va='center')

def main():
    fig, ax = plt.subplots()

    # Define the pieces
    # Piece 0: Square with 4 sides
    plot_polygon(ax, sides=4, radius=1, center=(0, 0), rotation=45)

    # Piece 1: Octagon with 8 sides
    plot_polygon(ax, sides=8, radius=1, center=(2.5, 0), rotation=22.5)

    # Draw a line connecting side 1 of Piece 0 and side 1 of Piece 1
    square_side_1_mid = (np.array([np.cos(np.radians(45)), np.sin(np.radians(45))]) +
                         np.array([np.cos(np.radians(135)), np.sin(np.radians(135))])) / 2
    octagon_side_1_mid = (np.array([2.5 + np.cos(np.radians(22.5)), np.sin(np.radians(22.5))]) +
                          np.array([2.5 + np.cos(np.radians(67.5)), np.sin(np.radians(67.5))])) / 2

    ax.plot([square_side_1_mid[0], octagon_side_1_mid[0]],
            [square_side_1_mid[1], octagon_side_1_mid[1]], 'k--')

    ax.set_aspect('equal')
    plt.axis('off')

    # Save the figure instead of showing it
    plt.savefig('matching_sides.png')

if __name__ == '__main__':
    main()
