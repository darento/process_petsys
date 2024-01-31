from collections import defaultdict
from matplotlib import pyplot as plt


def plot_chan_position(dict_coords: dict) -> None:
    """
    This function plots the position of the channels on a graph.

    Parameters:
    dict_coords (dict): A dictionary mapping channels to their positions.
                        The keys are the channel names and the values are lists of coordinates [x, y].

    Output:
    None. This function plots a graph and does not return anything.
    """
    # Initialize plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create a dictionary to store the count of channels at each position
    position_counts = defaultdict(int)

    # Plot each channel
    for ch, xy in dict_coords.items():
        x, y = xy
        ax.scatter(x, y, label=f"Channel {ch}")

        # Increment the count
        position_counts[xy] += 1

        # Adjust the text position based on the count
        offset = 0.35 * position_counts[xy]
        ax.text(x, y + offset, str(ch), va="center", ha="center")

    # Similarly plot for J2 and any other ports if needed

    # Set plot properties
    ax.set_xlabel("X position (mm)")
    ax.set_ylabel("Y position (mm)")
    ax.set_title("Floodmap Representation of Channels")
    ax.axis("equal")  # To maintain the aspect ratio
    plt.show()
