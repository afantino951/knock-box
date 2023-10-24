import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(6, 6))

# Define the image to insert
door_closed = mpimg.imread('door_closed.jpg')
door_open = mpimg.imread('door_open.jpg')

# Define a function to display or update the graphic with an image
def display_graphic(interaction, sequence_index, correct_sequence_len, reset_quiet):
    ax.clear()  # Clear the previous plot
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Set the positions for text and image
    text_y = 0.95
    image_y = 0.1
    
    ax.text(0.5, text_y, f"Interaction Detected: {interaction}", fontsize=16, ha="center")
    ax.text(0.5, text_y - 0.05, f"Correct Count: {sequence_index}/{correct_sequence_len}", ha="center")
    
    if reset_quiet:
        ax.text(0.5, text_y - 0.1, "Reset, quiet too long", ha="center")
    
    if (sequence_index == correct_sequence_len):
        ax.text(0.5, text_y - 0.11, "Success! Door opened!", ha="center")
        ax.imshow(door_open, extent=[0.2, 0.8, image_y, image_y + 0.7])  # Insert the image
    else:
        ax.imshow(door_closed, extent=[0.2, 0.8, image_y, image_y + 0.7])  # Insert the image
    
    ax.axis("off")
    plt.pause(0.1)  # Pause for a short duration to update the plot

display_graphic("knock", 1, 4, False)
time.sleep(2)
display_graphic("palm", 1, 4, True)
time.sleep(2)
display_graphic("knock", 1, 4, False)
time.sleep(2)
display_graphic("palm", 4, 4, False)

input("Press Enter to close the plot...")
