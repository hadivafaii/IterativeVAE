import sys
from tqdm import tqdm
import time

# Total iterations
epochs = 200

# Create tqdm progress bar
pbar = tqdm(
    total=epochs,
    position=0,
    leave=True,
    dynamic_ncols=True,
)

# Dummy loop to simulate work
for i in range(epochs):
    time.sleep(0.1)  # Simulate some work with sleep
    pbar.update(1)   # Update the progress bar by one step

pbar.close()  # Close the progress bar after loop completes
