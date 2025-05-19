#!/bin/bash


# Define the screen session name
screen_name="tqdm_test_session"

# Create a new detached screen session
screen -dmS "${screen_name}"

# Define the Python script to run
root="Dropbox/git/_PoissonVAE/scripts"
py_script="test_tqdm.py"

# Execute the Python script inside the screen session
cmd="python3 ${HOME}/${root}/${py_script}"
screen -S "${screen_name}" -X stuff "${cmd}\n"

echo "Python script '${py_script}' running inside session '${screen_name}'."
