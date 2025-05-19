#!/bin/bash


num_fits=${1}

# Find available GPUs by checking for existing fit files
function find_available_gpus {
  local gpus=()
  for file in "$(uname -n)"-cuda*-fit*.txt; do
    if [[ $file =~ "$(uname -n)"-cuda([0-9]+)-fit ]]; then
      gpus+=("${BASH_REMATCH[1]}")
    fi
  done
  # Return unique GPU numbers (properly quoted)
  printf "%s\n" "${gpus[@]}" | sort -u
}

# Find max index of fits for a specific GPU
function find_max_fit {
  local gpu=${1}
  local max_fit=-1
  local current_fit
  for file in "$(uname -n)"-cuda"${gpu}"-fit*.txt; do
    if [[ $file =~ "$(uname -n)"-cuda${gpu}-fit([0-9]+)\.txt ]]; then
      current_fit=${BASH_REMATCH[1]}
      (( current_fit > max_fit )) && max_fit=$current_fit
    fi
  done
  echo $(( max_fit + 1 ))  # fit indices start at 0
}

# Get list of available GPUs from existing files
mapfile -t available_gpus < <(find_available_gpus)

if [ ${#available_gpus[@]} -eq 0 ]; then
  echo "No fit files found. Exiting."
  exit 1
fi

# Main loop - only process GPUs that have fit files
for gpu in "${available_gpus[@]}"; do
  if [ -z "${num_fits}" ]; then
    num_fits=$(find_max_fit "${gpu}")
  fi
  for ((fit=0; fit<num_fits; fit++)); do
    name="cuda${gpu}-fit${fit}"
    name="$(uname -n)-${name}"
    if ! screen -list | grep -q "${name}"; then
      screen -dmS "${name}"
      echo "${name}: session created."
    else
      echo "Screen ${name} already exists —> executing."
    fi
    screen -S "${name}" -p 0 -X stuff "bash ${name}.txt\n"
  done
done
