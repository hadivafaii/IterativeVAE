#!/bin/bash

device=${1}
dataset=${2}
model=${3}

if [ -z "${device}" ]; then
  read -rp "enter device idx: " device
fi
if [ -z "${dataset}" ]; then
  read -rp "enter dataset: " dataset
fi
if [ -z "${model}" ]; then
  read -rp "enter model type: " model
fi

# Shift to remove the first three positional args
# then combine the remaining arguments into one
shift 3
args="${*}"

# Change to the parent directory of scripts/
cd "$(dirname "$0")/.." ||
{ echo "Failed to cd to parent dir."; exit 1; }

fit="python3 -m main.train \
  '${device}' \
  '${dataset}' \
  '${model}' \
   ${args}"
eval "${fit}"

printf '**************************************************************************\n'
printf "Done! —————— device = 'cuda:${device}' —————— (%s)\n" "$(date '+%m/%d/%Y %H:%M:%S')"
printf '**************************************************************************\n\n'