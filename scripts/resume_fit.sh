#!/bin/bash

device=${1}
model_name=${2}
fit_name=${3}

if [ -z "${device}" ]; then
  read -rp "enter device idx: " device
fi
if [ -z "${model_name}" ]; then
  read -rp "enter model name: " model_name
fi
if [ -z "${fit_name}" ]; then
  read -rp "enter fit name: " fit_name
fi

# Shift to remove the first two positional arguments
# then combine the remaining arguments into one
shift 3
args="${*}"

root="Dropbox/git/_IterativeVAE"
root="${HOME}/${root}"
cd "${root}" || exit

cmd="python3 -m main.resume_train \
  '${device}' \
  '${model_name}' \
  '${fit_name}' \
   ${args}"
eval "${cmd}"

printf '**************************************************************************\n'
printf "Done! —————— device = 'cuda:${device}' —————— (%s)\n" "$(date '+%m/%d/%Y %H:%M:%S')"
printf '**************************************************************************\n\n'