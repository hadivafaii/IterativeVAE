#!/bin/bash


pattern="$(hostname)-cuda"
screens=$(screen -ls | grep "${pattern}" | cut -d. -f1 | awk '{print $1}')

if [ -n "${screens}" ]; then
    echo "${screens}" | xargs -r kill
    echo "Killed screens:"
    echo "${screens}"
else
    echo "No matching screens to kill."
fi
