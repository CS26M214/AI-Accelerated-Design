#!/bin/bash

cd "$(dirname "$0")"

gh auth switch --user CS26M214

if [ "$(gh api user --jq .login)" != "CS26M214" ]; then
    echo "GitHub account is not CS26M214"
    exit 1
fi

git push -u origin main
