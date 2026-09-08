#!/bin/bash

cd "$(dirname "$0")"

old_account=$(gh api user --jq .login)
gh auth switch --user CS26M214

if [ "$(gh api user --jq .login)" != "CS26M214" ]; then
    echo "GitHub account is not CS26M214"
    exit 1
fi

git push -u origin main
result=$?

if [ "$old_account" != "CS26M214" ]; then
    gh auth switch --user "$old_account"
fi

exit $result
