#!/bin/bash

git add --all
commit_string=$1
git commit --message $commit_string 
echo "Committed the changes with this commit string -> $commit_string! "
git push
