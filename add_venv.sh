#!/usr/bin/env bash
VENVNAME=cv101
source $VENVNAME/bin/activate
python -m ipykernel install --user --name $VENVNAME --display-name "$VENVNAME"