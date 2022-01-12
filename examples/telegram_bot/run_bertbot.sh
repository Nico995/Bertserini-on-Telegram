#!/bin/sh
export TOKENIZERS_PARALLELISM=false
python bertbot.py --config ./bertbot_config.yaml
