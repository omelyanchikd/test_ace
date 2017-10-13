import ace
import json

with open('model_config.json', 'r') as f:
    model_config = json.load(f)

with open('run_config.json', 'r') as f:
    run_config = json.load(f)


from ace import run_ace
run_ace.run_ace(model_config, run_config)