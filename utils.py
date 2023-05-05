import os 
import yaml

def get_config(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    return config
