import json, os

file = '/home/rsulzer/python/HiSup/data/inria/train/annotation.json'

# Open and read the JSON file
with open(file, 'r') as file:
    data = json.load(file)

# Print the data
print(data)