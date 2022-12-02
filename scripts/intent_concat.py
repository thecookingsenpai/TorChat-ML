import os
import json

final_intents = { "intents": [] }

os.chdir(os.path.dirname(__file__))

for root, dirs, files in os.walk("../multintents/"):
    for file in files:
        if file.endswith(".json"):
            print(os.path.join(root, file))
            with open(os.path.join(root, file), 'r') as f:
                intents = json.load(f)
                print(intents)
                final_intents["intents"].extend(intents["intents"])

with open("../training/intents.json", "w+") as fintents:
    json.dump(final_intents, fintents)