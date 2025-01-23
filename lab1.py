import json
import numpy as np
import pandas as pd

file_path = "/home/kyumin/work/preprocessing/datasets/BF-PSR-Framework/JsonData/sample.txt"

df = pd.DataFrame()

with open(file_path) as json_file:
    conversations = json.load(json_file)
    messages = conversations["conversation"][0]["messages"]
    df = pd.DataFrame(messages)

print(df)
