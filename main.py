import json
import numpy as np

def file_open(file_path):
    with open(file_path) as json_file:
        conversations = json.load(json_file)
        for p in conversations['conversation']:
            m_id = 1
            print('Source: ' + p['source'])
            print('Label: ' + p['label'])
            for message in p['messages']:
                print(' Message: ',m_id)
                print("     Author: " + message['author'])
                print("     Text: " + message['text'])
                print("     Time: " + message['time'])
                m_id += 1
            print('*'*20)



if __name__ == "__main__":
    file_path = "/home/kyumin/work/preprocessing/datasets/BF-PSR-Framework/JsonData/sample.txt"
    file_open(file_path)