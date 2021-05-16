import pandas as pd
import json



def class2name():

    data_path = './data/ESC-50/meta/esc50.csv'
    df_audio = pd.read_csv(data_path)
    map_dict = {}
    for idx, row in df_audio.iterrows():
        map_dict[str(row['target'])] = [str(row['target']), str(row['category'])]

    return map_dict



if __name__ == "__main__":
	sample = class2name()
	with open('index_to_name.json', 'w') as fp:
	    json.dump(sample, fp)