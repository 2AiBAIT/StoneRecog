import pandas as pd
import json

rocks_db = "rocks_db/rocks_db_corrected.json"
with open(rocks_db) as data_file:
    rocks = json.load(data_file)
df = pd.DataFrame(rocks)

classes_name = list(df.Classe.unique())
with open('stone_classes_list_L0.json', 'w') as fp:
    json.dump(classes_name, fp)

classes_num = list(range(0,len(classes_name)))

classes_dict = dict(zip(classes_name, classes_num))

print(classes_dict)
with open('stone_classes.json', 'w') as fp:
    json.dump(classes_dict, fp)

# with open('classes_file.txt', 'w') as f:
#     for item in classes:
#         f.write("%s\n" % item)
