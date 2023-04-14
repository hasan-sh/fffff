zimport utils
import json

# Open a file for reading
with open("../data/labels_per_category.json", "r") as f:
    # Write the dictionary to the file in JSON format
    ocms = json.load(f)


with open("../data/id_to_label.json", "r") as f:
    # Write the dictionary to the file in JSON format
    id_to_label = json.load(f)

id_to_label = {int(i): l for i, l in id_to_label.items()}
label_to_id = {v: k for k, v in id_to_label.items()}


id_to_category = utils.make_id_category(ocms, label_to_id)


valid_ocms = list(id_to_label.keys())

