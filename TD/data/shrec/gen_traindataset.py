import numpy as np
import json
import os

root_database_path = '/kaggle/input/hand-gesture-sh/HandGestureDataset_SHREC2017'
train_txt_path = '/kaggle/input/hand-gesture-sh/HandGestureDataset_SHREC2017/train_gestures.txt'

# Create output directories
output_dir = "./shrec17_jsons/train_jsons/"
os.makedirs(output_dir, exist_ok=True)

train_txt = np.loadtxt(train_txt_path, dtype=int)

Samples_sum = train_txt.shape[0]

data_dict = []

for i in range(Samples_sum): 
    idx_gesture = train_txt[i][0]
    idx_finger = train_txt[i][1]
    idx_subject = train_txt[i][2]
    idx_essai = train_txt[i][3]
    label_14 = train_txt[i][4]
    label_28 = train_txt[i][5]
    T = train_txt[i][6]

    skeleton_path = root_database_path + '/gesture_' + str(idx_gesture) + '/finger_' \
                    + str(idx_finger) + '/subject_' + str(idx_subject) + '/essai_' + str(idx_essai)+'/skeletons_world.txt'

    if not os.path.exists(skeleton_path):
        print(f"File not found: {skeleton_path}")
        continue

    skeleton_data = np.loadtxt(skeleton_path)
    skeleton_data = skeleton_data.reshape([T, 22, 3])

    file_name = "g"+str(idx_gesture).zfill(2) + "f"+str(idx_finger).zfill(2) + "s"+str(idx_subject).zfill(2) + "e"+str(idx_essai).zfill(2)

    data_json = {"file_name": file_name, "skeletons": skeleton_data.tolist(), "label_14": label_14.tolist(), "label_28": label_28.tolist()}
    with open(output_dir + file_name + ".json", 'w') as f:
        json.dump(data_json, f)

    tmp_data_dict = {"file_name": file_name, "length": T.tolist(), "label_14": label_14.tolist(), "label_28": label_28.tolist()}
    data_dict.append(tmp_data_dict)

with open("./shrec17_jsons/" + "train_samples.json", 'w') as t:
    json.dump(data_dict, t)
