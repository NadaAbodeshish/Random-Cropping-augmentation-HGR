import numpy as np
import json
import os

# Define paths
root_database_path = '/kaggle/input/hand-gesture-sh/shrec17_dataset/HandGestureDataset_SHREC2017'
test_txt_path = '/kaggle/input/hand-gesture-sh/shrec17_dataset/HandGestureDataset_SHREC2017/test_gestures.txt'

# Create output directories
output_dir = "./shrec17_jsons/test_jsons/"
os.makedirs(output_dir, exist_ok=True)

# Load test gestures
test_txt = np.loadtxt(test_txt_path, dtype=int)


Samples_sum = test_txt.shape[0]

data_dict = []

# Process each sample
for i in range(Samples_sum): 
    idx_gesture = test_txt[i][0]  # gesture
    idx_finger = test_txt[i][1]  # finger
    idx_subject = test_txt[i][2]  # subject
    idx_essai = test_txt[i][3]    # essai
    label_14 = test_txt[i][4]     # label_14
    label_28 = test_txt[i][5]     # label_28
    T = test_txt[i][6]            # frames

    skeleton_path = root_database_path + '/gesture_' + str(idx_gesture) + '/finger_' \
                    + str(idx_finger) + '/subject_' + str(idx_subject) + '/essai_' + str(idx_essai) + '/skeletons_world.txt'

    if not os.path.exists(skeleton_path):
        print(f"File not found: {skeleton_path}")
        continue

    # Load and reshape skeleton data
    skeleton_data = np.loadtxt(skeleton_path)
    skeleton_data = skeleton_data.reshape([T, 22, 3])

    # Generate file name
    file_name = "g" + str(idx_gesture).zfill(2) + "f" + str(idx_finger).zfill(2) + "s" + str(idx_subject).zfill(2) + "e" + str(idx_essai).zfill(2)

    # Save skeleton data to JSON
    data_json = {
        "file_name": file_name,
        "skeletons": skeleton_data.tolist(),
        "label_14": label_14.tolist(),
        "label_28": label_28.tolist()
    }
    with open(output_dir + file_name + ".json", 'w') as f:
        json.dump(data_json, f)

    # Append metadata to data_dict
    tmp_data_dict = {
        "file_name": file_name,
        "length": T.tolist(),
        "label_14": label_14.tolist(),
        "label_28": label_28.tolist()
    }
    data_dict.append(tmp_data_dict)

# Save metadata to JSON
with open("./shrec17_jsons/" + "test_samples.json", 'w') as t:
    json.dump(data_dict, t)
