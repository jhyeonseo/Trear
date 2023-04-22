# making FPHA_val_list.txt
import os
val_list_path = "./annotations/FPHA_test.txt"
image_file_path = "./data/Video_files/"
save_file_path = "./annotations/FPHA_test_list.txt"
with open(val_list_path, 'r') as f:
    lines = f.readlines()
    new_list = []
    for line in lines:
        sequence = line.split(' ')[0]
        label = line.split(' ')[1].rstrip()
        location = image_file_path + sequence + '/color'
        img_names = os.listdir(location)
        img_names.sort()
        max_frame = len(img_names)
        new_list.append(' '.join([sequence, str(max_frame-1), label]) + '\n')
open(save_file_path, 'w').writelines(new_list)
