import os
import glob
import random
import itertools

def generate_AdobeFiveK_split_files(root, train_val_ratio=0.02):
    raw_files = glob.glob(os.path.join(root, "raw_photos/*/photos/*.dng"))
    random.shuffle(raw_files)
    sample_num = len(raw_files)
    with open(os.path.join(root, "train.txt"),"w") as f:
        for fpath in raw_files[int(sample_num*train_val_ratio):]:
            f.write(fpath +"\r")
    with open(os.path.join(root, "val.txt"),"w") as f:
        for fpath in raw_files[:int(sample_num*train_val_ratio)]:
            f.write(fpath +"\r")

def generate_SID_split_files(root, train_val_ratio=0.02):
    mappings = {
        "Sony_train_list.txt": "train.txt",
        "Sony_test_list.txt": "val.txt"
    }
    for key, val in mappings.items():
        with open(os.path.join(root, val),"w") as target:
            with open(os.path.join(root, key), "r") as source:
                for line in source.readlines():
                    info_list = line.split(" ")
                    target.write(os.path.join(root, info_list[0][2:]+"\n"))
                    target.write(os.path.join(root, info_list[1][2:]+"\n"))

                    
def generate_NUS_cross_val_split_files(root, split_num=5):
    raw_files = sorted(glob.glob(os.path.join(root, "RAW/*.*")))
    raw_processed_files = sorted(glob.glob(os.path.join(root, "Raw_Processed/*.*")))
    jpg_files = sorted(glob.glob(os.path.join(root, "JPG/*.*")))
    paired_files = list(zip(raw_files, raw_processed_files, jpg_files))
    random.shuffle(paired_files)
    sample_num = len(raw_files)
    train_val_ratio = 1/split_num
    for i in range(split_num):
        train_files = paired_files[:int(sample_num*train_val_ratio*i)]+paired_files[int(sample_num*train_val_ratio*(i+1)):]
        val_files = paired_files[int(sample_num*train_val_ratio*i):int(sample_num*train_val_ratio*(i+1))]
        with open(os.path.join(root, f"train_nocompress_f{i}.txt"),"w") as f:
            for raw_path, raw_processed_path, jpg_path in train_files:
                f.write(f"{raw_path}\r")
        with open(os.path.join(root, f"train_jpg_f{i}.txt"),"w") as f:
            for raw_path, raw_processed_path, jpg_path in train_files:
                f.write(f"{jpg_path}, {raw_processed_path}\r")
        
        # with open(os.path.join(root, "val.txt"),"w") as f:
        #     for raw_path, raw_processed_path, jpg_path in val_files:
        #         f.write(f"{jpg_path}, {raw_path}\r")
        #         f.write(f"{raw_path}\r")
        for device, qual in itertools.product(["Samsung", "Olympus", "Sony"], ["", "_jpg"]):
            with open(os.path.join(root, f"val_{device}{qual}_f{i}.txt"),"w") as f:
                test_pairs = paired_files[int(sample_num*train_val_ratio*i):int(sample_num*train_val_ratio*(i+1))]
                filter_device_pairs = list(filter(lambda x: device in x[0], test_pairs))
                for raw_path, raw_processed_path, jpg_path in filter_device_pairs:
                    if qual == "":
                        f.write(raw_path +"\r")
                    else:
                        f.write(f"{jpg_path}, {raw_processed_path}\r")
                  
def generate_NUS_split_files(root, train_val_ratio=0.02):
    raw_files = sorted(glob.glob(os.path.join(root, "RAW/*.*")))
    jpg_files = sorted(glob.glob(os.path.join(root, "JPG/*.*")))
    paired_files = list(zip(raw_files, jpg_files))
    random.shuffle(paired_files)
    sample_num = len(raw_files)

    with open(os.path.join(root, "train.txt"),"w") as f:
        for raw_path, jpg_path in paired_files[int(sample_num*train_val_ratio):]:
            f.write(f"{jpg_path}, {raw_path}\r")
            f.write(f"{raw_path}\r")
    with open(os.path.join(root, "val.txt"),"w") as f:
        for raw_path, jpg_path in paired_files[:int(sample_num*train_val_ratio)]:
            f.write(f"{jpg_path}, {raw_path}\r")
            f.write(f"{raw_path}\r")
    for device, qual in itertools.product(["Samsung", "Olympus", "Sony"], ["", "_jpg"]):
        with open(os.path.join(root, f"val_{device}{qual}.txt"),"w") as f:
            test_pairs = paired_files[:int(sample_num*train_val_ratio)]
            filter_device_pairs = list(filter(lambda x: device in x[0], test_pairs))
            for raw_path, jpg_path in filter_device_pairs:
                if qual == "":
                    f.write(raw_path +"\r")
                else:
                    f.write(f"{jpg_path}, {raw_path}\r")
if __name__ == "__main__":
    # generate_AdobeFiveK_split_files("/home/data/Dataset/fivek_dataset")
    # generate_SID_split_files("/home/data/Dataset/SID")
    generate_NUS_cross_val_split_files("./datasets/NUS", split_num=5)