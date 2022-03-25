import os
import sys

filepath = "/home/gupta.anik/ScanNet/scans_uncomp/"
folderlist = ["color", "depth", "pose"]

if __name__=="__main_":
    for scene_folder in sorted(os.listdir(filepath)):
        scene_folder_path = filepath + scene_folder

        print(f"Working on {scene_folder}")
        for folder in folderlist:
            folder_path = scene_folder_path + "/" + folder

            count = 0
            for filename in sorted(os.listdir(folder_path)):
                if filename.startswith("frame"):
                    dst = folder_path + "/" + str(count) + filename[-4:]
                    src = folder_path + "/" + filename
                    count += 1
                    os.rename(src, dst)
