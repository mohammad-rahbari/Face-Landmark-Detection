import os
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
class Sample_loader:
    def __init__(self):
        self.df = pd.read_csv("youtube_faces_with_keypoints_full.xls")
        self.file_lists = []
        self.file_lists.append(os.listdir("dataset/youtube_faces_with_keypoints_full_1"))
        self.file_lists.append(os.listdir("dataset/youtube_faces_with_keypoints_full_2"))
        self.file_lists.append(os.listdir("dataset/youtube_faces_with_keypoints_full_3"))
        self.file_lists.append(os.listdir("dataset/youtube_faces_with_keypoints_full_4"))
        self.sample_set = []

    def get_sample(self,n,max_framen,random_state=0):
        sample_set_info = self.df.sample(n,random_state=random_state)
        for i in range(len(sample_set_info)):
            sample = {}
            for j in range(1,5):
                path = f"dataset/youtube_faces_with_keypoints_full_{j}"
                if sample_set_info.iloc[i]["videoID"]+".npz" in os.listdir(path):
                    sample_npz = np.load(f"{path}/{sample_set_info.iloc[i]['videoID']}.npz")
                    break
            duration = min(max_framen, int(sample_set_info.iloc[i]["videoDuration"]))
            for frame_number in range(duration):
                sample["image"] = sample_npz["colorImages"][:, :, :, frame_number]
                sample["land_marks"] = sample_npz["landmarks2D"][:, :, frame_number]

                self.sample_set.append(sample)
            print(i)
        random.shuffle(self.sample_set)
    def sample_preview(self, nrows= 3, ncols= 5 ):
        step_size = len(self.sample_set) // ncols
        fig, axArray = plt.subplots(nrows=nrows,ncols=ncols)
        for i in range(ncols):
            for j in range(nrows):
                axArray[j][i].imshow(self.sample_set[i*step_size + j]["image"])
                axArray[j][i].scatter(x=self.sample_set[i*step_size + j]["land_marks"][:,0],y=self.sample_set[i*step_size + j]["land_marks"][:,1],s=1,c='g')
                axArray[j][i].set_title(f"sample N {i*step_size + j}", fontsize=8)
                axArray[j][i].set_axis_off()
        plt.show()
    def sample_resize(self , frame_size=200):

        for i in range(len(self.sample_set)):

            min_dim = min(self.sample_set[i]['image'].shape[:2])
            img_center =  np.array(self.sample_set[i]['image'].shape[:2]) //2
            self.sample_set[i]['image'] = self.sample_set[i]['image'][img_center[0] - min_dim//2: img_center[0] + min_dim//2,   img_center[1] - min_dim//2: img_center[1] + min_dim//2 , :]

            lm_tr = img_center - np.array(self.sample_set[i]['image'].shape[:2]) //2
            self.sample_set[i]['land_marks'][:,0] = self.sample_set[i]['land_marks'][:,0] -lm_tr[1]
            self.sample_set[i]['land_marks'][:,1] = self.sample_set[i]['land_marks'][:,1] -lm_tr[0]








            scale = [frame_size / self.sample_set[i]['image'].shape[0],frame_size / self.sample_set[i]['image'].shape[1]]
            self.sample_set[i]['image'] = cv2.resize(self.sample_set[i]['image'],(frame_size,frame_size))
            self.sample_set[i]['land_marks'][:,0] = self.sample_set[i]['land_marks'][:,0] * scale[1]
            self.sample_set[i]['land_marks'][:,1] = self.sample_set[i]['land_marks'][:,1] * scale[0]

    def grayscale(self):
        for i in range(len(self.sample_set)):
            try:
                self.sample_set[i]['image'] = cv2.cvtColor(self.sample_set[i]['image'], cv2.COLOR_BGR2GRAY)
            except:pass
            cv2.imshow("test", self.sample_set[i]['image'])
            cv2.waitKey(0)

test = Sample_loader()
test.get_sample(10,4)

test.sample_resize()
test.grayscale()
test.sample_preview(3,10)