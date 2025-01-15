import pickle
import numpy as np
from torch.utils.data import Dataset


class audio_skeleton_dataset(Dataset):
    def __init__(self, cfg, split="train", train_model="MotionVQVAE"):
        path = cfg["path"]
        self.aggr_length = cfg["aggr_length"]
        self.train_model = train_model
        with open(path, "rb") as f:
            data = pickle.load(f)[split]
        
        if train_model == "MotionVQVAE":
            self.keypoints = data["keypoints"]
        elif train_model == "AudioEnc":
            self.aud = data["aud"]
            self.keypoints = data["keypoints"]
            
        self.len = len(data["aud"])
        print(self.len)
        del data
        
    def audio_aggregate(self, aud, aggr_length=5):
        pad = (aggr_length-1) // 2
        aud = np.pad(aud, ((pad,pad),(0,0)))
        aud_expand = []
        for i in range(pad, len(aud)-pad):
            temp = aud[i-pad:i+(pad+1)]
            aud_expand.append(temp)
        aud_expand = np.array(aud_expand)
        return aud_expand
    
    def __getitem__(self, index):
        result = dict()
        if self.train_model == "MotionVQVAE":
            keypoints = self.keypoints[index]
            result["keypoints"] = keypoints
        
        elif self.train_model == "AudioEnc":
            aud = self.aud[index]
            keypoints = self.keypoints[index]
            if self.aggr_length != None:
                result["ori_aud"] = aud
                aud = self.audio_aggregate(aud, aggr_length=self.aggr_length)
            result["aud"] = aud
            result["keypoints"] = keypoints
        return result

    def __len__(self):
        return self.len