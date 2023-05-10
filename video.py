import gc
import os
import torch
from torchinfo import summary
from super_gradients.training import models
import argparse

def detect_objects(video):
    device = torch.device("cpu")
    model = models.get("yolo_nas_s", pretrained_weights="coco").to(device)
    out = model.predict(video)

    out.save("predictions")

    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    os.makedirs("predictions", exist_ok=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required = True, help = "path to input video")
    args = vars(ap.parse_args())

    detect_objects(args["video"])
