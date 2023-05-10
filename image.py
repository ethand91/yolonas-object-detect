import gc
import os
import torch
from torchinfo import summary
from super_gradients.training import models
import argparse

def detect_objects(image):
    device = torch.device("cpu")
    model = models.get("yolo_nas_s", pretrained_weights="coco").to(device)
    out = model.predict(image, conf=0.6)

    out.save("predictions");

    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    os.makedirs("predictions", exist_ok=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Path to input file")
    args = vars(ap.parse_args())

    detect_objects(args["image"])
