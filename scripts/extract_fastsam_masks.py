from argparse import ArgumentParser
import os
import numpy as np
import cv2
import torch

from ultralytics import FastSAM

def main(args):
    os.makedirs(args.output_dir+"masks",exist_ok=True)
    bgr_img = cv2.imread(args.img_filepath)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = '/FastSAM.pt'
    fastsam_model = FastSAM(model_path)

    everything_results = fastsam_model(
        bgr_img,
        device=device,
        retina_masks=True,
        imgsz=bgr_img.shape[:2],
        conf=0.5,
        iou=0.9,
    )

    masks = everything_results[0].masks.data
    if len(masks) == 0:
        print(f"Masks: {masks}")
        print("NO MASKS FOUND")
        exit()
    cpu_masks = masks.cpu().numpy().astype(bool)

    for mask_idx in range(cpu_masks.shape[0]):
        mask = (cpu_masks[mask_idx,:,:] * 255).astype(np.uint8)
        # cv2.imshow("Mask",mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(args.output_dir+f"masks/mask{mask_idx}.png", mask)

def argparser():
    parser = ArgumentParser()
    parser.add_argument('--img_filepath',type=str,help='Filepath to image for masking')
    parser.add_argument('--output_dir',type=str,help='/my/directory/ Directory to save mask folder and masks')
    return parser.parse_args()

if __name__ == '__main__':
    args = argparser()
    main(args)
