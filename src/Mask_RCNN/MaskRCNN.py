import os
import sys
import cv2
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

from PIL import Image

class Semantic(object):
    def __init__(self):
        print ('Initializing Mask RCNN network...')
    
        # Root directory of the project
        ROOT_DIR = os.path.abspath("/mnt/lySLAM/src/Mask_RCNN/")

        # Import Mask RCNN
        sys.path.append(ROOT_DIR)  # To find local version of the library
        from mrcnn import utils
        import mrcnn.model as modellib
        from mrcnn import visualize
        # Import COCO config
        sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
        import coco
        
        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")

        # Local path to trained weights file
        COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco_0160.h5")

        # Download COCO trained weights from Releases if needed
        if not os.path.exists(COCO_MODEL_PATH):
            utils.download_trained_weights(COCO_MODEL_PATH)

        # Directory of images to run detection on
        IMAGE_DIR = os.path.join(ROOT_DIR, "images")

        class InferenceConfig(coco.CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()

        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

        # Load weights trained on MS-COCO
        self.model.load_weights(COCO_MODEL_PATH, by_name=True)

        self.class_names = ['BG', 'person', 'chair', 'book']
        

    def semanticseg(self, image, name):

        # Run detection
        results = self.model.detect([image], verbose=0)
        
        r = results[0]
        i = 0

        h = image.shape[0]
        w = image.shape[1]
        mask = np.zeros((h,w))
        for roi in r['rois']:
            if self.class_names[r['class_ids'][i]] == 'person' and r['scores'][i]>0.98:
                image_m = r['masks'][:,:,i]
                mask[image_m == 1] = 1.
            i += 1

        kernel = np.ones((19, 19), np.uint8)
        mask = cv2.dilate(mask, kernel)
        
        cv2.imwrite("/mnt/lySLAM/src/Mask_RCNN/a_masks/%s" % name, mask)
        
        mask *= 255  # 变换为0-255的灰度值

        m = Image.fromarray(mask)
        m = m.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
        m.save("/mnt/lySLAM/src/Mask_RCNN/results/%s" % name)
        
        return m

def main():
    seg = Semantic()
    
    file_path = '/mnt/rgbd_dataset_freiburg3_sitting_static/rgb.txt'
    base_path = "/mnt/rgbd_dataset_freiburg3_sitting_static/"
    
    f = open(file_path, 'r')
    lines = f.readlines()
    lines = lines[3::]
    
    j = 0
    for line in lines:
        line = line.strip('\n')
        line = line[18::]
        rgb_path = base_path + line
        name = line[4::]
        
        print(name)
        img = skimage.io.imread(rgb_path)
        seg.semanticseg(img, name)
        j += 1

if __name__ == "__main__":
    main()

    
    
    
    
    
    # img = skimage.io.imread("/mnt/lySLAM/src/Mask_RCNN/images/1341846647.834093.png")
    #   /mnt/rgbd_dataset_freiburg3_walking_xyz/rgb

    # semanticseg(img)
    
# Visualize results
# r1 = results1[0]
# r2 = results2[0]
# visualize.display_instances(image, r1['rois'], r1['masks'], r1['class_ids'], 
#                             class_names, r1['scores'])
# visualize.display_instances(image2, r2['rois'], r2['masks'], r2['class_ids'], 
#                             class_names, r2['scores'])