import tensorflow as tf
import wandb #실험 관리 툴입니다.

# Unknown: Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above.
# 위 error 처리 코드
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import tensorflow as tf
import keras

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

#Mask R-CNN으로 coco datatset을 학습한 모델의 가중치를 가져와 학습하였습니다.
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_apple_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

#detection 대상 클래스입니다.
class_names = ['Bitter rot', 'brown rot', 'Mar blotch', 'White rot', 'Normal', 'Sooty/Fly']
class_dict = {class_: id_ for id_, class_ in enumerate(class_names, 1)}

############################################################
#  Configurations: 모델의 하이퍼파라미터 세부 조정
############################################################
targets = {}
cur_epochs = 500  # epoch 여기서 지정
layer_stage = "heads"  #레이어 개수 선정: heads, 3+, 4+, 5+, all

# Wandb: 실험 관리 툴입니다.
#사전에 만들어놓은 wandb의 project에 해당 실험을 저장하기 위한 초기화작업입니다.
run = wandb.init(project=[project name],name=f"{datetime.datetime.now()}")
class AppleConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "apple"

    # Train on 1 GPU and 1 images per GPU. We can put multiple images on each
    # GPU. Batch size is (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = len(class_names) + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 50

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9  # default: 0.9

    # set validation steps
    VALIDATION_STEPS = 58  # default: 50

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512  # default:800
    IMAGE_MAX_DIM = 1024  # default:1024

    IMAGE_MIN_SCALE = 0

    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # default: (32, 64, 128, 256, 512)

    TRAIN_ROIS_PER_IMAGE = 200  # default: 200

    TRAIN_BN = False  # default: False

    RPN_NMS_THRESHOLD = 0.7

    # Backbone (default=resnet101)
    BACKBONE = "resnet50"

    # optimizer
    OPTIMIZER = 'SGD'

    # Learning rate
    LEARNING_RATE = 0.001

    WEIGHT_DECAY = 0.0001

    GRADIENT_CLIP_NORM = 5.0  # default: 5.0

    FPN_CLASSIF_FC_LAYERS_SIZE = 1024  # default:1024

    PRE_NMS_LIMIT = 6000

    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33  # default:0.33
    LEARNING_MOMENTUM = 0.9  # default 0.9

    # Wandb -> 각 실험 모델의 변경사항을 비교할 하이퍼파라미터
    targets = {"OPTIMIZER": OPTIMIZER,
               "LEARNING_RATE": LEARNING_RATE,
               "RPN_NMS_THRESHOLD": RPN_NMS_THRESHOLD,
               "BACKBONE": BACKBONE,
               "TRAIN_ROIS_PER_IMAGE": TRAIN_ROIS_PER_IMAGE,
               "TRAIN_BN": TRAIN_BN,
               "IMAGE_MIN_DIM": IMAGE_MIN_DIM,
               "IMAGE_MAX_DIM": IMAGE_MAX_DIM,
               "VALIDATION_STEPS": VALIDATION_STEPS,
               "STEPS_PER_EPOCH": STEPS_PER_EPOCH,
               "DETECTION_MIN_CONFIDENCE": DETECTION_MIN_CONFIDENCE,
               "ROI_POSITIVE_RATIO": ROI_POSITIVE_RATIO,
               "WEIGHT_DECAY": WEIGHT_DECAY,
               "IMAGE_RESIZE_MODE": IMAGE_RESIZE_MODE,
               "LAYER_STAGE": layer_stage,
               "EPOCHS": cur_epochs,"BATCH_SIZE":GPU_COUNT*IMAGES_PER_GPU,
              "LEARNING_MOMENTUM": LEARNING_MOMENTUM,
              "GRADIENT_CLIP_NORM":GRADIENT_CLIP_NORM}
    run.config.update(targets)


#wandb에 로그를 남기기 위한 callback함수입니다.
class PerformanceCallback(keras.callbacks.Callback):
    def __init__(self, run):
        self.run = run
    def on_epoch_end(self, epoch, logs):
        print("Uploading metrics to wandb...")
        self.run.log(logs)

############################################################
#  Dataset: 데이터와 json파일(어노테이션 정보)을 로드합니다.
############################################################

class AppleDataset(utils.Dataset):

    def load_apple(self, dataset_dir, subset):
        # Add classes. We have only one class to add.

        for e, target in enumerate(class_names, 1):
            self.add_class("apple", e, target)  # ---변경

        # Train or validation dataset?

        assert subset in ['train', 'val']
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())
        annotations = [a for a in annotations if a['regions']]
        # Add images
        for e, a in enumerate(annotations):
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            num_ids = []
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
                # print("regions:",a['regions']), augmentation = augmentation
                try:
                    objects = [s['region_attributes']['apple'] for s in a['regions'].values()]
                except:
                    print(objects)
                    continue

            else:
                polygons = [r['shape_attributes'] for r in a['regions']]
                try:
                    objects = [s['region_attributes']['apple'] for s in a['regions']][0]
                except:
                    continue

            num_ids = [class_dict[name_] for name_ in objects]
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path, plugin='matplotlib')
            height, width = image.shape[:2]

            self.add_image(
                "apple",
                image_id=a['filename'],
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "apple":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        info = self.image_info[image_id]
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        num_ids = np.array(num_ids, dtype=np.int32)

        return mask, num_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "apple":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = AppleDataset()
    dataset_train.load_apple(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = AppleDataset()
    dataset_val.load_apple(args.dataset, "val")
    dataset_val.prepare()

    # callback함수 추가
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                  patience=15, min_lr=0.0001)
    callbacks = [PerformanceCallback(run), reduce_lr]

    ap_config = AppleConfig()
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=ap_config.LEARNING_RATE,
                epochs=cur_epochs,
                layers=layer_stage, custom_callbacks=callbacks)  # 앞쪽레이어만 적용


def color_splash(image, mask):
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training:
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect apple')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/train/",
                        help='Directory of the Apple dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video, \
            "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = AppleConfig()
    else:
        class InferenceConfig(AppleConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
