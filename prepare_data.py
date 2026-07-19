import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

CLASS_NAMES = [
    "Bud Root Dropping",
    "Bud Rot",
    "Gray Leaf Spot",
    "Leaf Rot",
    "Stem Bleeding",
]
NUM_CLASSES = len(CLASS_NAMES)

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

MODEL_NAME = "efficientnet_b3"  
PRETRAINED  = True
IMAGE_SIZE  = 300                
DROPOUT     = 0.4

DEVICE       = "cuda"           
BATCH_SIZE   = 32
NUM_EPOCHS   = 40
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 8                 

T_MAX        = NUM_EPOCHS

LABEL_SMOOTHING = 0.1
MIXUP_ALPHA     = 0.4

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]