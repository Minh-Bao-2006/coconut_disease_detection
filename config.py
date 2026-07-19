RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
FEATURES_DIR  = "features"
CHECKPOINTS   = "checkpoints"
RESULTS_DIR   = "results"
RANDOM_SEED   = 42

CLASSES = [
    "Bud_Root_Dropping",
    "Bud_Rot",
    "Gray_Leaf_Spot",
    "Leaf_Rot",
    "Stem_Bleeding",
]

BACKBONE   = "resnet18"   
IMG_SIZE   = 224
BATCH_SIZE = 64

SVM_C      = 10           
SVM_GAMMA  = "scale"      
SVM_KERNEL = "rbf"
PCA_DIMS   = 256          

RF_N_TREES   = 500
RF_MAX_DEPTH = 25

GRID_CV_FOLDS  = 5
GRID_SVM_C     = [1, 5, 10, 50]
GRID_SVM_GAMMA = ["scale", "auto"]  