# Copyright (c) NEC Laboratories America, Inc.
from detectron2.config import CfgNode as CN

def add_sas_det_config(cfg):
    _C = cfg

    ### configs from RegionCLIP
    ################### Text Tokenizer from MSR-CLIP ##################
    _C.INPUT.TEXT_TOKENIZER = "openai_bpe" # "bert-base-cased"
    ################## Data Augmentation from MSR-CLIP ##################
    _C.AUG = CN()
    _C.AUG.SCALE = (0.08, 1.0)
    _C.AUG.RATIO = (3.0/4.0, 4.0/3.0)
    _C.AUG.COLOR_JITTER = [0.4, 0.4, 0.4, 0.1, 0.0]
    _C.AUG.GRAY_SCALE = 0.0
    _C.AUG.GAUSSIAN_BLUR = 0.0
    _C.AUG.DROPBLOCK_LAYERS = [3, 4]
    _C.AUG.DROPBLOCK_KEEP_PROB = 1.0
    _C.AUG.DROPBLOCK_BLOCK_SIZE = 7
    _C.AUG.MIXUP_PROB = 0.0
    _C.AUG.MIXUP = 0.0
    _C.AUG.MIXCUT = 0.0
    _C.AUG.MIXCUT_MINMAX = []
    _C.AUG.MIXUP_SWITCH_PROB = 0.5
    _C.AUG.MIXUP_MODE = 'batch'
    _C.AUG.MIXCUT_AND_MIXUP = False
    _C.AUG.INTERPOLATION = 3
    _C.AUG.USE_TIMM = False
    _C.AUG.TIMM_AUG = CN(new_allowed=True)
    _C.AUG.TIMM_AUG.USE_LOADER = False
    _C.AUG.TIMM_AUG.USE_TRANSFORM = False
    _C.AUG.TRAIN = CN()
    _C.AUG.TRAIN.IMAGE_SIZE = [224, 224]  # width * height, ex: 192 * 256
    _C.AUG.TRAIN.MAX_SIZE = None  # the maximum size for longer edge after resizing
    _C.AUG.TEST = CN()
    _C.AUG.TEST.IMAGE_SIZE = [224, 224]  # width * height, ex: 192 * 256
    _C.AUG.TEST.MAX_SIZE = None   # the maximum size for longer edge after resizing
    _C.AUG.TEST.CENTER_CROP = False
    _C.AUG.TEST.INTERPOLATION = 3

    ################## Data Loading from MSR-CLIP ##################
    # List of dataset class names for training
    _C.DATASETS.FACTORY_TRAIN = ()
    # List of dataset folder for training
    _C.DATASETS.PATH_TRAIN = ()
    # List of the dataset names for auxilary training, as present in paths_catalog.py
    _C.DATASETS.AUX = ()
    # List of dataset class names for auxilary training
    _C.DATASETS.FACTORY_AUX = ()
    # List of dataset folder for auxilary training
    _C.DATASETS.PATH_AUX = ()
    # List of dataset class names for testing
    _C.DATASETS.FACTORY_TEST = ()
    # List of dataset folder for testing
    _C.DATASETS.PATH_TEST = ()
    # Labelmap file to convert to tsv or for demo purpose
    _C.DATASETS.LABELMAP_FILE = ''
    _C.DATASETS.ATTR_LABELMAP_FILE = ''
    _C.DATASETS.FILTERED_CLASSIFICATION_DATASETS = ''
    # hierarchy file for test time score aggregation (developed on OpenImages)
    _C.DATASETS.HIERARCHY_FILE = ''
    # List of box extra fields for training/testing
    # If given, will not infer from the other cfgs.
    _C.DATASETS.BOX_EXTRA_FIELDS = ()
    _C.DATASETS.NUM_CLASSES = 0
    _C.DATASETS.ROOT = ''
    _C.DATASETS.TRAIN_SET = 'train'
    _C.DATASETS.VAL_SET = ''
    _C.DATASETS.TEST_SET = 'val'
    # The maximum total input sequence length after WordPiece tokenization
    # Sequences longer than this will be truncated, and sequences shorter than this will be padded.
    _C.DATASETS.MAX_SEQ_LENGTH = 35  

    # ---------------------------------------------------------------------------- #
    # CLIP options
    # ---------------------------------------------------------------------------- #
    _C.MODEL.CLIP = CN()
    _C.MODEL.CLIP.CROP_REGION_TYPE = "" # options: "GT", "RPN"
    _C.MODEL.CLIP.BB_RPN_WEIGHTS = None # the weights of pretrained MaskRCNN
    _C.MODEL.CLIP.IMS_PER_BATCH_TEST = 8 # the #images during inference per batch
    _C.MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER = False # if True, use the CLIP text embedding as the classifier's weights
    _C.MODEL.CLIP.TEXT_EMB_PATH = None # "/mnt/output_storage/trained_models/lvis_cls_emb/lvis_1203_cls_emb.pth"
    _C.MODEL.CLIP.OFFLINE_RPN_CONFIG = None # option: all configs of pretrained RPN
    _C.MODEL.CLIP.NO_BOX_DELTA = False  # if True, during inference, no box delta will be applied to region proposals
    _C.MODEL.CLIP.BG_CLS_LOSS_WEIGHT = None # if not None, it is the loss weight for bg regions
    _C.MODEL.CLIP.ONLY_SAMPLE_FG_PROPOSALS = False  # if True, during training, ignore all bg proposals and only sample fg proposals
    _C.MODEL.CLIP.MULTIPLY_RPN_SCORE = False  # if True, during inference, multiply RPN scores with classification scores
    _C.MODEL.CLIP.VIS = False # if True, when visualizing the object scores, we convert them to the scores before multiplying RPN scores
    _C.MODEL.CLIP.OPENSET_TEST_NUM_CLASSES = None  # if an integer, it is #all_cls in test
    _C.MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH = None # if not None, enables the openset/zero-shot training, the category embeddings during test
    _C.MODEL.CLIP.CLSS_TEMP = 0.01 # normalization + dot product + temperature
    _C.MODEL.CLIP.RUN_CVPR_OVR = False # if True, train CVPR OVR model with their text embeddings
    _C.MODEL.CLIP.FOCAL_SCALED_LOSS = None # if not None (float value for gamma), apply focal loss scaling idea to standard cross-entropy loss
    _C.MODEL.CLIP.OFFLINE_RPN_NMS_THRESH = None # the threshold of NMS in offline RPN
    _C.MODEL.CLIP.OFFLINE_RPN_POST_NMS_TOPK_TEST = None # the number of region proposals from offline RPN
    _C.MODEL.CLIP.PRETRAIN_IMG_TXT_LEVEL = True # if True, pretrain model using image-text level matching
    _C.MODEL.CLIP.PRETRAIN_ONLY_EOT = False # if True, use end-of-token emb to match region features, in image-text level matching
    _C.MODEL.CLIP.PRETRAIN_RPN_REGIONS = None # if not None, the number of RPN regions per image during pretraining
    _C.MODEL.CLIP.PRETRAIN_SAMPLE_REGIONS = None # if not None, the number of regions per image during pretraining after sampling, to avoid overfitting
    _C.MODEL.CLIP.RANDOM_SAMPLE_REGION = False
    _C.MODEL.CLIP.GATHER_GPUS = False # if True, gather tensors across GPUS to increase batch size
    _C.MODEL.CLIP.GRID_REGIONS = False # if True, use grid boxes to extract grid features, instead of object proposals
    _C.MODEL.CLIP.CONCEPT_POOL_EMB = None # if not None, it provides the file path of embs of concept pool and thus enables region-concept matching
    _C.MODEL.CLIP.CONCEPT_THRES = None # if not None, the threshold to filter out the regions with low matching score with concept embs, dependent on temp (default: 0.01)
    _C.MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED = False # if True, use large-scale jittering (LSJ) pretrained RPN
    _C.MODEL.CLIP.TEACHER_RESNETS_DEPTH = 50 # the type of visual encoder of teacher model, sucha as ResNet 50, 101, 200 (a flag for 50x4)
    _C.MODEL.CLIP.TEACHER_CONCEPT_POOL_EMB = None # if not None, it uses the same concept embedding as student model; otherwise, uses a seperate embedding of teacher model
    _C.MODEL.CLIP.TEACHER_POOLER_RESOLUTION = 14 # RoIpooling resolution of teacher model
    _C.MODEL.CLIP.TEACHER_ROI_HEADS_NAME = None # by zsy, if used a different roi_heads for teacher model
    _C.MODEL.CLIP.TEXT_EMB_DIM = 1024 # the dimension of precomputed class embeddings
    _C.INPUT_DIR = "./datasets/custom_images" # the folder that includes the images for region feature extraction
    _C.MODEL.CLIP.GET_CONCEPT_EMB = False # if True (extract concept embedding), a language encoder will be created
    _C.MODEL.CLIP.FREEZE_BACKBONE =False    # freeze the whole backbone including attention pool

    # Use soft NMS instead of standard NMS if set to True
    _C.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False
    # See soft NMS paper for definition of these options
    _C.MODEL.ROI_HEADS.SOFT_NMS_METHOD = "gaussian" # "linear"
    _C.MODEL.ROI_HEADS.SOFT_NMS_SIGMA = 0.5
    # For the linear_threshold we use NMS_THRESH_TEST
    _C.MODEL.ROI_HEADS.SOFT_NMS_PRUNE = 0.001

    _C.SOLVER.WEIGHT_DECAY_BIAS = _C.SOLVER.WEIGHT_DECAY


    ### configs for OVD and PLs
    _C.WITH_IMAGE_LABELS = False # Turn on co-training with classification data

    _C.MODEL.WEAK_LOSS = CN()
    _C.MODEL.WEAK_LOSS.IMAGE_LOSS_WEIGHT = 0.1  # CLIP.PRETRAIN_IMG_TXT_LEVEL control if use ita loss
    
    # bipariti matching, weak alginment loss configs
    _C.MODEL.WEAK_LOSS.WEAK_LOSS_WEIGHT = -1.0  # vldet loss, weak alginment loss
    _C.MODEL.WEAK_LOSS.BOX_SELECT_THRES = 0.9
    _C.MODEL.WEAK_LOSS.WEAK_LOSS_TYPE = 'no_weak'
    _C.MODEL.WEAK_LOSS.NEG_CONCEPT_NUM = 10

    # regionclip loss configs
    _C.MODEL.WEAK_LOSS.PSEUDO_LOSS_WEIGHT = -1.0
    _C.MODEL.WEAK_LOSS.PSEUDO_USE_DISTILL = False
    _C.MODEL.WEAK_LOSS.PSEUDO_USE_CONTRASTIVE = False

    _C.MODEL.WEAK_LOSS.MOMENTUM = 0.999     # ema update for pretraining, not used for now

    _C.MODEL.OVD = CN()
    _C.MODEL.OVD.WITH_PSEUDO_LABELS = False
    _C.MODEL.OVD.EVAL_PSEUDO_LABELS = False
    _C.MODEL.OVD.PL_THRESHOLD = 0.9
    _C.MODEL.OVD.PL_NMS_THRES = 0.6
    _C.MODEL.OVD.EMA_MOMENTUM = -1.0    # if update the teacher model periodically 
    _C.MODEL.OVD.RPN_FUSION_METHOD = "regionclip"  # "avg_norm_scores", "avg_logits"
    _C.MODEL.OVD.CATEGORY_INFO = None  # for coco by default

    _C.MODEL.OVD.BOX_CONFIDENCE_THRES = -1.0 # box confidences scores for box regression with PLs

    _C.MODEL.OVD.USE_ADAPTIVE_THRES = False
    _C.MODEL.OVD.MIN_AVG_PLS = 1.0
    _C.MODEL.OVD.MAX_AVG_PLS = 3.0
    _C.MODEL.OVD.ADAPTIVE_THRES_DELTA = 0.05   # how many changes on PL_THRESHOLD each time when # PLs are out of range

    _C.MODEL.OVD.USE_PERIODIC_UPDATE = False
    _C.MODEL.OVD.PERIODIC_STEPS = (40000, 60000, 80000)

    _C.MODEL.OVD.USE_CONFIDENCE_WEIGHT = False # use confidence to weight loss

    # ensemble as ViLD, Detpro or F-VLM
    _C.MODEL.OVD.USE_ENSEMBLE_EVAL = False
    _C.MODEL.OVD.ENSEMBLE_ALPHA = 0.5


    ### configs for SAF head
    _C.MODEL.ENSEMBLE = CN()
    _C.MODEL.ENSEMBLE.TEST_CATEGORY_INFO = None

    # fusion weights of image/text head outputs for base/novel cats. See more details in F-VLM
    _C.MODEL.ENSEMBLE.ALPHA = 0.33   # weights of base cats for open-branch
    _C.MODEL.ENSEMBLE.BETA = 0.67   # weights of novel cats for open-branch

    _C.MODEL.ENSEMBLE.USE_IMG_HEAD_BOX_REG = False

    # _C.MODEL.ENSEMBLE.USE_OFFLINE_PL = False


