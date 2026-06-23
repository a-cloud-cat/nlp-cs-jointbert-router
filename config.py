import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

INTENT_LABELS = ['退货申请', '物流查询', '改地址', '开发票', '领优惠券']

SLOT_LABELS = ['O', 'B-订单号', 'I-订单号', 'B-手机号', 'I-手机号', 
               'B-商品名', 'I-商品名', 'B-收货地址', 'I-收货地址']

BERT_MODEL_NAME = os.path.join(MODEL_DIR, 'bert-base-chinese')

MAX_SEQ_LEN = 64
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 10
EARLY_STOP_PATIENCE = 3
DROPOUT_RATE = 0.3

NUM_INTENT_CLASSES = len(INTENT_LABELS)
NUM_SLOT_CLASSES = len(SLOT_LABELS)

INTENT_LABEL_MAP = {label: idx for idx, label in enumerate(INTENT_LABELS)}
SLOT_LABEL_MAP = {label: idx for idx, label in enumerate(SLOT_LABELS)}
SLOT_IDX_MAP = {idx: label for label, idx in SLOT_LABEL_MAP.items()}
