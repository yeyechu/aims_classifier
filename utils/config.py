# 데이터 경로
DATA_DIR = "./data"

RANDOM_SEED = 42

NUM_WORKERS = 4
NUM_CLASSES = 6

BATCH_SIZE = 4
EPOCHES = 40

#LEARNING_RATE = 0.0001

# Student용
LEARNING_RATE = 0.001

# 이미지 전처리 설정
IMAGE_SIZE = 512

TEMPERATURE = 3.0
ALPHA = 0.7

K_FOLDS = 5

LABELS = ["검정고시합격증명서", "국민체력100", "기초생활수급자증명서", "주민등록본", "체력평가", "생활기록부대체양식"]
THRESHOLD = 0.5
PATIENCE = 10