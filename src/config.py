import transformers
class config:
    MAX_LEN = 128
    OVERLAP = 126
    TRAIN_BATCH_SIZE = 64
    DEV_BATCH_SIZE = 64
    GPU_DEVICE = 'cuda:0'#'cpu'
    FREEZE_EPOCHS = 10
    FREEZE_LEARNING_RATE = 5e-5
    UNFREEZE_EPOCHS = 10
    UNFROZEN_LAYERS = 6
    UNFREEZE_LEARNING_RATE = 8e-6
    BASE_MODEL_PATH = 'distilbert-base-uncased'
    DATA_DIR = '/home/nxingyu/project/data/'
    TRAIN_DATASET = DATA_DIR+'ted_talks_processed.dev.pt'
    DEV_DATASET = DATA_DIR+'ted_talks_processed.dev.pt'
    ALPHA = 0.8
    hidden_dropout_prob = 0.3
    EMBEDDING_DIM = 768
    HIDDEN_DIM = 128
    SELF_ADJUSTING=True
    SQUARE_DENOMINATOR=True
    USE_LSTM=True
    USE_CRF=False
    MODEL_NAME = 'bertcrf'
    MODEL_PATH = "/home/nxingyu/project/logs/models/"

    
    TOKENIZER = transformers.DistilBertTokenizerFast.from_pretrained(
        BASE_MODEL_PATH,
    )