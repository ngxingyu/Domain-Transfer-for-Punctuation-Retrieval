import transformers
class config:
    MAX_LEN = 128
    OVERLAP = 126
    TRAIN_BATCH_SIZE = 64
    DEV_BATCH_SIZE = 128
    EPOCHS = 15
    BASE_MODEL_PATH = 'distilbert-base-uncased'
    DATA_DIR = '/home/nxingyu/project/data/'
    TRAIN_DATASET = DATA_DIR+'ted_talks_processed.train.pt'
    DEV_DATASET = DATA_DIR+'ted_talks_processed.dev.pt'
    ALPHA = 0.8
    hidden_dropout_prob = 0.3
    EMBEDDING_DIM = 768
    HIDDEN_DIM = 128
    LEARNING_RATE = 1e-5
    SELF_ADJUSTING=True
    SQUARE_DENOMINATOR=True
    USE_CRF=False
    MODEL_NAME = 'bertblstmcrf'
    MODEL_PATH = "/home/nxingyu/project/logs/models/"
    UNFROZEN_LAYERS = 0
    TOKENIZER = transformers.DistilBertTokenizerFast.from_pretrained(
        BASE_MODEL_PATH,
    )