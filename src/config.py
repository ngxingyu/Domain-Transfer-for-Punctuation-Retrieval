import transformers
class config:
    MAX_LEN = 128
    OVERLAP = 42
    TRAIN_BATCH_SIZE = 64
    DEV_BATCH_SIZE = 64
    EPOCHS = 10
    BASE_MODEL_PATH = 'distilbert-base-uncased'
    DATA_DIR = '/home/nxingyu/data/'
    TRAIN_DATASET = DATA_DIR+'ted_talks_processed.dev.pt'
    DEV_DATASET = DATA_DIR+'ted_talks_processed.dev.pt'
    ALPHA = 0.8
    hidden_dropout_prob = 0.3
    EMBEDDING_DIM = 5
    HIDDEN_DIM = 768
    LEARNING_RATE = 1e-4
    SELF_ADJUSTING=True
    SQUARE_DENOMINATOR=True
    MODEL_PATH = "/home/nxingyu/project/logs/models/"
    UNFROZEN_LAYERS = 0
    TOKENIZER = transformers.DistilBertTokenizerFast.from_pretrained(
        BASE_MODEL_PATH,
    )