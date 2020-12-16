import transformers
class config:
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 32
    DEV_BATCH_SIZE = 8
    EPOCHS = 3
    BASE_MODEL_PATH = 'distilbert-base-uncased'#"../input/bert-base-uncased/"
    MODEL_PATH = "model.bin"
    TRAINING_FILE = "../input/entity-annotated-corpus/ner_dataset.csv"
    TOKENIZER = transformers.DistilBertTokenizerFast.from_pretrained(
        BASE_MODEL_PATH,
    )
