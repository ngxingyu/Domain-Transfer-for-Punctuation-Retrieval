from transformers import DistilBertTokenizerFast
class config:
    max_len = 128
    overlap = 126
    train_batch_size = 64
    dev_batch_size = 64
    gpu_device = 'cuda:3'#'cpu'#
    freeze_epochs = 20
    freeze_lr = 1e-4
    unfreeze_epochs = 20
    unfreeze_layers = 6
    unfreeze_lr = 1e-5
    base_model_path = 'distilbert-base-uncased'
    data_dir = '/home/nxingyu/project/data/'
    train_dataset = data_dir+'ted_talks_processed.dev.pt'
    dev_dataset = data_dir+'ted_talks_processed.dev.pt'
    alpha = 0.8
    hidden_dropout_prob = 0.3
    embedding_dim = 768
    num_labels = 10
    hidden_dim = 128
    self_adjusting=True
    square_denominator=False
    use_crf=True
    model_name = 'bertcrf'
    model_path = "/home/nxingyu/project/logs/models/"
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        base_model_path,
    )
