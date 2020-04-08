import transformers


MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
BERT_PATH = "./bert-base-multi-cased/"
MODEL_PATH = "./bert-base-multi-cased/pytorch_model.bin"
TRAINING_FILE = "./dataset/train.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)
