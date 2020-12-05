"""
This examples trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) for the STSbenchmark from scratch. It generates sentence embeddings
that can be compared using cosine-similarity to measure the similarity.

Usage:
python training_nli.py

OR
python training_nli.py pretrained_transformer_model_name
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses, models, util
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.readers import STSBenchmarkDataReader, InputExample
import logging
from datetime import datetime
import sys
import os
import gzip
import csv

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#Check if dataset exsist. If not, download and extract  it
dataset_path = 'data/doc_classification_debug'


#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-chinese'

# Read the dataset
train_batch_size = 16
num_epochs = 4
max_length = 256
model_save_path = 'output/doc_classification_'+model_name.replace("/", "-")+'_'+str(train_batch_size)+'_'+str(num_epochs)+'_'+str(max_length)


# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name, max_seq_length=max_length)
print(word_embedding_model.get_config_dict())
# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
# Convert the dataset to a DataLoader ready for training
logging.info("Read Document Classification train dataset")

test_samples = []

with open(os.path.join(dataset_path,'test.txt'), 'rt', encoding='utf8') as fIn:
    reader = fIn.readlines()
    for row in reader:
        row = row.strip().split('\t')
        if len(row) != 4:
            continue
        label = float(row[3])
        sent1 = row[0]
        sent2 = row[1]+' '+row[2]
        inp_example = InputExample(texts=[sent1, sent2], label=label)
        test_samples.append(inp_example)

model = SentenceTransformer(model_save_path)
test_evaluator = BinaryClassificationEvaluator.from_input_examples(test_samples, name='doc-test')
test_evaluator(model, output_path=model_save_path)
