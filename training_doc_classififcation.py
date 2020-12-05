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

train_samples, dev_samples = [], []
with open(os.path.join(dataset_path,'train.txt'), 'rt', encoding='utf8') as fIn:
    reader = fIn.readlines()
    for row in reader:
        row = row.strip().split('\t')
        if len(row) != 4:
            continue
        label = float(row[3])
        sent1 = row[0]
        sent2 = row[1]+' '+row[2]
        inp_example = InputExample(texts=[sent1, sent2], label=label)
        train_samples.append(inp_example)

with open(os.path.join(dataset_path,'dev.txt'), 'rt', encoding='utf8') as fIn:
    reader = fIn.readlines()
    for row in reader:
        row = row.strip().split('\t')
        if len(row) != 4:
            continue
        label = float(row[3])
        sent1 = row[0]
        sent2 = row[1]+' '+row[2]
        inp_example = InputExample(texts=[sent1, sent2], label=label)
        dev_samples.append(inp_example)

with open('./tmp.txt', 'w', encoding='utf-8') as file:
    for sample in train_samples[:10]:
        file.write(str(sample.texts) + ' ' + str(sample.label) + '\n')

train_dataset = SentencesDataset(train_samples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)

logging.info("Read dev dataset")
evaluator = BinaryClassificationEvaluator.from_input_examples(dev_samples, name='doc-dev')

# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)
