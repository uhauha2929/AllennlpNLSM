# -*- coding: utf-8 -*-
# @Time    : 2019/5/30 15:39
# @Author  : uhauha2929
from itertools import chain

import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score

from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import SnliReader
from allennlp.data.iterators import BucketIterator
from allennlp.models import BiMpm
from allennlp.modules import BiMpmMatching, Embedding, FeedForward
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import Activation
from allennlp.nn.util import move_to_device
from allennlp.training import Trainer

max_vocab_size = 50000
vocab_dir = 'vocab/'
serialization_dir = 'checkpoints/'

device = 5
torch.cuda.set_device(device)

batch_size = 64
hid_dim = 100
embed_dim = 300

lr = 1e-3
grad_clipping = 5
dropout = 0.1
# lazy=False将数据一次性加入内存, lazy=True边训练边加载
print('data loading, please wait...')
reader = SnliReader(lazy=True)
train_dataset = reader.read("snli_1.0/snli_1.0_train.jsonl")
dev_dataset = reader.read("snli_1.0/snli_1.0_dev.jsonl")
test_dataset = reader.read("snli_1.0/snli_1.0_test.jsonl")

if os.path.exists(vocab_dir):
    vocab = Vocabulary.from_files(vocab_dir)
else:
    vocab = Vocabulary.from_instances(chain(train_dataset, dev_dataset), max_vocab_size=max_vocab_size)
    vocab.save_to_files(vocab_dir)

print("vocab_size: {}".format(vocab.get_vocab_size()))

train_iterator = BucketIterator(batch_size=batch_size, sorting_keys=[("premise", "num_tokens")])
dev_iterator = BucketIterator(batch_size=batch_size, sorting_keys=[("premise", "num_tokens")])
test_iterator = BucketIterator(batch_size=batch_size, sorting_keys=[("premise", "num_tokens")])
train_iterator.index_with(vocab)
dev_iterator.index_with(vocab)
test_iterator.index_with(vocab)

en_embedding = Embedding(num_embeddings=vocab.get_vocab_size(), embedding_dim=embed_dim)
embedder = BasicTextFieldEmbedder({"tokens": en_embedding})

# 词级别的匹配, 隐层向量维度即为词向量维度, 这里我就不加full_match了
matcher_word = BiMpmMatching(hidden_dim=embed_dim, num_perspectives=20, with_full_match=False)

encoder1 = PytorchSeq2SeqWrapper(torch.nn.GRU(embed_dim, hid_dim, batch_first=True, bidirectional=True))
matcher_forward1 = BiMpmMatching(hid_dim, 20, is_forward=True)
matcher_backward1 = BiMpmMatching(hid_dim, 20, is_forward=False)

encoder2 = PytorchSeq2SeqWrapper(torch.nn.GRU(2 * hid_dim, hid_dim, batch_first=True, bidirectional=True))
matcher_forward2 = BiMpmMatching(hid_dim, 20, is_forward=True)
matcher_backward2 = BiMpmMatching(hid_dim, 20, is_forward=False)

aggregator = PytorchSeq2VecWrapper(torch.nn.GRU(matcher_word.get_output_dim()
                                                + matcher_forward1.get_output_dim()
                                                + matcher_backward1.get_output_dim()
                                                + matcher_forward2.get_output_dim()
                                                + matcher_backward2.get_output_dim(),
                                                hid_dim,
                                                batch_first=True,
                                                bidirectional=True))

classifier = FeedForward(hid_dim * 2 * 2, 3,
                         [hid_dim * 2, hid_dim, vocab.get_vocab_size('labels')],
                         Activation.by_name('leaky_relu')(),
                         dropout=dropout)

model = BiMpm(vocab=vocab,
              text_field_embedder=embedder,
              matcher_word=matcher_word,
              encoder1=encoder1,
              matcher_forward1=matcher_forward1,
              matcher_backward1=matcher_backward1,
              encoder2=encoder2,
              matcher_forward2=matcher_forward2,
              matcher_backward2=matcher_backward2,
              aggregator=aggregator,
              classifier_feedforward=classifier,
              dropout=dropout)
print(model)


def train():
    model.cuda(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=train_iterator,
                      train_dataset=train_dataset,
                      validation_dataset=dev_dataset,
                      patience=5,
                      validation_iterator=dev_iterator,
                      num_epochs=20,
                      serialization_dir=serialization_dir,
                      grad_clipping=grad_clipping,
                      cuda_device=device)

    trainer.train()


def evaluate(model_path):
    model.load_state_dict(torch.load(model_path))
    model.cuda(device)
    model.eval()
    generator = test_iterator(test_dataset, num_epochs=1)
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in generator:
            batch = move_to_device(batch, device)
            output_dict = model.forward(batch['premise'], batch['hypothesis'])
            y_true.extend(batch['label'].cpu().numpy().tolist())
            y_pred.extend([np.argmax(p) for p in output_dict['probs'].cpu().numpy()])

    print(accuracy_score(y_true, y_pred))


if __name__ == '__main__':
    # train()
    evaluate("checkpoints/best.th")  # 0.8256311074918566
