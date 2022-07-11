from termios import CKILL
import time
from tkinter import W
from sklearn import cluster
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data.dataset import random_split
from torch import nn
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
import torch
import torch.nn.functional as F
from torchtext.datasets import AG_NEWS
from sklearn.mixture import GaussianMixture
import numpy as np
import psutil
from torch.distributions.multivariate_normal import MultivariateNormal

train_iter = iter(AG_NEWS(split='train'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'running on {device}')
tokenizer = get_tokenizer('basic_english')
train_iter = AG_NEWS(split='train')


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


# build_vocab_from_iterator accepts iterator that yield list or iterator of tokens
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
# text pipeline converts a text string into a list of integers based on look up table
# text -> tokenizer -> vocab indices
# e.g. 'here is the an example' -> [475, 21, 2, 30, 5297]
def text_pipeline(x): return vocab(tokenizer(x))
# label pipline converts the string label into integers, e.g. '10' -> 9
def label_pipeline(x): return int(x) - 1


def collate_batch(batch):
    label_list, text_list = [], []
    for (_label, _text) in batch:
        # print(_label, _text)
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        processed_text = F.pad(
            processed_text, (0, 50 - processed_text.shape[0]))
        # if processed_text.shape[0] != 50:
        # print(processed_text.shape[0] )
        # print(len(tokenizer(_text)), processed_text.shape)
        text_list.append(processed_text.unsqueeze(0))
        # offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    # offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list, 0)
    # text_list has shape (32, 50), i.e (batch_size, tokenized_text_size)
    return label_list.to(device), text_list.to(device)


train_iter = AG_NEWS(split='train')
dataloader = DataLoader(train_iter, batch_size=8,
                        shuffle=False, collate_fn=collate_batch)


class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class, p=0.1):
        super(TextClassificationModel, self).__init__()
        self.vocab_size = vocab_size
        self.label_size = num_class
        self.embed_dim = embed_dim
        # self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(in_channels=embed_dim,
                              out_channels=32, kernel_size=7, padding="same")
        self.dropout = nn.Dropout(p)
        self.fc = nn.Linear(32, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        # add conv layer
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text):
        # get embedding, output shape (batch_size, embed_size, text_length)
        embedded = self.embedding(text)
        embedded = embedded.transpose(1, 2)
        # apply conv layer, output shape (batch_size, out_channels, text_length)
        out = self.conv(embedded)
        # max over text_length, output shape (batch_size, out_channels)
        # this is the feature space that we want to sample from
        out, _ = out.max(dim=-1)
        feature = out
        # apply dropout to avoid overfitting
        out = self.dropout(out)
        out = self.fc(out)
        return out, feature
    
    def energy(self, feature):
        out = self.dropout(feature)
        logit = self.fc(out)
        return -1 * torch.log(torch.exp(logit).sum())

class MLP(nn.Module):
    pass


train_iter = AG_NEWS(split='train')
num_class = len(set([label for (label, t4xt) in train_iter]))
vocab_size = len(vocab)
emsize = 64
print(f'Vocab size is {vocab_size}')
print(f'Embedding size is {emsize}')
print(f'Num of class is {num_class}')
print('------------------------------')
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
# consturct variables for GMM
# shape (num_class, feature_dim=out_channels)
class_mean = torch.from_numpy(np.zeros((4, 32))).float().to(device)
class_cov = torch.eye(32).float()
# assume num of classes is 4
class_cov = class_cov.repeat(4, 1, 1).to(device)
class_count = torch.from_numpy(np.zeros(4)).to(device)
feature_batch = None


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def is_pos_semi_def(x):
    return np.all(np.linalg.eigvals(x) >= 0)

def is_symmetric(x):
    return (x == x.transpose(0, 1)).all()

def update_bayesian(means, covs):
    precs = torch.linalg.inv(covs)
    diff_precs = precs.sum(0) - precs
    precs_det = torch.linalg.det(precs)
    total_precs_det = torch.linalg.det(precs.sum(0))
    diff_precs_det = torch.linalg.det(diff_precs)
    omegas = (total_precs_det - diff_precs_det + precs_det) / \
        (precs.shape[0] * total_precs_det +
         (precs_det - diff_precs_det).sum(0))
    omegas = torch.unsqueeze(torch.unsqueeze(omegas, 1), 2)
    weighted_precs = omegas * precs
    final_cov = torch.linalg.inv(weighted_precs.sum(0))
    final_mean = torch.squeeze(
        final_cov @ (weighted_precs @ torch.unsqueeze(means, 2)).sum(0))
    return final_mean, final_cov


# online update gradient with a batch (not training batch size) of feature observations
def update_mean(class_mean, class_count, feature_batch, label_batch):
    # class_mean of shape (num_class, feature_dim)
    # feature_batch of shape (batch_size, feature_dim)
    # label_batch of shape (batch_size)
    for i in range(feature_batch.shape[0]):
        label = label_batch[i]
        class_mean[label] += (feature_batch[i] -
                              class_mean[label]) / (1 + class_count[label])

# online update covariance matrix


def update_cov(class_mean, class_cov, class_count, feature_batch, label_batch):
    # class_mean of shape (num_class, feature_dim)
    # class_cov of shape (num_class, feature_dim, feature_dim)
    # feature_batch of shape (batch_size, feature_dim)
    # label_batch of shape (batch_size)
    for i in range(feature_batch.shape[0]):
        label = label_batch[i]
        if not is_symmetric(class_cov[label]):
            print('not symmetric before update')
        x = feature_batch[i]
        u = class_mean[label]
        t = class_count[label]
        delta = (x - u).reshape((1, -1))
        class_cov[label] *= t / (1 + t)
        # class_cov[label] += t / ((1 + t) ** 2) \
            #  * (feature_batch[i] - class_mean[label]).reshape((-1, 1)) @ (feature_batch[i] - class_mean[label]).reshape((1, -1))
        class_cov[label] += t / ((1 + t) ** 2) * delta.T @ delta
        if not is_symmetric(delta.T @ delta):
            print('not symmetric after update')

def train(dataloader, class_count, class_mean, class_cov, feature_batch, sample=True):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text) in enumerate(dataloader):
        predicted_label, feature = model(text)
        cls_loss = criterion(predicted_label, label)
        cls_loss.backward()
        # clip the gradient (really necessary?)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)

        # code for calculate mean and covariance for GMM
        # try to avoid for loop here and update every batch
        # the features that has correct label prediction
        feature_batch = feature[predicted_label.argmax(1) == label]
        label_batch = label[predicted_label.argmax(1) == label]
        # todo: increment the class count
        class_count += torch.bincount(label_batch, minlength=4)
        # add initalization
        # if not feature_batch:
        #     feature_batch = feature.reshape((1, -1))
        #     label_batch = label
        # else:
        #     torch.cat((feature_batch, feature), 0)
        #     torch.cat((label_batch, label), 0)
        # if feature_batch and feature_batch.shape[0] == 50:
        # only use 1000 samples for each class
        # here assume the number of classes is 4
        exclude = torch.arange(0, 4)[class_count > 1000]
        for i in range(exclude.shape[0]):
            select = [label_batch != exclude[i]]
            label_batch = label_batch[select]
            feature_batch = feature_batch[select]
        # mean, cov = update_bayesian(class_mean, class_cov)
        # cov += (0.01 * torch.eye(32, device=device))
        # if (cov < 0).sum() > 0:
            # print('??')
        # print(mean.shape, cov.shape)
        print(f'is positive semi definite {is_pos_semi_def(class_cov[0].detach().cpu().numpy())}')
        print(f'is positive definite {is_pos_def(class_cov[0].detach().cpu().numpy())}')
        update_mean(class_mean, class_count, feature_batch, label_batch)
        update_cov(class_mean, class_cov, class_count,
                   feature_batch, label_batch)
        #     feature_batch, label_batch = None, None

        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

        # only start sampling after some epoches
        if sample:
            print('chech if the cov matrix is symmetric')
            print(f'symetric: {is_symmetric(class_cov[0])}')
            # print(class_mean[0].shape, class_cov[0].shape)
            # m = MultivariateNormal(class_mean[0], class_cov[0] + 0.01 * torch.eye(32, device=device))
            print(f'is positive definite {is_pos_def(class_cov[0].detach().cpu().numpy())}')
            m = MultivariateNormal(class_mean[0], class_cov[0])
            # m = MultivariateNormal(mean, cov)
            # m = MultivariateNormal(torch.zeros(32), torch.eye(32))
            # normally distributed with mean=`[0,0]` and covariance_matrix=`I`
            r = m.sample((1000,))
            p = m.log_prob(r)
            # the feature space outlier
            outlier = torch.amin(p, 0)
            # compute the uncertainty loss
            # uncertainty_loss = -1 * torch.log(torch.sigmoid(mlp(model.energy(outlier))))
            # to-do: not sure how many iid features are needed
            # uncertainty_loss += -1 * torch.log(torch.sigmoid(mlp(model.energy(feature[0]))))
            # uncertainty_loss.backward()


def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text) in enumerate(dataloader):
            predicted_label, feature = model(text)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count


# Hyperparameters
EPOCHS = 10  # epoch
LR = 5  # learning rate
BATCH_SIZE = 32  # batch size for training
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None
train_iter, test_iter = AG_NEWS()
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = \
    random_split(train_dataset, [num_train, len(train_dataset) - num_train])

train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_batch)

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader, class_count, class_mean, class_cov, feature_batch)
    # print(class_count, class_mean, class_cov)
    accu_val = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))
    print('-' * 59)
