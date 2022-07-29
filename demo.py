import time
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data.dataset import random_split
from torch import nn
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
import torch
import torch.nn.functional as F
from torchtext.datasets import AG_NEWS
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import Sampler
torch.autograd.set_detect_anomaly(True)

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
        # if _label == 4:
        #     pass
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


class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class, theta, p=0.1):
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
        # for mlp
        self.logistic_regression = nn.Sequential(
            nn.Linear(1, 2),
            nn.ReLU(),
            nn.Linear(2, 1)
        )
        self.theta = theta
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        # add conv layer
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

        def init_energy(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)
        self.logistic_regression.apply(init_energy)

    def forward(self, text):
        # get embedding, output shape (batch_size, embed_size, text_length)
        embedded = self.embedding(text)
        embedded = embedded.transpose(1, 2)
        # apply conv layer, output shape (batch_size, out_channels, text_length)
        out = self.conv(embedded)
        # max over text_length, output shape (batch_size, out_channels)
        # this is the feature space that we want to sample from
        out, _ = out.max(dim=-1)
        feature = out.detach()
        # apply dropout to avoid overfitting
        out = self.dropout(out)
        out = self.fc(out)
        return out, feature

    def energy(self, feature):
        # to-do: do we need drop out?
        # out = self.dropout(feature)
        # print(f'feature has shape {feature.shape}')
        logit = self.fc(feature)
        return -1 * (torch.log(torch.tensor(1/3)) + torch.logsumexp(logit, dim=1, keepdim=True))

    def mlp(self, x):
        # return self.logistic_regression(x).squeeze()
        return (self.theta * x).squeeze()
        # out = self.input_fc(x.reshape((-1, 1)))
        # out = self.hidden_fc(out)
        # return self.output_fc(out).squeeze()


train_iter = AG_NEWS(split='train')
num_class = len(set([label for (label, text) in train_iter])) - 1
vocab_size = len(vocab)
emsize = 64
theta = torch.tensor(1.0, dtype=float, device=device, requires_grad=True)
print(f'Vocab size is {vocab_size}')
print(f'Embedding size is {emsize}')
print(f'Num of class is {num_class}')
print('-' * 59)
model = TextClassificationModel(
    vocab_size, emsize, num_class, theta).to(device)
# mlp = MLP().to(device)

# consturct variables for GMM
# shape (num_class, feature_dim=out_channels)
class_mean = torch.from_numpy(np.zeros((num_class, 32))).float().to(device)
class_cov = torch.eye(32).float()
# assume num of classes is 4 - 1 = 3
class_cov = class_cov.repeat(num_class, 1, 1).to(device)
class_count = torch.from_numpy(np.zeros(num_class)).to(device)
feature_batch = None


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def is_pos_semi_def(x):
    return np.all(np.linalg.eigvals(x) >= 0)


def is_symmetric(x):
    # return (x == x.transpose(0, 1)).all()
    return (torch.abs(x - x.transpose(0, 1)) < 1e-5).all()


def print_diff(x):
    # return (x == x.transpose(0, 1)).all()
    print((torch.abs(x - x.transpose(0, 1))).sum())


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
        x = feature_batch[i]
        u = class_mean[label]
        t = class_count[label]
        delta = (x - u).reshape((1, -1))
        class_cov[label] *= t / (1 + t)
        # class_cov[label] += t / ((1 + t) ** 2) \
        #  * (feature_batch[i] - class_mean[label]).reshape((-1, 1)) @ (feature_batch[i] - class_mean[label]).reshape((1, -1))
        class_cov[label] += t / ((1 + t) ** 2) * delta.T @ delta


def train(dataloader, class_count, class_mean, class_cov, feature_batch, beta, sample=False):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    classidx_to_remove = 3
    for idx, (label, text) in enumerate(dataloader):
        # seperate id and ood data
        select_id = label != classidx_to_remove
        label, text = label[select_id], text[select_id]

        predicted_label, feature = model(text)
        cls_loss = criterion(predicted_label, label)
        cls_loss.backward(retain_graph=True)
        # clip the gradient (really necessary?)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)

        # code for calculate mean and covariance for GMM
        # select features that has correct label prediction
        feature_batch = feature[predicted_label.argmax(1) == label]
        label_batch = label[predicted_label.argmax(1) == label]
        class_count += torch.bincount(label_batch, minlength=num_class)
        # only use 1000 samples for each class, here assume the number of classes is 3
        exclude = torch.arange(0, num_class)[class_count > 1000]
        for i in range(exclude.shape[0]):
            select = [label_batch != exclude[i]]
            label_batch = label_batch[select]
            feature_batch = feature_batch[select]
        # print(
        #     f'epoch {epoch}, batch {idx}: pre-update symmetric {is_symmetric(class_cov[0].detach())}')
        update_mean(class_mean, class_count, feature_batch, label_batch)
        update_cov(class_mean, class_cov, class_count,
                   feature_batch, label_batch)

        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

        # only start sampling after some epoches
        if sample:
            m = MultivariateNormal(
                class_mean[0], (class_cov[0] + class_cov[0].T)/2)
            r = m.sample((1000,))
            p = m.log_prob(r)
            # the feature space outlier
            _, indices = torch.max(p, 0)
            outlier = r[indices]
            # compute the uncertainty loss
            uncertainty_loss = -1 * beta * \
                F.logsigmoid(model.mlp(model.energy(outlier.unsqueeze(0))))
            # to-do: update the formula, Done
            # to-do: not sure how many iid features are needed
            # to-do: gradient update?
            uncertainty_loss += -1 * beta * \
                F.logsigmoid(-1 * model.mlp(model.energy(feature[:1])))
            uncertainty_loss.backward()
            optimizer.step()
            # L.register_hook(lambda grad: print(grad))
            if model.theta != 1.0:
                print(model.theta)


def evaluate(dataloader, gamma):
    model.eval()
    total_acc, total_count = 0, 0
    # measure the accuracy of detecting id data as id
    total_id_acc, total_id_count = 0, 0
    # measure the accuracy of detecting ood data as ood
    total_ood_acc, total_ood_count = 0, 0

    classidx_to_remove = 3
    with torch.no_grad():
        for idx, (label, text) in enumerate(dataloader):
            select = label != classidx_to_remove
            label, text = label[select], text[select]
            select_ood = label == classidx_to_remove
            label_ood, text_ood = label[select_ood], text[select_ood]

            predicted_label, feature = model(text)
            _, feature_ood = model(text_ood)

            # print(f'energy in evaluation: {torch.sigmoid(model.energy(feature)).mean()}')
            id = torch.sigmoid(model.mlp(model.energy(feature))) >= gamma
            # ood = torch.sigmoid(model.mlp(model.energy(feature))) > gamma
            ood = torch.sigmoid(model.mlp(model.energy(feature_ood))) < gamma

            loss = criterion(predicted_label[id], label[id])
            total_acc += (predicted_label[id].argmax(1)
                          == label[id]).sum().item()
            total_count += label[id].size(0)
            total_id_acc += id.sum()
            total_id_count += label.shape[0]
            total_ood_count += label_ood.size(0)
            total_ood_acc += ood.sum()

    return total_acc/total_count, total_id_acc/total_id_count, total_ood_acc/total_ood_count


def get_gamma(dataloader):
    model.eval()
    # the number of id data is about 4500
    # will have trailing zeros in the end
    l, c = 120000, 0
    energies = torch.zeros(l)

    classidx_to_remove = 3
    with torch.no_grad():
        for idx, (label, text) in enumerate(dataloader):
            select = label != classidx_to_remove
            label, text = label[select], text[select]

            _, feature = model(text)
            # print(torch.sigmoid(model.mlp(model.energy(feature))).shape)
            # print(label.shape[0], count + label.shape[0])
            energies[c: c + label.shape[0]
                     ] = torch.sigmoid(model.mlp(model.energy(feature)))
            c += label.shape[0]
    print(f'c = {c}')
    energies = energies[:c]
    energies, _ = torch.sort(energies)
    print(energies)
    print(energies.mean())
    gamma = torch.quantile(energies, 0.95)
    return gamma


# Hyperparameters
EPOCHS = 10  # epoch
LR = 5  # learning rate
BATCH_SIZE = 32  # batch size for training
beta = 0.1  # weight of uncertainty loss
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

sample = False
for epoch in range(1, EPOCHS + 1):
    if epoch >= 1:
        sample = True
    epoch_start_time = time.time()
    train(train_dataloader, class_count, class_mean,
          class_cov, feature_batch, beta, sample=sample)
    gamma = get_gamma(train_dataloader)
    print(f'computed gamma is {gamma}')
    accu_val, accu_id, accu_ood = evaluate(valid_dataloader, gamma)
    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:6.3f} | id accuracy {:6.3f} | ood accuracy {:6.3f}'.format(epoch,
                                                                                       time.time() - epoch_start_time,
                                                                                       accu_val, accu_id, accu_ood))
    print('-' * 59)
