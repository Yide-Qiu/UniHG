import argparse
import os
import torch
import torch.nn as nn
import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
import pickle as pkl
import pdb
import utils
from tqdm import tqdm
from layers import FeedForwardNetII, FeedForwardNet

class R_GAMLP(nn.Module):  # recursive GAMLP
    def __init__(self, nfeat, hidden, nclass, num_hops, dropout, input_drop, att_dropout, alpha, n_layers_1, n_layers_2, loss_function, 
                 act="relu", pre_process=False, residual=False,pre_dropout=False,bns=False):
        super(R_GAMLP, self).__init__()
        self.num_hops = num_hops
        self.prelu = nn.PReLU()
        if pre_process:
            self.lr_att = nn.Linear(hidden + hidden, 1)
            self.lr_output = FeedForwardNetII(
                hidden, hidden, nclass, n_layers_2, dropout, alpha,bns)
            self.process = nn.ModuleList(
                [FeedForwardNet(nfeat, hidden, hidden, 2, dropout, bns) for i in range(num_hops)])
        else:
            self.lr_att = nn.Linear(nfeat + nfeat, 1)
            self.lr_output = FeedForwardNetII(
                nfeat, hidden, nclass, n_layers_2, dropout, alpha,bns)
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        self.att_drop = nn.Dropout(att_dropout)
        self.pre_process = pre_process
        self.res_fc = nn.Linear(nfeat, hidden)
        self.residual = residual
        self.pre_dropout=pre_dropout
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        self.loss_fn = loss_function
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.lr_att.weight, gain=gain)
        nn.init.zeros_(self.lr_att.bias)
        nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        nn.init.zeros_(self.res_fc.bias)
        self.lr_output.reset_parameters()
        if self.pre_process:
            for layer in self.process:
                layer.reset_parameters()

    def get_logits(self, feature_list):
        num_node = feature_list[0].shape[0]
        feature_list = [self.input_drop(feature) for feature in feature_list]
        input_list = []
        if self.pre_process:
            for i in range(self.num_hops):
                input_list.append(self.process[i](feature_list[i]))
        else:
            input_list = feature_list
        attention_scores = []
        attention_scores.append(self.act(self.lr_att(
            torch.cat([input_list[0], input_list[0]], dim=1))))
        for i in range(1, self.num_hops):
            history_att = torch.cat(attention_scores[:i], dim=1)
            att = F.softmax(history_att, 1)
            history = torch.mul(input_list[0], self.att_drop(
                att[:, 0].view(num_node, 1)))
            for j in range(1, i):
                history = history + \
                    torch.mul(input_list[j], self.att_drop(
                        att[:, j].view(num_node, 1)))
            attention_scores.append(self.act(self.lr_att(
                torch.cat([history, input_list[i]], dim=1))))
        attention_scores = torch.cat(attention_scores, dim=1)
        attention_scores = F.softmax(attention_scores, 1)
        right_1 = torch.mul(input_list[0], self.att_drop(
            attention_scores[:, 0].view(num_node, 1)))
        for i in range(1, self.num_hops):
            right_1 = right_1 + \
                torch.mul(input_list[i], self.att_drop(
                    attention_scores[:, i].view(num_node, 1)))
        if self.residual:
            right_1 += self.res_fc(feature_list[0])
            right_1 = self.dropout(self.prelu(right_1))
        if self.pre_dropout:
            right_1=self.dropout(right_1)
        return right_1


    def forward(self, feature_list, y=None):
        num_node = feature_list[0].shape[0]
        feature_list = [self.input_drop(feature) for feature in feature_list]
        input_list = []
        if self.pre_process:
            for i in range(self.num_hops):
                input_list.append(self.process[i](feature_list[i]))
        else:
            input_list = feature_list
        attention_scores = []
        attention_scores.append(self.act(self.lr_att(
            torch.cat([input_list[0], input_list[0]], dim=1))))
        for i in range(1, self.num_hops):
            history_att = torch.cat(attention_scores[:i], dim=1)
            att = F.softmax(history_att, 1)
            history = torch.mul(input_list[0], self.att_drop(
                att[:, 0].view(num_node, 1)))
            for j in range(1, i):
                history = history + \
                    torch.mul(input_list[j], self.att_drop(
                        att[:, j].view(num_node, 1)))
            attention_scores.append(self.act(self.lr_att(
                torch.cat([history, input_list[i]], dim=1))))
        attention_scores = torch.cat(attention_scores, dim=1)
        attention_scores = F.softmax(attention_scores, 1)
        right_1 = torch.mul(input_list[0], self.att_drop(
            attention_scores[:, 0].view(num_node, 1)))
        for i in range(1, self.num_hops):
            right_1 = right_1 + \
                torch.mul(input_list[i], self.att_drop(
                    attention_scores[:, i].view(num_node, 1)))
        if self.residual:
            right_1 += self.res_fc(feature_list[0])
            right_1 = self.dropout(self.prelu(right_1))
        if self.pre_dropout:
            right_1=self.dropout(right_1)



        right_1 = self.lr_output(right_1)

        if y is not None:
            loss = self.loss_fn(right_1, y)
            return right_1, loss
        else:
            return right_1


def train(model, x, labels, train_loader, optimizer):
    # x [k*[n,d]]
    model.train()
    total_loss, iter_num = 0, 0
    for batch in tqdm(train_loader):
        batch_feat = [feat[batch].cuda() for feat in x]
        y_true = labels[batch]
        y_true = utils.label_platten(y_true, num_classes) # [b,c]
        _, loss = model(batch_feat, y_true.cuda())
        loss_train = loss
        total_loss += loss_train
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        iter_num+=1
    loss = total_loss / iter_num
    return loss


@torch.no_grad()
def test(model, x, labels, test_loader):
    model.eval()

    bs = 0
    acc_values, precision_values, recall_values, f1_values = 0, 0, 0, 0
    for batch in test_loader:
        batch_feat = [feat[batch].cuda() for feat in x]
        logits= model(batch_feat) # tensor [b,c]
        y_true = labels[batch] # tensor [b,c]
        y_true = utils.label_platten(y_true, num_classes) # [b,c]
        acc_value, precision_value, recall_value, f1_value = utils.evaluate(logits, y_true.cuda())
        acc_values += acc_value
        precision_values += precision_value
        recall_values += recall_value
        f1_values += f1_value
        bs += 1
    acc, precision, recall, f1 = utils.average(acc_values, precision_values, recall_values, f1_values, bs)
    return acc, precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description='UniKG-GAMLP')
    parser.add_argument('--dataset', type=str, default='wiki_full')
    parser.add_argument('--use_gamlp_embedding', action='store_true')
    parser.add_argument('--num_hops', type=int, default=5)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--save_steps', type=int, default=10)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--n_layers_1', type=int, default=1)
    parser.add_argument('--num-heads', type=int, default=1)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--input_drop', type=float, default=0) 
    parser.add_argument('--att_dropout', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=300000)
    parser.add_argument('--alpha', type=float, default=0.5)
    args = parser.parse_args()
    print(args)

    device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    global loss_function, num_classes
    torch.manual_seed(123)
    # pdb.set_trace()

    if not args.use_gamlp_embedding:
        raise RuntimeError('UniKG need gamlp embedding.')

    if args.dataset == 'wiki_full':
        x = []
        x.append(pkl.load(open('feature_all/feature_all_pca.pkl', 'rb')))
        for i in range(1, args.num_hops):
            x.append(torch.load(f'feature_all/feature_all_pca_{i}.pth'))
        y_true = torch.load('multilabels/labels_cluster.pth')
        num_classes = torch.max(y_true).item()+1
        print(y_true , y_true.shape)

    if args.dataset == 'wiki_1M':
        x = []
        x.append(torch.load('feature_1M/feature_1M_pca.pth'))
        for i in range(1, args.num_hops):
            x.append(torch.load(f'feature_1M/feature_1M_pca_{i}.pth'))
        y_true = torch.load('multilabels/labels_cluster_1M.pth')
        num_classes = 2000
        print(y_true , y_true.shape)

    if args.dataset == 'wiki_10M':
        x = []
        x.append(torch.load('feature_10M/feature_10M_pca.pth'))
        for i in range(1, args.num_hops):
            x.append(torch.load(f'feature_10M/feature_10M_pca_{i}.pth'))
        y_true = torch.load('multilabels/labels_cluster_10M.pth')
        num_classes = 2000
        print(y_true , y_true.shape)

    entity_num = x[0].shape[0]
    shuffle_idx = torch.randperm(x[0].shape[0])
    split = [0.8]
    train_idx = shuffle_idx[:int(split[0]*entity_num)]
    # val_idx = shuffle_idx[int(split[0]*entity_num):int(split[1]*entity_num)]
    test_idx = shuffle_idx[int(split[0]*entity_num):]
    
    train_loader = torch.utils.data.DataLoader(train_idx, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_idx, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_function = nn.BCEWithLogitsLoss()
    model = R_GAMLP(x[0].size(-1), args.hidden_channels, num_classes, args.num_hops, args.dropout, args.input_drop, args.att_dropout,
                  args.alpha, args.n_layers_1, args.num_layers, loss_function).cuda()
    log = open(f'{args.dataset}-{args.num_layers}-{args.hidden_channels}-{parser.description}.txt','w')

    for run in range(args.runs):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, x, y_true, train_loader, optimizer)
            acc, precision, recall, f1 = test(model, x, y_true, test_loader)
            if epoch % args.save_steps == 0:
                torch.save(model, f'model/gamlp/{parser.description}-{args.dataset}-{epoch}.pth')
            txt = f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {100 * acc:.2f}%, precision: {100 * precision:.2f}%, recall: {100 * recall:.2f}%, f1: {100 * f1:.2f}\n'
            if epoch % args.log_steps == 0:
                print(txt)
            log.write(txt)
            log.flush()

if __name__ == "__main__":
    main()