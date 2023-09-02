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

# from ogb.nodeproppred import Evaluator

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, loss_fn):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.loss_fn = loss_fn
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, y=None):
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        if y is not None:
            loss = self.loss_fn(x, y)
            return x, loss
        else:
            return x


def train(model, x, labels, train_loader, optimizer):
    model.train()
    total_loss, iter_num = 0, 0
    for batch in tqdm(train_loader):
        batch_feat = x[batch].cuda() ### out of index
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
        batch = batch
        batch_feat = x[batch].cuda()
        logits = model(batch_feat) # tensor [b,c]
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
    parser = argparse.ArgumentParser(description='UniKG-SGC')
    parser.add_argument('--dataset', type=str, default='wiki_full')
    parser.add_argument('--use_sgc_embedding', action='store_true')
    parser.add_argument('--num_sgc_iterations', type=int, default = 3)
    # parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--save_steps', type=int, default=10)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=300000)
    args = parser.parse_args()
    print(args)

    device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    global loss_function, num_classes
    torch.manual_seed(123)
    # pdb.set_trace()

    if not args.use_sgc_embedding:
        raise RuntimeError('UniKG need sgc embedding.')

    if args.dataset == 'wiki_full':
        x = torch.load(f'feature_all/feature_all_pca_{args.num_sgc_iterations}.pth')
        y_true = torch.load('multilabels/labels_cluster.pth')
        num_classes = torch.max(y_true).item()+1
        print(y_true , y_true.shape)

    if args.dataset == 'wiki_1M':
        x = torch.load(f'feature_1M/feature_1M_pca_{args.num_sgc_iterations}.pth')
        y_true = torch.load('multilabels/labels_cluster_1M.pth')
        num_classes = 2000
        print(y_true , y_true.shape)

    if args.dataset == 'wiki_10M':
        x = torch.load(f'feature_10M/feature_10M_pca_{args.num_sgc_iterations}.pth')
        y_true = torch.load('multilabels/labels_cluster_10M.pth')
        num_classes = 2000
        print(y_true , y_true.shape)

    entity_num = x.shape[0]
    shuffle_idx = torch.randperm(x.shape[0])
    split = [0.8]
    train_idx = shuffle_idx[:int(split[0]*entity_num)]
    # val_idx = shuffle_idx[int(split[0]*entity_num):int(split[1]*entity_num)]
    test_idx = shuffle_idx[int(split[1]*entity_num):]
    
    train_loader = torch.utils.data.DataLoader(train_idx, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_idx, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_function = nn.BCEWithLogitsLoss()
    model = MLP(x.size(-1), args.hidden_channels, num_classes, args.num_layers, args.dropout, loss_function).cuda()

    log = open(f'{args.dataset}-{args.num_layers}-{args.hidden_channels}-{parser.description}.txt','w')

    for run in range(args.runs):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, x, y_true, train_loader, optimizer)
            acc, precision, recall, f1 = test(model, x, y_true, test_loader)
            if epoch % args.save_steps == 0:
                torch.save(model, f'model/mlp/{parser.description}-{args.dataset}-{epoch}.pth')

            txt = f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {100 * acc:.2f}%, precision: {100 * precision:.2f}%, recall: {100 * recall:.2f}%, f1: {100 * f1:.2f}\n'
            if epoch % args.log_steps == 0:
                print(txt)
            log.write(txt)
            log.flush()

if __name__ == "__main__":
    main()