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
from layers import GroupMLP, MultiHeadBatchNorm

class SAGN(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, num_hops, n_layers, num_heads, loss_function, weight_style="attention", alpha=0.5, focal="first",
                 hop_norm="softmax", dropout=0.0, input_drop=0.0, attn_drop=0.0, negative_slope=0.2, zero_inits=False, position_emb=False):
        super(SAGN, self).__init__()
        self._num_heads = num_heads
        self._hidden = hidden
        self._out_feats = out_feats
        self._weight_style = weight_style
        self._alpha = alpha
        self._hop_norm = hop_norm
        self._zero_inits = zero_inits
        self._focal = focal
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_drop)
        # self.bn = nn.BatchNorm1d(hidden * num_heads)
        self.bn = MultiHeadBatchNorm(num_heads, hidden * num_heads)
        self.relu = nn.ReLU()
        self.input_drop = nn.Dropout(input_drop)
        self.multihop_encoders = nn.ModuleList([GroupMLP(in_feats, hidden, hidden, num_heads, n_layers, dropout) for i in range(num_hops)])
        self.res_fc = nn.Linear(in_feats, hidden * num_heads, bias=False)
        self.loss_fn = loss_function
        
        if weight_style == "attention":
            self.hop_attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, hidden)))
            self.hop_attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, hidden)))
            self.leaky_relu = nn.LeakyReLU(negative_slope)
        
        if position_emb:
            self.pos_emb = nn.Parameter(torch.FloatTensor(size=(num_hops, in_feats)))
        else:
            self.pos_emb = None
        
        self.post_encoder = GroupMLP(hidden, hidden, out_feats, num_heads, n_layers, dropout)
        # self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for encoder in self.multihop_encoders:
            encoder.reset_parameters()
        nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        if self._weight_style == "attention":
            if self._zero_inits:
                nn.init.zeros_(self.hop_attn_l)
                nn.init.zeros_(self.hop_attn_r)
            else:
                nn.init.xavier_normal_(self.hop_attn_l, gain=gain)
                nn.init.xavier_normal_(self.hop_attn_r, gain=gain)
        if self.pos_emb is not None:
            nn.init.xavier_normal_(self.pos_emb, gain=gain)
        self.post_encoder.reset_parameters()
        self.bn.reset_parameters()

    def get_logits(self, feats):
        out = 0
        feats = [self.input_drop(feat) for feat in feats]
        if self.pos_emb is not None:
            feats = [f +self.pos_emb[[i]] for i, f in enumerate(feats)]
        hidden = []
        for i in range(len(feats)):
            hidden.append(self.multihop_encoders[i](feats[i]).view(-1, self._num_heads, self._hidden))
        
        a = None
        if self._weight_style == "attention":
            if self._focal == "first":
                focal_feat = hidden[0]
            if self._focal == "last":
                focal_feat = hidden[-1]
            if self._focal == "average":
                focal_feat = 0
                for h in hidden:
                    focal_feat += h
                focal_feat /= len(hidden)
                
            astack_l = [(h * self.hop_attn_l).sum(dim=-1).unsqueeze(-1) for h in hidden]
            a_r = (focal_feat * self.hop_attn_r).sum(dim=-1).unsqueeze(-1)
            astack = torch.stack([(a_l + a_r) for a_l in astack_l], dim=-1)
            if self._hop_norm == "softmax":
                a = self.leaky_relu(astack)
                a = F.softmax(a, dim=-1)
            if self._hop_norm == "sigmoid":
                a = torch.sigmoid(astack)
            if self._hop_norm == "tanh":
                a = torch.tanh(astack)
            a = self.attn_dropout(a)
            
            for i in range(a.shape[-1]):
                out += hidden[i] * a[:, :, :, i]

        if self._weight_style == "uniform":
            for h in hidden:
                out += h / len(hidden)
        
        if self._weight_style == "exponent":
            for k, h in enumerate(hidden):
                out += self._alpha ** k * h

        out += self.res_fc(feats[0]).view(-1, self._num_heads, self._hidden)
        out = out.flatten(1, -1)
        out = self.dropout(self.relu(self.bn(out)))
        out = out.view(-1, self._num_heads, self._hidden)
        out = self.post_encoder(out)
        out = out.mean(1)
        return out

    def forward(self, feats, y=None):
        out = 0
        feats = [self.input_drop(feat) for feat in feats]
        if self.pos_emb is not None:
            feats = [f +self.pos_emb[[i]] for i, f in enumerate(feats)]
        hidden = []
        for i in range(len(feats)):
            hidden.append(self.multihop_encoders[i](feats[i]).view(-1, self._num_heads, self._hidden))
        
        a = None
        if self._weight_style == "attention":
            if self._focal == "first":
                focal_feat = hidden[0]
            if self._focal == "last":
                focal_feat = hidden[-1]
            if self._focal == "average":
                focal_feat = 0
                for h in hidden:
                    focal_feat += h
                focal_feat /= len(hidden)
                
            astack_l = [(h * self.hop_attn_l).sum(dim=-1).unsqueeze(-1) for h in hidden]
            a_r = (focal_feat * self.hop_attn_r).sum(dim=-1).unsqueeze(-1)
            astack = torch.stack([(a_l + a_r) for a_l in astack_l], dim=-1)
            if self._hop_norm == "softmax":
                a = self.leaky_relu(astack)
                a = F.softmax(a, dim=-1)
            if self._hop_norm == "sigmoid":
                a = torch.sigmoid(astack)
            if self._hop_norm == "tanh":
                a = torch.tanh(astack)
            a = self.attn_dropout(a)
            
            for i in range(a.shape[-1]):
                out += hidden[i] * a[:, :, :, i]

        if self._weight_style == "uniform":
            for h in hidden:
                out += h / len(hidden)
        
        if self._weight_style == "exponent":
            for k, h in enumerate(hidden):
                out += self._alpha ** k * h

        out += self.res_fc(feats[0]).view(-1, self._num_heads, self._hidden)
        out = out.flatten(1, -1)
        out = self.dropout(self.relu(self.bn(out)))
        out = out.view(-1, self._num_heads, self._hidden)
        out = self.post_encoder(out)
        out = out.mean(1)
        
        if y is not None:
            return out, self.loss_fn(out, y), a.mean(1) if a is not None else None
        else:
            return out, a.mean(1) if a is not None else None


def train(model, x, labels, train_loader, optimizer):
    # x [k*[n,d]]
    model.train()
    total_loss, iter_num = 0, 0
    for batch in tqdm(train_loader):
        batch_feat = [feat[batch].cuda() for feat in x]
        y_true = labels[batch]
        y_true = utils.label_platten(y_true, num_classes) # [b,c]
        _, loss, _ = model(batch_feat, y_true.cuda())
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
        logits, _ = model(batch_feat) # tensor [b,c]
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
    parser = argparse.ArgumentParser(description='UniKG-SAGN')
    parser.add_argument('--dataset', type=str, default='wiki_full')
    parser.add_argument('--use_sagn_embedding', action='store_true')
    parser.add_argument('--num_hops', type=int, default=5)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--save_steps', type=int, default=10)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num-heads', type=int, default=1)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--input_drop', type=float, default=0) 
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

    if not args.use_sagn_embedding:
        raise RuntimeError('UniKG need sagn embedding.')

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
    model = SAGN(x[0].size(-1), args.hidden_channels, num_classes, args.num_hops, args.num_layers, args.num_heads, loss_function).cuda()

    log = open(f'{args.dataset}-{args.num_layers}-{args.hidden_channels}-{parser.description}2.txt','w')

    for run in range(args.runs):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, x, y_true, train_loader, optimizer)
            acc, precision, recall, f1 = test(model, x, y_true, test_loader)
            if epoch % args.save_steps == 0:
                torch.save(model, f'model/sagn/{parser.description}2-{args.dataset}-{epoch}.pth')

            txt = f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {100 * acc:.2f}%, precision: {100 * precision:.2f}%, recall: {100 * recall:.2f}%, f1: {100 * f1:.2f}\n'
            if epoch % args.log_steps == 0:
                print(txt)
            log.write(txt)
            log.flush()

if __name__ == "__main__":
    main()