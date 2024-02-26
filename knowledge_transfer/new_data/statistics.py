import pickle

amazon_train = '/data4/LightGCN-PyTorch/new_data/amazon-book/train.txt'
amazon_test = '/data4/LightGCN-PyTorch/new_data/amazon-book/test.txt'

yelp_train = '/data4/LightGCN-PyTorch/new_data/yelp2018/train.txt'
yelp_test = '/data4/LightGCN-PyTorch/new_data/yelp2018/test.txt'

citeulike_train = '/data4/LightGCN-PyTorch/new_data/citeulike-a/train.txt'
citeulike_test = '/data4/LightGCN-PyTorch/new_data/citeulike-a/test.txt'

def get_edges(path_train, path_test):
    edges = 0
    with open(path_train, 'r') as f:
        for l in f.readlines():
            l = l.strip('\n').split(' ')
            if len(l) > 1:
                edges += len(l) - 1
    with open(path_test, 'r') as f:
        for l in f.readlines():
            l = l.strip('\n').split(' ')
            if len(l) > 1:
                edges += len(l) - 1

    return edges


amazon_edges = get_edges(amazon_train, amazon_test)
yelp_edges = get_edges(yelp_train, yelp_test)
citeulike_edges = get_edges(citeulike_train, citeulike_test)

print(f'amazon_edges:{amazon_edges}')
print(f'yelp_edges:{yelp_edges}')
print(f'citeulike_edges:{citeulike_edges}')