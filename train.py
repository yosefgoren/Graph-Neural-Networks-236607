import argparse
from itertools import islice
import math
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from tqdm import tqdm

# from torch_geometric.utils import erdos_renyi_graph
from networkx import erdos_renyi_graph

from util import load_data, separate_data
from powerful_gnns.models.graphcnn import GraphCNN
from utils import collision_probability

from torch.utils.data import DataLoader


class AverageMeter(object):
    """Computes and stores the average and current value.

    Examples::
        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class DistanceLoss(nn.Module):
    def __init__(self, p, permutation_num=1) -> None:
        super().__init__()
        self.p = p
        self.permutation_num = permutation_num
        
    
    def forward(self, x):
        # apply random permutation to x

        avg_meter = AverageMeter()

        for i in range(self.permutation_num):

            perm = torch.randperm(x.shape[0])
            x_perm = x[perm]

            # calculate distance between x and x_perm
            dist = torch.norm(x - x_perm, p=self.p, dim=1)


            avg_meter.update(dist.mean())
        
        return avg_meter.avg


class GraphDatset(torch.utils.data.IterableDataset):
    def __init__(self, num_nodes, edge_prob, seed) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.edge_prob = edge_prob
        self.seed = seed

        self.index = 0
    
    def __iter__(self):
        graph =  erdos_renyi_graph(self.num_nodes,
                          self.edge_prob,
                          seed=(self.seed + self.index))
        self.index += 1

        return graph


criterion = DistanceLoss()

def train(args, model, device, graph_loader, optimizer, epoch):
    model.to(device)
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    for pos in pbar:

        batch_graph = islice(graph_loader, args.batch_size).to(device, non_blocking=True)
        output = model(batch_graph)

        #compute loss
        loss = criterion(output)

        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()
        

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        #report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum/total_iters
    print("loss training: %f" % (average_loss))
    
    return average_loss

###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs_loader, total_samples_num , minibatch_size = 64):
    model.eval()
    output = []
    for i in range(0, math.ceil(total_samples_num / minibatch_size)):
        graphs_minibatch = islice(graphs_loader, minibatch_size)
        with torch.inference_mode():
            output_minibatch = model(graphs_minibatch)

        output.append(output_minibatch)
    return torch.cat(output, 0)

def test(args, model, device, graph_loader, test_size, epoch):
    model.eval()

    output = pass_data_iteratively(model, graph_loader)

    cp = collision_probability(output.tolist())

    print("collision probability: %f" % (cp))

    return cp

def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=350,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', action="store_true",
    					help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type = str, default = "",
                                        help='output file')
    args = parser.parse_args()

    #set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    #load dataset
    dataset = GraphDatset(num_nodes = 1024, edge_prob = 0.5, seed = 22000)
    #create dataloader
    graph_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)



    ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
    train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)

    model = GraphCNN(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)


    for epoch in range(1, args.epochs + 1):
        scheduler.step()

        avg_loss = train(args, model, device, train_graphs, optimizer, epoch)
        acc_train, acc_test = test(args, model, device, train_graphs, test_graphs, epoch)

        if not args.filename == "":
            with open(args.filename, 'w') as f:
                f.write("%f %f %f" % (avg_loss, acc_train, acc_test))
                f.write("\n")
        print("")

        print(model.eps)
    

if __name__ == '__main__':
    main()
