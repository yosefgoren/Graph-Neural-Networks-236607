import argparse
from itertools import islice
import math
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import itertools
from tqdm import tqdm

# from torch_geometric.utils import erdos_renyi_graph
from networkx import erdos_renyi_graph
from powerful_gnns.util import S2VGraph

from powerful_gnns.models.graphcnn import GraphCNN
from utils import INPUT_DIM, collision_probability
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
    def __init__(self, p, max_distance=10) -> None:
        super().__init__()
        self.p = p
        self.max_distance = max_distance
        
    
    def forward(self, x):
        # apply random permutation to x

        avg_meter = AverageMeter()

        for i in range(1, x.shape[0]):

            x_shifted = torch.roll(x, shifts=i, dims=0) 
            
            # calculate distance between x and x_perm
            dist = - torch.minimum(torch.norm(x - x_shifted, p=self.p, dim=1), torch.tensor(self.max_distance))


            avg_meter.update(dist.mean())
        
            return avg_meter.avg


class GraphDatset(torch.utils.data.IterableDataset):
    def __init__(self, num_nodes, edge_prob, seed, device) -> None:
        super().__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.edge_prob = edge_prob
        self.seed = seed

        self.index = 0

        self.node_features = torch.rand((num_nodes, INPUT_DIM), device=device)

    
    def __iter__(self):
        while True:
            permutation =  erdos_renyi_graph(self.num_nodes,
                              self.edge_prob,
                              seed=(self.seed + self.index))

            graph = S2VGraph(self.device, permutation, None, None, self.node_features)

            self.index += 1

            yield graph



def train(args, model, device, graph_loader, optimizer, epoch, criterion):
    model.to(device)
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    graph_iter = iter(graph_loader)

    loss_accum = 0
    for pos in pbar:
        

        batch_graph = next(graph_iter)
        output = model(batch_graph)

        #compute loss
        loss = criterion(output)

        print(loss)

        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)        
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
def pass_data_iteratively(model, graph_loader, total_samples_num , minibatch_size = 16):#@@@
    model.eval()
    output = []
    
    graph_iter = iter(graph_loader)

    minibatch_size=min(minibatch_size, total_samples_num)


    for i in range(0, math.ceil(total_samples_num / minibatch_size)):
        graphs_minibatch = [next(graph_iter) for i in range(minibatch_size)]
        graphs_minibatch = list(itertools.chain.from_iterable(graphs_minibatch))
        with torch.inference_mode():
            output_minibatch = model(graphs_minibatch)

        output.append(output_minibatch)
    return torch.cat(output, 0)

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def test(args, model, device, graph_loader, test_size, epoch):
    model.eval()

    output = pass_data_iteratively(model, graph_loader, test_size)

    assert not output.isnan().any()

    cp = collision_probability(output[..., 0].tolist())

    print("collision probability: %f" % (cp))

    return cp

def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,#@@@
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=20,#@@@
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=21000,
                        help='dataset seed')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="average", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="average", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--filename', type = str, default = "output.txt",
                                        help='output file')
    parser.add_argument('--test_size', type = str, default = 256,
                                        help='number of test graphs')
    parser.add_argument('--clip', type = float, default = float("inf"),
                                        help='gradient clipping value')
    parser.add_argument('--max_distance', type = float, default = float("inf"), #0.00025,
                                        help='max loss value between two points')
    parser.add_argument('--save_path', type=str, help='path to save the checkpoint model', default = "model.pt")


    args = parser.parse_args()

    #set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    #load dataset @@@
    dataset = GraphDatset(num_nodes = 512, edge_prob = 0.5, seed = args.seed, device=device)
    #create dataloader
    train_graph_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda x: x, num_workers=0)
    test_graph_loader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x, num_workers=0)

    model_parms = {
        'num_layers': args.num_layers,
        'num_mlp_layers': args.num_mlp_layers,
        'input_dim': INPUT_DIM,
        'hidden_dim': args.hidden_dim,
        'output_dim': 1,
        'final_dropout': args.final_dropout,
        'learn_eps': True,
        'eps_freeze': True,
        'graph_pooling_type': args.graph_pooling_type,
        'neighbor_pooling_type': args.neighbor_pooling_type,
        'device': device,
        'eps_initialization': 'random'  # 1e-4

    }


    model = GraphCNN(**model_parms).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    criterion = DistanceLoss(p=1, max_distance=args.max_distance)


    for epoch in range(1, args.epochs + 1):
        avg_loss = train(args, model, device, train_graph_loader, optimizer, epoch, criterion)
        collision_probability = test(args, model, device, test_graph_loader, args.test_size, epoch)

        scheduler.step()

        if not args.filename == "":
            with open(args.filename, 'w') as f:
                f.write("%f %f" % (avg_loss, collision_probability))
                f.write("\n")
        if not args.save_path == "":
            save_model(model, args.save_path)
        print("")

        print(model.eps)
    

if __name__ == '__main__':
    main()
