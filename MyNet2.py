import torch
from torch import nn
from torch.functional import F
import numpy as np

# generate just indices
def generator (mols, batch_size = 4, seed=42):
    np.random.seed(seed)
    mols_shuffled = mols.copy()
    np.random.shuffle(mols_shuffled)
    for i in range(0, len(mols_shuffled), batch_size):
        yield np.sort(mols_shuffled[i:i+batch_size])
        

def expand_matrix(x, twoD = True):
    if (twoD):
        xx = np.zeros((29,29), dtype=x.dtype)
    else:
        xx = np.zeros((29,29, x.shape[2]), dtype=x.dtype)
    xx[:x.shape[0], :x.shape[1]] = x
    return xx

def expand_features(fvalues, atom_counts):
    xx = np.zeros((atom_counts.shape[0],29, fvalues.shape[1]), dtype='float32')
    startf = 0
    i = 0
    for n_atoms in atom_counts:
        xx[i, 0:+n_atoms] = fvalues[startf:(startf+n_atoms)]
        startf+=n_atoms
        i+=1
    return xx


class EmbedH(nn.Module):
    def __init__(self, l_in, l_h):
        super(EmbedH, self).__init__()
        self.fc = nn.Linear(l_in, l_h)
        #self.fc2 = nn.Linear(l_h, l_h)

    def forward(self, h):
        h = F.relu(self.fc(h))
        #h = F.relu(self.fc2(h))
        return h
    
class EmbedE(nn.Module):
    def __init__(self, l_in, l_h, l_g):
        super(EmbedE, self).__init__()
        self.fc = nn.Linear(l_in, l_h*l_g)
        # self.fc2 = nn.Linear(l_h*l_g, l_h*l_g)
    def forward(self, h):
        h = F.relu(self.fc(h))
        # h = F.relu(self.fc2(h))
        return h
    
    
class myNet(nn.Module):
    def __init__(self, h_in, e_in, h_h=64, h_g=16, nCells=5):
        super(myNet, self).__init__()
        self.nCells = nCells
        self.embed_h = EmbedH(h_in, h_h)
        self.embed_e = EmbedE(e_in, h_h, h_g)
        self.gru = nn.GRUCell(h_g, h_h)
        self.h_h = h_h
        self.h_g = h_g
    
    # edgelist -- torch.tensor
    # edgesfrom, edgeswhere - index of pair of atoms for which there is an edge in a graph i.e. A[from,where]==1
    def forward(self, A, h, E): #edgesfrom, edgeswhere):
        maskh = (A.sum(dim=-1, keepdim=True) > 0.1).float()
        maskE = A
        N_b = h.shape[0]
        n_a = A.shape[1]
        #N_ba = h.shape[0]
        
        
        h = self.embed_h(h) * maskh
        E = self.embed_e(E).view(N_b, n_a, n_a, self.h_g, self.h_h) * \
            maskE.unsqueeze(-1).unsqueeze(-1).float()
        
        h = h.view(N_b, 1, n_a, self.h_h, 1)
        #E = E.view(N_b, n_a, n_a, h_g, h_h)
        
        for it in range(self.nCells):
            M = E.matmul(h).sum(dim=1).view(N_b*n_a, self.h_g)
            h = self.gru(M, h.view(N_b*n_a, self.h_h)).view(N_b, 1, n_a, self.h_h, 1)
            
        # multiply h by mask
        # E not h*h but somel*h
        
        return h.view(N_b, n_a, self.h_h)
    
class poolNet(nn.Module):
    def __init__(self, h_in, h_h=64, nHidLayers=2):
        super(poolNet, self).__init__()
        self.inplayer = nn.Linear(h_in, h_h)
        self.hidlayers = nn.ModuleList([nn.Linear(h_h,h_h) for _ in range(nHidLayers)])
        self.outlayer = nn.Linear(h_h, 1)
    def forward(self, x):
        x = F.relu(self.inplayer(x))
        for layer in self.hidlayers:
            x = F.relu(layer(x))
        x = self.outlayer(x)
        return x