import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_sparse import matmul
import math


class BaseMessageLayer(nn.Module):
   def __init__(self, d_model, update_type='lstm'):
       super().__init__()
       self.d_model = d_model
       if update_type == 'lstm':
           self.updater = nn.LSTM(d_model, d_model)
       elif update_type == 'rnn':
           self.updater = nn.RNN(d_model, d_model)
       elif update_type == 'linear':
           self.updater = nn.Sequential(
               nn.Linear(d_model, d_model),
               nn.ReLU(),
               nn.Linear(d_model, d_model)
           )

class LitToClauseLayer(BaseMessageLayer):
   def forward(self, adj_t, x_l, hidden):
        msg = matmul(adj_t, x_l)
        if self.updater.__class__.__name__ == 'RNN':
           hidden = hidden[0].unsqueeze(0)
           msg, new_hidden = self.updater(msg.unsqueeze(0), hidden)
           return [new_hidden[0].squeeze(0), None]
        elif self.updater.__class__.__name__  == 'LSTM':
           hidden = (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0))
           msg, new_hidden = self.updater(msg.unsqueeze(0), hidden)
           return [new_hidden[0].squeeze(0), new_hidden[1].squeeze(0)]
        
class VarToClauseLayer(BaseMessageLayer):    
    def forward(self, adj_t, x_v, hidden):        
        msg = matmul(adj_t, x_v)        
        if self.updater.__class__.__name__ == 'RNN':            
            hidden = hidden[0].unsqueeze(0)
            msg, new_hidden = self.updater(msg.unsqueeze(0), hidden)
            return [new_hidden.squeeze(0), None]
        elif self.updater.__class__.__name__ == 'LSTM':            
            hidden = (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0))
            msg, new_hidden = self.updater(msg.unsqueeze(0), hidden)
            return [new_hidden[0].squeeze(0), new_hidden[1].squeeze(0)]

class ClauseToLitLayer(BaseMessageLayer):
    def __init__(self, d_model, update_type='lstm'):
        self.d_model = d_model
        super().__init__(d_model, update_type)
        
        if update_type == 'lstm':
            self.updater = nn.LSTM(input_size=2*d_model, hidden_size=d_model)
        elif update_type == 'rnn':
            self.updater = nn.RNN(input_size=2*d_model, hidden_size=d_model)
        else:
            self.updater = nn.Sequential(
                nn.Linear(2*d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
            )

    def _flip_literals(self, x_l, l_batch):
        counts = torch.bincount(l_batch)
        starts = torch.cat([torch.tensor([0], device=x_l.device), torch.cumsum(counts[:-1], 0)])
        flipped = []
        for count, start in zip(counts, starts):
            n_vars = count // 2
            pos = x_l[start:start+n_vars]
            neg = x_l[start+n_vars:start+2*n_vars]
            flipped.append(torch.cat([neg, pos]))
        return torch.cat(flipped)

    def forward(self, adj_t, x_c, hidden, l_batch):
        msg = matmul(adj_t.t(), x_c)
        x_l = hidden[0]
        flipped = self._flip_literals(x_l, l_batch)
        msg = torch.cat([msg, flipped], dim=-1).unsqueeze(0)
        if self.updater.__class__.__name__ == 'RNN':
            hidden = x_l.unsqueeze(0)
            msg, new_hidden = self.updater(msg, hidden)
            new_hidden = [new_hidden[0].squeeze(0), None]
        elif self.updater.__class__.__name__ == 'LSTM':
            hidden = (x_l.unsqueeze(0), hidden[1].unsqueeze(0))
            msg, new_hidden = self.updater(msg, hidden)
            new_hidden = [new_hidden[0].squeeze(0), new_hidden[1].squeeze(0)]
        return new_hidden

class ClauseToVarLayer(BaseMessageLayer):
    def forward(self, adj_t, x_c, hidden, v_batch):
        msg = matmul(adj_t.t(), x_c)
        x_v = hidden[0]
        
        if self.updater.__class__.__name__ == 'RNN':
            hidden = x_v.unsqueeze(0)
            msg, new_hidden = self.updater(msg.unsqueeze(0), hidden)
            new_hidden = [new_hidden[0].squeeze(0), None]
        elif self.updater.__class__.__name__ == 'LSTM':
            hidden = (x_v.unsqueeze(0), hidden[1].unsqueeze(0))
            msg, new_hidden = self.updater(msg.unsqueeze(0), hidden)
            new_hidden = [new_hidden[0].squeeze(0), new_hidden[1].squeeze(0)]
        return new_hidden

class GNN_SAT(nn.Module):    
    def __init__(self, d_model, update_type='lstm', graph_type='lit', collect_embeddings=False):
        super().__init__()
        self.d_model = d_model
        self.graph_type = graph_type
        self.collect_embeddings = collect_embeddings
        
        if graph_type == 'lit':
            self.unk_to_clause = LitToClauseLayer(d_model, update_type)
            self.clause_to_unk = ClauseToLitLayer(d_model, update_type)
            self.get_x = lambda data: data.x_l
            self.get_batch = lambda data: data.x_l_batch
            self.get_adj = lambda data: data.adj_t_lit
        elif graph_type == 'var':
            self.unk_to_clause = VarToClauseLayer(d_model, update_type)
            self.clause_to_unk = ClauseToVarLayer(d_model, update_type)
            self.get_x = lambda data: data.x_v
            self.get_batch = lambda data: data.x_v_batch
            self.get_adj = lambda data: data.adj_t_var
        else:
            raise ValueError(f"Invalid graph type: {graph_type}")
         
        self.output = nn.Linear(d_model, 1)
        self.init_embeddings = nn.Linear(1, d_model)
        self.init_ts = torch.ones(1)
        self.init_ts.requires_grad = False
        self.L_init = nn.Linear(1, d_model)
        self.C_init = nn.Linear(1, d_model)

    def random_init_embeddings(self, data,device):
        n_unks, n_clauses = self.get_x(data).size(0), data.x_c.size(0)
        #initialize x_l and x_c
        init_ts = self.init_ts.to(device)
        x_unk = torch.rand((n_unks,self.d_model),requires_grad=False).to(device)
        C_init = self.C_init(init_ts)
        x_c = C_init.repeat(n_clauses, 1)
        # initialize lstm cell states
        x_l_h = torch.zeros(x_unk.shape).to(data.x_l.device)
        x_c_h = torch.zeros(x_c.shape).to(data.x_l.device)   
        unk_hidden = (x_unk, x_l_h)
        c_hidden = (x_c, x_c_h)    
        return unk_hidden, c_hidden

    def forward(self, data, num_iters):
        device = self.get_x(data).device
        unk_hidden, c_hidden = self.random_init_embeddings(data,device)        
        all_unk_votes = []
        all_unk_embeds = []
        all_c_embeds = []
        batch = self.get_batch(data)
        adj = self.get_adj(data)
        
        for _ in range(num_iters):
            c_hidden = self.unk_to_clause(adj, unk_hidden[0], c_hidden)
            unk_hidden = self.clause_to_unk(adj, c_hidden[0], unk_hidden, batch)
            if self.collect_embeddings:
                all_unk_embeds.append(unk_hidden[0])
                all_c_embeds.append(c_hidden[0])
                votes = self.output(unk_hidden[0])
                all_unk_votes.append(votes)

        votes = self.output(unk_hidden[0])
        vote_reduced = global_mean_pool(votes, batch)

        return {
            'vote_reduced': vote_reduced,
            'final_votes': votes,
            'all_votes': all_unk_votes,
            'final_embeds': unk_hidden[0],
            'all_unk_embeds': all_unk_embeds,
            'all_c_embeds': all_c_embeds
        }
