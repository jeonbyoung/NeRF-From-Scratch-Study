import torch
import torch.nn as nn
import numpy as np



class Customed_NeRF(nn.Module):
    def __init__(self, pos_encd_dim = 6, dir_dim = 4, num_of_hidden_nodes = 128):
        super().__init__()

        self.pos_encd_dim = pos_encd_dim
        self.dir_dim = dir_dim

        input_dim = pos_encd_dim*2*3
        dir_dim = dir_dim*2*3

        self.box1 = nn.Sequential(nn.Linear(3+input_dim, num_of_hidden_nodes), nn.ReLU(),
                                  nn.Linear(num_of_hidden_nodes, num_of_hidden_nodes), nn.ReLU())
        
        self.box2 = nn.Sequential(nn.Linear(num_of_hidden_nodes+dir_dim, num_of_hidden_nodes // 2), nn.ReLU())
        self.density = nn.Sequential(nn.Linear(num_of_hidden_nodes, 1), nn.ReLU())

        self.box3 = nn.Sequential(nn.Linear(num_of_hidden_nodes // 2, 3), nn.Sigmoid())

        
        self.test_layer = nn.Sequential(nn.Linear(num_of_hidden_nodes, 3), nn.Sigmoid())

    @staticmethod
    def positional_encoding(pixel_coordinate,L):
        ret = []
        for i in range(L):
            ret.append(torch.sin(2**i*np.pi*pixel_coordinate))
            ret.append(torch.cos(2**i*np.pi*pixel_coordinate))

        return torch.cat(ret, dim=-1)
        
    def forward(self, pixel_coordinate, dir):
        pos_x = self.positional_encoding(pixel_coordinate, self.pos_encd_dim)
        pos_dir = self.positional_encoding(dir, self.dir_dim)

        pos_x = torch.cat([pixel_coordinate, pos_x], dim=-1)

        session1 = self.box1(pos_x)
        sigma = torch.relu(self.density(session1))

        x_dir = torch.cat([session1,pos_dir],dim=-1)

        session2 = self.box2(x_dir)

        rgb = self.box3(session2)

        return rgb, sigma
    
    


