import pytest
import torch

from pycrystalgen.data.batching import Structures
from pycrystalgen.graph.atomic_radius import atomic_radius
from pycrystalgen.graph.sparse import scatter

def mix(x):
    x &= 0xffffffff
    x ^= x >> 16
    x = (x*0x85ebca6b)&0xffffffff
    x ^= x >> 13
    x = (x*0xc2b2ae35)&0xffffffff
    x ^= x >> 16
    x &= 0xffffffff
    return x

def hash_single(x):
    h1 = 0
    h2 = 0
    h3 = 0
    for i in x:
        h = mix(int(i.item()))
        h1 = (h1+h)&0xffffffff
        h2 = (h2+(h*h))&0xffffffff
        h3 = (h3+(h*h*h))&0xffffffff
    
    return mix(h1^h2^h3)

def test_scatter_hash():
    x0=torch.tensor([6,4,5,8,6,2,3,6,8])
    x1=torch.tensor([6,3,1,7,9,5])
    x2=torch.tensor([3,6,4,9,6,2,1,4])

    x= torch.cat([x0,x1,x2])
    batch = torch.cat([torch.full_like(x0,fill_value=0,dtype=torch.int64),torch.full_like(x1,fill_value=1,dtype=torch.int64),torch.full_like(x2,fill_value=2,dtype=torch.int64)])

    original_hash = scatter(x,batch,n_out=3,reduce="hash")

    perm = torch.randperm(x0.shape[0]+x1.shape[0]+x2.shape[0])
    x=x[perm]
    batch=batch[perm]

    permuted_hashed = scatter(x,batch,n_out=3,reduce="hash")

    assert (original_hash==permuted_hashed).all()
    
    assert original_hash[0]==hash_single(x0)
    assert original_hash[1]==hash_single(x1)
    assert original_hash[2]==hash_single(x2)


