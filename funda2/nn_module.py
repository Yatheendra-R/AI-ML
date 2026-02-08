import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

"""
What is nn.Module

nn.Module is the base class for all neural network models in PyTorch.

It gives your model:
    a place to store parameters
    a place to define computation
    automatic gradient tracking
    easy device management

Think of it as a smart box that:
    knows what can be learned
    knows how data flows

Without nn.Module, PyTorch would not know:

    which tensors are trainable
    how to collect parameters
    how to move model to GPU
    how to switch train/eval modes
"""

"""
class MyModel(nn.Module): ##nn.module is parent of it , inheritance 
    def __init__(self):
        super().__init__()   # calls constructor of parent of MyModel, which is nn.Module

        ####
        initializes the parent nn.Modules
        activates parameter tracking
        enables .parameters(), .state_dict(), .to()


        A registry is just:

        “A place where PyTorch keeps track of important things.”

        Why does PyTorch need registries?

        Because it must answer questions like:
        What are my parameters?
        What submodules do I have?
        What tensors should move to GPU?
        What needs gradients?


        reates an internal registry:

        _parameters
        _modules
        _buffers

        Enables:
        model.parameters()
        model.state_dict()
        Allows optimizers to find parameters

        Without it :
        Your model exists
        But PyTorch cannot see what is trainable
        
        ####


        self.w = nn.Parameter(torch.randn(1))

        ###
        y=mx+c
        y=wx+b
        w and b are parameters
        They are not fixed
        Training = finding better values for them

        This tensor is a learnable parameter of the mode



        Parameter registry
        self._parameters = {
            "w": Parameter(...)
        }
        ###

        #self.w = torch.randn(1, requires_grad=True)   wrong way , it does not not tracked



    def forward(self, x):
        return x


    ###
    Defines how input transforms to output
    PyTorch builds a computation graph here
    You never call forward() directly

    y = w * x + b
    In PyTorch, computation: happens inside forward() -> builds a computation graph dynamically
            w ─┐
           ├─ (*) ─┐
        x ─┘       ├─ (+) ── y_pred ── loss
        b ─────────┘

    ###

model = MyModel()
list(model.parameters())
"""


"""
What is _parameters?

Inside every nn.Module there’s a dictionary:

self._parameters


Keys → names of parameters (strings)

Values → tensors wrapped as nn.Parameter

It’s like a registry shelf that says:

“These are the tensors that are trainable in my model.”


import torch
from torch import nn

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

model = LinearModel()



What is _parameters?

    Inside every nn.Module there’s a dictionary:

    self._parameters

    Keys → names of parameters (strings)
    Values → tensors wrapped as nn.Parameter

    It’s like a registry shelf that says: “These are the tensors that are trainable in my model.”

Example:
import torch
from torch import nn

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

model = LinearModel()

What happens internally?

When you do:  self.w = nn.Parameter(torch.randn(1))


PyTorch internally calls nn.Module.__setattr__()
Detects that the value is an nn.Parameter

Stores it in self._parameters dictionary:
self._parameters = {
    "w": Parameter(tensor([...], requires_grad=True))
}


Then: self.b = nn.Parameter(torch.randn(1))


Adds another entry:
self._parameters = {
    "w": Parameter(tensor([...], requires_grad=True)),
    "b": Parameter(tensor([...], requires_grad=True))
}


What exactly is a nn.Parameter object?

nn.Parameter is just a subclass of torch.Tensor with one flag:requires_grad = True


So when you see:Parameter(tensor([0.5]), requires_grad=True)


It means:

It behaves like a normal tensor
PyTorch knows it’s a learnable parameter
.grad will be populated after loss.backward()
"""
