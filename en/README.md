# BrainPy Introduction

In this chapter, we will briefly introduce how to implement computational neuroscience models with BrainPy. For more detailed documents and tutorials, please check our Github repository [BrainPy](https://github.com/PKU-NIP-Lab/BrainPy) and [BrainModels](https://github.com/PKU-NIP-Lab/BrainModels).

`BrainPy` is a Python platform for computational neuroscience and brain-inspired computation. To model with BrainPy, users should follow 3 steps:

1) Define Python classes for neuron and synapse models. BrainPy provides base classes for different kinds of models, users only need to inherit from those base classes, and define specific methods to tell BrainPy what operations they want the models to take during the simulation. In this process, BrainPy will assist users in the numerical integration of differential equations (ODE, SDE, etc.), adaptation of various backends (`Numpy`, `PyTorch`, etc.), and other functions to simplify code logic.

2) Instantiate Python classes as objects of neuron group and synapse connection groups, pass the instantiated objects to BrainPy class `Network`, and call method `run` to simulate the network.

3) Call BrainPy modules like the `measure` module and the `visualize` module to display the simulation results.

With this overall concept of BrainPy, we will go into more detail about implementations in the following sections. In neural systems, neurons are connected by synapses to build networks, so we will introduce [neuron models](neurons.md), [synapse models](synapses.md), and [network models](networks.md) in order.
