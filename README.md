# Perceptron

A perceptron is a simple mathematical model inspired by the way biological neurons receive signals and produce a response. This project explores a single-layer perceptron with two inputs and one binary output using the OR, AND, and XOR logic gates.

## Inspiration from biological neurons

Biological neurons receive signals through their dendrites. Those signals influence the electrical state of the cell, and when the conditions at the axon initial segment reach a firing threshold, the neuron produces an action potential that travels along its axon. Synapses influence how strongly one neuron affects another, and changes in synaptic strength are part of biological learning.

A perceptron simplifies this idea into inputs, weights, a bias, and an activation function:

![A perceptron neuron with inputs, weights, and an output](https://github.com/ZahidHasan/Perceptron/blob/master/img/perceptron_neuron.png?raw=true "Perceptron neuron")

| Biological concept | Perceptron counterpart |
|---|---|
| Incoming signals | Numerical inputs |
| Influence of synaptic connections | Input weights |
| Integration of incoming signals | Weighted sum |
| Firing threshold | Decision threshold, shifted by the bias |
| Neuron response | Binary output |
| Changes in connection strength during learning | Updates to the weights |

This connection is an inspiration, rather than a complete model of a real neuron. Biological neurons process signals that change over time and involve complex electrical and chemical mechanisms. A perceptron represents its decision with a weighted sum and a simple threshold.

## How the perceptron makes a decision

Each input is multiplied by its weight. The model adds these weighted inputs and a bias, then applies a step function to the result. A negative result produces an output of zero; a result of zero or greater produces an output of one.

The weights determine how each input influences the decision. A positive weight pushes the result toward one, while a negative weight pushes it toward zero. The bias shifts the decision threshold, allowing the model to change its output even when both inputs are zero.

![Schematic of a single-layer perceptron](https://sebastianraschka.com/images/blog/2015/singlelayer_neural_networks_files/perceptron_schematic.png "Single-layer perceptron")

The network in this project has two inputs connected directly to one output neuron. It has no hidden layer. Its decision boundary is a straight line that separates the two predicted classes in the input plane.

## How learning works

During training, the perceptron receives an input pair and its expected output. It makes a prediction and compares that prediction with the expected answer.

When the answer is correct, the weights and bias remain unchanged. When the answer is incorrect, the learning rule adjusts the weights according to the error and the input values, and adjusts the bias once for that sample. The learning rate controls the size of these changes.

One complete pass through the training samples is called an epoch. Repeating the samples gives the model opportunities to correct its decisions. The notebook uses a learning rate of 0.9 and runs for 10 epochs. Training uses a step function; the sigmoid shown during interactive testing does not take part in learning.

## Logic gates explored in this project

The training data contains all four possible pairs of binary inputs. Each gate assigns a different expected output to these pairs.

| First input | Second input | OR | AND | XOR |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | 0 | 1 |
| 1 | 0 | 1 | 0 | 1 |
| 1 | 1 | 1 | 1 | 0 |

**OR** produces one when at least one input is one. **AND** produces one only when both inputs are one. Both problems can be separated by a straight line, so a single-layer perceptron can learn them.

**XOR** produces one when the inputs are different. Its two classes occupy opposite corners of the input square, so no single straight line can separate them correctly. This perceptron cannot learn the entire XOR truth table, even if the number of training epochs is increased. A network with a suitable hidden layer and nonlinear activation, or an appropriate nonlinear transformation of the inputs, can overcome this limitation.

## Understanding the visualizations

The data plot shows the four input pairs, using different colors and markers for the expected output classes. The intermediate decision boundaries show how the separating line moves as the parameters change during training. The final boundary shows the model's decision after the last training step; it does not by itself guarantee that every sample is classified correctly.

The error plot records the difference between the expected answer and the prediction for each sample during training. Zero indicates a correct prediction. Positive one means the model predicted zero when the expected output was one; negative one means it predicted one when the expected output was zero. With four samples and ten epochs, a fresh training run records 40 such values.

## Exploring the notebook

Choose one logic gate, prepare its target outputs, and initialize the model before training. After training, inspect the decision boundary and errors, then test individual binary input pairs. Reset the parameters and error history before starting an independent experiment with another gate.

Comparing OR and AND with XOR demonstrates both the learning ability and the central limitation of a single-layer perceptron: it can learn a linear separation, but it cannot represent every possible relationship between inputs and outputs.
