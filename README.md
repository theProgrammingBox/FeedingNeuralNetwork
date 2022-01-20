# FeedingNeuralNetwork

## Purpose
Neural Networks are limited to the data given to them. This prevents them from exibiting more advanced behaviors that rely on memory. The Feeding Neural Network breaks through this with its newest inovation of feeding a partitioned number of outputs back into its input in the next iteration. This acts as a memory that may carry information from many iteration before the current iteration. The training does not guide the memory on what it should be but instead indirectly optimizes the memory values in order to decrease the cost of a sequence of iterations. In short, this is a neural network that feeds partitioned output values to the next iteration of partitioned input values.

## The Training Data
The current demo demonstrates an XOR function. The input is 5 bits (0 or 1) and the output is 5 bits. The expected output of the very first iteration is the very same input of the first iteration. Afterwards, the expected output is the XOR of the current input and previous input. Although this example may seem redundent because it is possible to just input both required inputs, it demonstrates the potential of memory retention of longer durations which can be used for audio restoration and behavior prediction.

## What Am I Seeing?
Currently, the network is trained to be stable after 10 iterations. You are given the option to either continue to train the network or to test it. If you choose to test the network, you are given the option to choose how many iterations to run the network. You can think of it as how long your brain remains active with all the memory inside it, except you are pausing/unpausing time for the agent. The network keeps its memory during the entire time the program is running. Even after 30,000 iterations, the network has an error of less then 0.001. If you choose to train it, you are shown the average error per output every training cycle. The training automaticly saves so feel free to exit the program anytime.

## Meet The Team
TheDukeVin & theProgrammingBox

## Problems With This Design
1. Can be a bit unstable with longer iterations and/or larger learning rates due to compounding changes.
2. It needs to store the entire sequence of networks to train properly so bigger networks and longer iterations may cause storage issues resulting in a crash.
