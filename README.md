# FeedingNeuralNetwork

## Purpose
Neural Networks are limited to the data given to them. This prevents them from exibiting more advanced behaviors that rely on memory. The Feeding Neural Network breaks through this with its newest inovation of feeding a partitioned number of outputs back into its input in the next iteration. This acts as a memory that may carry information from many iteration before the current iteration. The training does not guide the memory on what it should be but instead indirectly optimizes the memory values in order to decrease the cost of a sequence of iterations. In short, this is a neural network that feeds partitioned output values to the next iteration of partitioned input values.

## The Training Data
The current demo demonstrates an XOR function. The input is 5 bits (0 or 1) and the output is 5 bits. The expected output of the very first iteration is the very same input of the first iteration. Afterwards, the expected output is the XOR of the current input and previous input. Although this example may seem redundent because it is possible to just input both required inputs, it demonstrates the potential of memory retention of longer durations which can be used for audio restoration and behavior prediction.

## What Am I Seeing?
Currently, the network is trained to be stable after 10 iterations. We are currently running the network for 100 iterations withought training to demonstrate that the output is still stable with a very small error of around ±0.0005.

## Meet The Team
TheDukeVin & theProgrammingBox

## Problems With This Design
1. Curerrently still unstable with longer iterations and/or larger learning rates due to compounding changes.
2. The current training method requires a lot of storage so bigger networks and longer iterations will cause a crash.

## Ideas To Counteract Problems With This Design
1. Divide change by number of iterations
2. Larger batch size
3. Subdividing sequence training (probably not going to work)
4. optimizing storage like only having one weights and bias array in training
5. Storing data in heap or in seperate files
