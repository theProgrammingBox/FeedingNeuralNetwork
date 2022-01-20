#include <iostream>
#include <fstream>
#include <iomanip>
#include "RandAndTime.h"

#define STARTING_PRAM_RANGE 0.1		// starting range for random weights and bias
#define CHECK_IN_ITERATION 10000	// number of training cycles till evaluation and save
#define EVALUATION_ITERATION 100	// the number of evaluations ran to get average error

#define PRINT_PRECISION 6			// number of decimals shown
#define PROMPT_COL 17				// space for the print prompt
#define PRINT_COL 9					// space for every number displayed

#define LEARNING_RATE 0.002			// rate of which the network improvement is applies

#define SEQ_LENGTH 10				// how many iterations is the network running for
#define BATCH_SIZE 1				// number of trainings averaged together

#define NUM_LAYERS 2				// not including the input layer
#define NUM_INPUT_NODES 5			// self explanatory
#define NUM_OUTPUT_NODES 5			// self explanatory
#define NUM_FEEDING_NODES 6			// nodes passed into the next iteration
#define MAX_NODES_IN_LAYER 30		// number ceiling to define the array sizes

using namespace std;

const float EMPTY_FEEDING_VALUES[NUM_FEEDING_NODES]{};  // default values passed into the first iteration of feeding nodes

const int NODES_IN_LAYER[NUM_LAYERS + 1] =				// network structure
{
	NUM_INPUT_NODES + NUM_FEEDING_NODES,
	11,
	NUM_OUTPUT_NODES + NUM_FEEDING_NODES
};

void GenerateData(float* inputs, float* outputs)		// generates the inputs and expected outputs for the set number of iterations
{
	bool bits[NUM_INPUT_NODES]{};
	for (int iteration = 0; iteration < SEQ_LENGTH; iteration++)
	{
		for (int node = 0; node < NUM_INPUT_NODES; node++)
		{
			bool bit = rand() % 2;
			bits[node] ^= bit;
			inputs[iteration * NUM_INPUT_NODES + node] = bit;
			outputs[iteration * NUM_INPUT_NODES + node] = bits[node];
		}
	}
}

void PrintOutputOfIteration(int iteration, float* output)	// prints the output/input of an iteration
{
	for (int node = 0; node < NUM_OUTPUT_NODES; node++)
	{
		cout << setw(PRINT_COL) << output[iteration * NUM_OUTPUT_NODES + node] << " ";
	}
	cout << endl;
}

class NetworkParameters		// stores the weights and biases for the network
{
public:
	float weight[NUM_LAYERS][MAX_NODES_IN_LAYER][MAX_NODES_IN_LAYER]{};
	float bias[NUM_LAYERS][MAX_NODES_IN_LAYER]{};

	NetworkParameters()
	{
		Randomize();
	}

	void Randomize()		// randomized the weights and biases
	{
		int layer, childNode, parentNode;
		for (layer = 0; layer < NUM_LAYERS; layer++)
		{
			for (parentNode = 0; parentNode < NODES_IN_LAYER[layer + 1]; parentNode++)
			{
				bias[layer][parentNode] = DoubleRand() * STARTING_PRAM_RANGE;
				for (childNode = 0; childNode < NODES_IN_LAYER[layer]; childNode++)
				{
					weight[layer][childNode][parentNode] = DoubleRand() * STARTING_PRAM_RANGE;
				}
			}
		}
	}

	void ExportNetwork()	// exports the weights and biases to files
	{
		ofstream netOut1("Network.txt");
		int layer, parentNode, childNode;
		netOut1 << NUM_LAYERS << ' ';
		for (layer = 0; layer < NUM_LAYERS + 1; layer++)
		{
			netOut1 << NODES_IN_LAYER[layer] << ' ';
		}
		for (layer = 0; layer < NUM_LAYERS; layer++)
		{
			for (parentNode = 0; parentNode < NODES_IN_LAYER[layer + 1]; parentNode++)
			{
				netOut1 << bias[layer][parentNode] << ' ';
				for (childNode = 0; childNode < NODES_IN_LAYER[layer]; childNode++)
				{
					netOut1 << weight[layer][childNode][parentNode] << ' ';
				}
			}
		}
		netOut1.close();
		ofstream netOut2("NetworkBackup.txt");
		netOut2 << NUM_LAYERS << ' ';
		for (layer = 0; layer < NUM_LAYERS + 1; layer++)
		{
			netOut2 << NODES_IN_LAYER[layer] << ' ';
		}
		for (layer = 0; layer < NUM_LAYERS; layer++)
		{
			for (parentNode = 0; parentNode < NODES_IN_LAYER[layer + 1]; parentNode++)
			{
				netOut2 << bias[layer][parentNode] << ' ';
				for (childNode = 0; childNode < NODES_IN_LAYER[layer]; childNode++)
				{
					netOut2 << weight[layer][childNode][parentNode] << ' ';
				}
			}
		}
		netOut2.close();
	}

	void ImportNetwork()	// imports the weights and biases to files
	{
		ifstream netCheck1("Network.txt", ifstream::ate | ifstream::binary);
		ifstream netCheck2("NetworkBackup.txt", ifstream::ate | ifstream::binary);
		ifstream netIn;
		if (netCheck1.tellg() < netCheck2.tellg())
		{
			netIn.open("NetworkBackup.txt");
			cout << "NetworkBackup.txt was opened\n";
		}
		else
		{
			netIn.open("Network.txt");
			cout << "Network.txt was opened\n";
		}
		netCheck1.close();
		netCheck2.close();
		int numLayers, layer, parentNode, childNode;
		netIn >> numLayers;
		if (numLayers != NUM_LAYERS)
		{
			cout << "Different network structure detected\n";
			return;
		}
		for (layer = 0; layer < NUM_LAYERS + 1; layer++)
		{
			netIn >> parentNode;
			if (parentNode != NODES_IN_LAYER[layer])
			{
				cout << "Different network structure detected\n";
				return;
			}
		}
		for (layer = 0; layer < NUM_LAYERS; layer++)
		{
			for (parentNode = 0; parentNode < NODES_IN_LAYER[layer + 1]; parentNode++)
			{
				netIn >> bias[layer][parentNode];
				for (childNode = 0; childNode < NODES_IN_LAYER[layer]; childNode++)
				{
					netIn >> weight[layer][childNode][parentNode];
				}
			}
		}
		netIn.close();
	}
};

class Network							// holds the data necessary to run and solve the training of a network parameter
{
private:
	float ActivationFunction(float x)	// the activation function
	{
		if (x > 0) return x;
		return 0.1 * x;
	}

	float DActivationFunction(float x)	// the derivative of the activation function
	{
		if (x > 0) return 1;
		return 0.1;
	}

public:
	NetworkParameters* networkParameters;
	float preactivation[NUM_LAYERS][MAX_NODES_IN_LAYER]{};		// holds the sum of the previous layer's pass
	float activation[NUM_LAYERS + 1][MAX_NODES_IN_LAYER]{};		// holds the preactivation after the activation function is applied
	float dPreactivation[NUM_LAYERS][MAX_NODES_IN_LAYER]{};		// holds the effect of the node before the activation function to the cost
	float dActivation[NUM_LAYERS + 1][MAX_NODES_IN_LAYER]{};	// holds the effect of the node after the activation function to the cost

	Network()								// necessary for some reason
	{
	}

	Network(NetworkParameters* nParameters) // stores a network parameter to train on
	{
		networkParameters = nParameters;
	}

	void ForwardPropagate()					// runs the network parameter
	{
		int layer, parentNode, childNode;
		for (layer = 0; layer < NUM_LAYERS; layer++)
		{
			for (parentNode = 0; parentNode < NODES_IN_LAYER[layer + 1]; parentNode++)
			{
				preactivation[layer][parentNode] = networkParameters->bias[layer][parentNode];
				for (childNode = 0; childNode < NODES_IN_LAYER[layer]; childNode++)
				{
					preactivation[layer][parentNode] += networkParameters->weight[layer][childNode][parentNode] * activation[layer][childNode];
				}
				activation[layer + 1][parentNode] = ActivationFunction(preactivation[layer][parentNode]);
			}
		}
	}

	void BackPropagate()					// trains the network parameter
	{
		int layer, parentNode, childNode;
		for (layer = NUM_LAYERS - 1; layer >= 0; layer--)
		{
			for (parentNode = 0; parentNode < NODES_IN_LAYER[layer + 1]; parentNode++)
			{
				dPreactivation[layer][parentNode] = dActivation[layer + 1][parentNode] * DActivationFunction(preactivation[layer][parentNode]);
			}
			for (childNode = 0; childNode < NODES_IN_LAYER[layer]; childNode++)
			{
				dActivation[layer][childNode] = 0;
				for (parentNode = 0; parentNode < NODES_IN_LAYER[layer + 1]; parentNode++)
				{
					dActivation[layer][childNode] += dPreactivation[layer][parentNode] * networkParameters->weight[layer][childNode][parentNode];
				}
			}
		}
	}

	void PrintOutput()						// prints the output of the run
	{
		for (int node = 0; node < NUM_OUTPUT_NODES; node++)
		{
			cout << setw(PRINT_COL) << activation[NUM_LAYERS][node] << " ";
		}
		cout << endl;
	}
};

class NetworkTrainer								// trains an iteration of networks using its own output as input
{
private:
	NetworkParameters agentParameters;				// the one network parameter
	Network network[SEQ_LENGTH]{};					// list of networks under the one network parameter
	float dBias[NUM_LAYERS][MAX_NODES_IN_LAYER]{};	// stores the change to the biases
	float dWeight[NUM_LAYERS][MAX_NODES_IN_LAYER][MAX_NODES_IN_LAYER]{}; // stores the change to the weights

public:
	NetworkTrainer()								// creates a random network parameter to begin training
	{
		agentParameters = NetworkParameters();
	}

	void ImportNetwork()							// imports the network parameter
	{
		agentParameters.ImportNetwork();
	}

	void ExportNetwork()							// imports the network parameter
	{
		agentParameters.ExportNetwork();
	}

	void ForwardPropagate(float* inputs)			// runs the sequence of networks and carries over the partitioned outputs
	{
		network[0] = Network(&agentParameters);
		memcpy(network[0].activation[0], inputs, NUM_INPUT_NODES * sizeof(float));
		memcpy(network[0].activation[0] + NUM_INPUT_NODES, EMPTY_FEEDING_VALUES, NUM_FEEDING_NODES * sizeof(float));
		network[0].ForwardPropagate();

		for (int iteration = 1; iteration < SEQ_LENGTH; iteration++)
		{
			network[iteration] = Network(&agentParameters);
			memcpy(network[iteration].activation[0], inputs + iteration * NUM_INPUT_NODES, NUM_INPUT_NODES * sizeof(float));
			memcpy(network[iteration].activation[0] + NUM_INPUT_NODES, network[iteration - 1].activation[NUM_LAYERS] + NUM_OUTPUT_NODES, NUM_FEEDING_NODES * sizeof(float));
			network[iteration].ForwardPropagate();
		}
	}

	void BackPropagate(float* inputs, float* expected) // solves the affects of the nodes to the cost
	{
		ForwardPropagate(inputs);
		int iteration = SEQ_LENGTH - 1, layer, parentNode, childNode;
		for (int i = 0; i < NUM_OUTPUT_NODES; i++)
		{
			network[iteration].dActivation[NUM_LAYERS][i] = 2 * (network[iteration].activation[NUM_LAYERS][i] - expected[iteration * NUM_OUTPUT_NODES + i]);
		}
		memcpy(network[iteration].dActivation[NUM_LAYERS] + NUM_OUTPUT_NODES, EMPTY_FEEDING_VALUES, NUM_FEEDING_NODES * sizeof(float));
		network[iteration].BackPropagate();

		for (int iteration = SEQ_LENGTH - 2; iteration >= 0; iteration--)
		{
			for (int parentNode = 0; parentNode < NUM_OUTPUT_NODES; parentNode++)
			{
				network[iteration].dActivation[NUM_LAYERS][parentNode] = 2 * (network[iteration].activation[NUM_LAYERS][parentNode] - expected[iteration * NUM_OUTPUT_NODES + parentNode]);
			}
			memcpy(network[iteration].dActivation[NUM_LAYERS] + NUM_OUTPUT_NODES, EMPTY_FEEDING_VALUES, NUM_FEEDING_NODES * sizeof(float));
			memcpy(network[iteration].dActivation[NUM_LAYERS] + NUM_OUTPUT_NODES, network[iteration + 1].dActivation[0] + NUM_INPUT_NODES, NUM_FEEDING_NODES * sizeof(float));
			network[iteration].BackPropagate();
		}
		for (iteration = 0; iteration < SEQ_LENGTH; iteration++)
		{
			for (layer = 0; layer < NUM_LAYERS; layer++)
			{
				for (parentNode = 0; parentNode < NODES_IN_LAYER[layer + 1]; parentNode++)
				{
					dBias[layer][parentNode] += network[iteration].dPreactivation[layer][parentNode];
					for (childNode = 0; childNode < NODES_IN_LAYER[layer]; childNode++)
					{
						dWeight[layer][childNode][parentNode] += network[iteration].dPreactivation[layer][parentNode] * network[iteration].activation[layer][childNode];
					}
				}
			}
		}
	}

	void Update()		// applies the changes to the weights and biases
	{
		int layer, parentNode, childNode;
		for (layer = 0; layer < NUM_LAYERS; layer++)
		{
			for (parentNode = 0; parentNode < NODES_IN_LAYER[layer + 1]; parentNode++)
			{
				agentParameters.bias[layer][parentNode] -= dBias[layer][parentNode] * LEARNING_RATE / (BATCH_SIZE * SEQ_LENGTH);
				dBias[layer][parentNode] = 0;
				for (childNode = 0; childNode < NODES_IN_LAYER[layer]; childNode++)
				{
					agentParameters.weight[layer][childNode][parentNode] -= dWeight[layer][childNode][parentNode] * LEARNING_RATE / (BATCH_SIZE * SEQ_LENGTH);
					dWeight[layer][childNode][parentNode] = 0;
				}
			}
		}
	}

	void Evaluate()		// evaluates the average error per each output node
	{
		float input[NUM_INPUT_NODES * SEQ_LENGTH];
		float output[NUM_OUTPUT_NODES * SEQ_LENGTH];
		float error = 0;
		int round, iteration, node;
		for (round = 0; round < EVALUATION_ITERATION; round++)
		{
			GenerateData(input, output);
			ForwardPropagate(input);
			for (iteration = 0; iteration < SEQ_LENGTH; iteration++)
			{
				for (node = 0; node < NUM_OUTPUT_NODES; node++)
				{
					error += abs(network[iteration].activation[NUM_LAYERS][node] - output[iteration * NUM_OUTPUT_NODES + node]);
				}
			}
		}
		error /= EVALUATION_ITERATION * SEQ_LENGTH * NUM_OUTPUT_NODES;
		cout << "Average Error: " << error << endl;
		if (isnan(error))
		{
			cout << "irreversible damage detected, reverting\n";
			ImportNetwork();
		}
	}

	void PrintOutputOfIteration(int iteration)	// prints the output for a certain iteration of the networks
	{
		network[iteration].PrintOutput();
	}
};

int main()
{
	cout << setprecision(PRINT_PRECISION) << fixed;

	NetworkTrainer trainer;
	float input[NUM_INPUT_NODES * SEQ_LENGTH];
	float output[NUM_OUTPUT_NODES * SEQ_LENGTH];
	trainer.ImportNetwork();

	/*GenerateData(input, output);
	trainer.ForwardPropagate(input);
	for (int iteration = 0; iteration < SEQ_LENGTH; iteration++)
	{
		cout << "Iteration " << (iteration + 1) << endl;
		cout << left << setw(PROMPT_COL) << "AI Input: " << right;
		PrintOutputOfIteration(iteration, input);
		cout << left << setw(PROMPT_COL) << "AI Output: " << right;
		trainer.PrintOutputOfIteration(iteration);
		cout << left << setw(PROMPT_COL) << "Expected Output: " << right;
		PrintOutputOfIteration(iteration, output);
		cout << endl;
	}*/

	int iteration = 0;
	while (true)
	{
		for (int batch = 0; batch < BATCH_SIZE; batch++)
		{
			GenerateData(input, output);
			trainer.BackPropagate(input, output);
		}
		trainer.Update();
		if (++iteration == CHECK_IN_ITERATION)
		{
			iteration = 0;
			trainer.Evaluate();
			trainer.ExportNetwork();
		}
	}/**/

	cout << setprecision(6);
	cout.unsetf(ios::fixed);
	return 0;
}
