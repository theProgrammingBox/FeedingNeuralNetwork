#include <iostream>
#include <fstream>
#include <iomanip>
#include "RandAndTime.h"

#define NUM_TRAIN_ITERATIONS 1000000
#define NUM_EVAL_ITERATIONS 100
#define STARTING_PRAM_RANGE 0.1

#define PROMPT_COL 17
#define PRINT_COL 9
#define PRINT_PRECISION 6

#define MOMENTUM 0
#define LEARNING_RATE 0.001

#define SEQ_LENGTH 300
#define BATCH_SIZE 4

#define NUM_LAYERS 2
#define NUM_INPUT_NODES 5
#define NUM_OUTPUT_NODES 5
#define NUM_FEEDING_NODES 7
#define MAX_NODES_IN_LAYER 12

using namespace std;

const float EMPTY_FEEDING_VALUES[NUM_FEEDING_NODES]{};

const int NODES_IN_LAYER[NUM_LAYERS + 1] =
{
	NUM_INPUT_NODES + NUM_FEEDING_NODES,
	12,
	NUM_OUTPUT_NODES + NUM_FEEDING_NODES
};

void GenerateData(float* inputs, float* outputs)
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

void PrintOutputOfIteration(int iteration, float* output)
{
	for (int node = 0; node < NUM_OUTPUT_NODES; node++)
	{
		cout << setw(PRINT_COL) << output[iteration * NUM_OUTPUT_NODES + node] << " ";
	}
	cout << endl;
}

class Network {
private:
	float ActivationFunction(float x)
	{
		if (x > 0) return x;
		return 0.1 * x;
	}

	float DActivationFunction(float x)
	{
		if (x > 0) return 1;
		return 0.1;
	}

public:
	float weight[NUM_LAYERS][MAX_NODES_IN_LAYER][MAX_NODES_IN_LAYER]{};
	float bias[NUM_LAYERS][MAX_NODES_IN_LAYER]{};
	float preactivation[NUM_LAYERS][MAX_NODES_IN_LAYER]{};
	float activation[NUM_LAYERS + 1][MAX_NODES_IN_LAYER]{};
	float dPreactivation[NUM_LAYERS][MAX_NODES_IN_LAYER]{};
	float dActivation[NUM_LAYERS + 1][MAX_NODES_IN_LAYER]{};

	Network()
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

	void ForwardPropagate()
	{
		int layer, parentNode, childNode;
		for (layer = 0; layer < NUM_LAYERS; layer++)
		{
			for (parentNode = 0; parentNode < NODES_IN_LAYER[layer + 1]; parentNode++)
			{
				preactivation[layer][parentNode] = bias[layer][parentNode];
				for (childNode = 0; childNode < NODES_IN_LAYER[layer]; childNode++)
				{
					preactivation[layer][parentNode] += weight[layer][childNode][parentNode] * activation[layer][childNode];
				}
				activation[layer + 1][parentNode] = ActivationFunction(preactivation[layer][parentNode]);
			}
		}
	}

	void BackPropagate()
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
					dActivation[layer][childNode] += dPreactivation[layer][parentNode] * weight[layer][childNode][parentNode];
				}
			}
		}
	}

	void ExportNetwork()
	{
		ofstream netOut("Network.txt");
		int layer, parentNode, childNode;
		for (layer = 0; layer < NUM_LAYERS; layer++)
		{
			for (parentNode = 0; parentNode < NODES_IN_LAYER[layer + 1]; parentNode++)
			{
				netOut << bias[layer][parentNode] << ' ';
				for (childNode = 0; childNode < NODES_IN_LAYER[layer]; childNode++)
				{
					netOut << weight[layer][childNode][parentNode] << ' ';
				}
			}
		}
		netOut.close();
	}

	void ImportNetwork()
	{
		ifstream netIn("Network.txt");
		int layer, parentNode, childNode;
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

	void PrintOutput()
	{
		for (int node = 0; node < NUM_OUTPUT_NODES; node++)
		{
			cout << setw(PRINT_COL) << activation[NUM_LAYERS][node] << " ";
		}
		cout << endl;
	}
};

class NetworkTrainer {
private:
	Network agent = Network();
	Network network[SEQ_LENGTH]{};
	float dBias[NUM_LAYERS][MAX_NODES_IN_LAYER]{};
	float dWeight[NUM_LAYERS][MAX_NODES_IN_LAYER][MAX_NODES_IN_LAYER]{};

public:
	void ImportNetwork()
	{
		agent.ImportNetwork();
	}

	void ExportNetwork()
	{
		agent.ExportNetwork();
	}

	void ForwardPropagate(float* inputs)
	{
		network[0] = agent;
		memcpy(network[0].activation[0], inputs, NUM_INPUT_NODES * sizeof(float));
		memcpy(network[0].activation[0] + NUM_INPUT_NODES, EMPTY_FEEDING_VALUES, NUM_FEEDING_NODES * sizeof(float));
		network[0].ForwardPropagate();
		for (int iteration = 1; iteration < SEQ_LENGTH; iteration++)
		{
			network[iteration] = agent;
			memcpy(network[iteration].activation[0], inputs + iteration * NUM_INPUT_NODES, NUM_INPUT_NODES * sizeof(float));
			memcpy(network[iteration].activation[0] + NUM_INPUT_NODES, network[iteration - 1].activation[NUM_LAYERS] + NUM_OUTPUT_NODES, NUM_FEEDING_NODES * sizeof(float));
			network[iteration].ForwardPropagate();
		}
	}

	void BackPropagate(float* inputs, float* expected)
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

	void Update()
	{
		int layer, parentNode, childNode;
		for (layer = 0; layer < NUM_LAYERS; layer++)
		{
			for (parentNode = 0; parentNode < NODES_IN_LAYER[layer + 1]; parentNode++)
			{
				agent.bias[layer][parentNode] -= dBias[layer][parentNode] * LEARNING_RATE / BATCH_SIZE;
				dBias[layer][parentNode] *= MOMENTUM;
				for (childNode = 0; childNode < NODES_IN_LAYER[layer]; childNode++)
				{
					agent.weight[layer][childNode][parentNode] -= dWeight[layer][childNode][parentNode] * LEARNING_RATE / BATCH_SIZE;
					dWeight[layer][childNode][parentNode] *= MOMENTUM;
				}
			}
		}
	}

	void Evaluate()
	{
		float input[NUM_INPUT_NODES * SEQ_LENGTH];
		float output[NUM_OUTPUT_NODES * SEQ_LENGTH];
		float error = 0;
		int round, iteration, node;
		for (round = 0; round < NUM_EVAL_ITERATIONS; round++)
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
		cout << "Average Error: " << error / (NUM_EVAL_ITERATIONS * SEQ_LENGTH * NUM_OUTPUT_NODES) << endl;
	}

	void PrintOutputOfIteration(int iteration)
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

	GenerateData(input, output);
	trainer.ForwardPropagate(input);
	for (int iteration = 0; iteration < SEQ_LENGTH; iteration++)
	{
		cout << left << setw(PROMPT_COL) << "AI Input: " << right;
		PrintOutputOfIteration(iteration, input);
		cout << left << setw(PROMPT_COL) << "AI Output: " << right;
		trainer.PrintOutputOfIteration(iteration);
		cout << left << setw(PROMPT_COL) << "Expected Output: " << right;
		PrintOutputOfIteration(iteration, output);
		cout << endl;
	}/**/

	/*for (int iteration = 0; iteration < NUM_TRAIN_ITERATIONS; iteration++)
	{
		for (int batch = 0; batch < BATCH_SIZE; batch++)
		{
			GenerateData(input, output);
			trainer.BackPropagate(input, output);
		}
		trainer.Update();
		if (iteration % 10000 == 0)
		{
			trainer.Evaluate();
		}
	}
	trainer.ExportNetwork();*/

	cout << setprecision(6);
	cout.unsetf(ios::fixed);
	return 0;
}
