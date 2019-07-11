#pragma once

/// DLL created by SERGIO SANCHEZ-URAN LOPEZ for the final project degree in design and development of videogames
/// it´s a supervised neural network for that always need an array with the expected outputs
/// For the activation function it use the sigmoid 

#include <string>
#include <vector>
using namespace std;

#ifdef NEURALNETWORKDLL_EXPORTS
#define NEURALNETWORKDLL_API __declspec(dllexport)
#else
#define NEURALNETWORKDLL_API __declspec(dllimport)
#endif

namespace NeuralNetworkLib {

	/// <summary>
	/// Create a neural network if not file exist, by default if the user not introduce a vector with the topology it will create by default
	/// a neural network with 3 layers of size 1-1-1, but if exist its loaded from file
	/// </summary>
	extern "C" NEURALNETWORKDLL_API void LoadOrCreateNeuralNetwork(string Path, string filename, vector<int> *topology = new vector<int>(3, 1));

	/// <summary>
	/// Introduce the values to train your custom Neural Network, is a supervised Neural Network for that it needs an expected output array
	/// to make a correct back propagation
	/// </summary>
	extern "C" NEURALNETWORKDLL_API void Train(vector<double> *inputs, vector<double> *expectedOutputs);

	/// <summary>
	/// Get the prediction values of a trained neural network if this NN was not enough trained this outputs couldn´t be correct
	/// </summary>
	extern "C" NEURALNETWORKDLL_API double* GetPrediction(vector<double> &inputs);

	/// <summary>
	/// it saves the current values in the path and with the name selected when the neural network was created
	/// </summary>
	extern "C" NEURALNETWORKDLL_API void SaveNeuralNetwork();

	/// <summary>
	/// returns the size of the output layer
	/// </summary>
	extern "C" NEURALNETWORKDLL_API int getOutputLayerSize();

	/// <summary>
	/// returns the size of the input layer
	/// </summary>
	extern "C" NEURALNETWORKDLL_API int getInputLayerSize();

	/// <summary>
	/// Print the input values of the neurons, activated values and derived values of every neuron and his weights
	/// In the input layer, input value, activated value and derived values are the same
	/// </summary>
	extern "C" NEURALNETWORKDLL_API void printNN();

}