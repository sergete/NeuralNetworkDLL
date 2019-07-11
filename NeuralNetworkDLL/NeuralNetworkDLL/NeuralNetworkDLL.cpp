// NeuralNetworkDLL.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "NN_Functions/NeuralNetwork.h"
#include "NeuralNetworkDLL.h"

//DLL variables
NeuralNetwork *NN = nullptr;

#pragma region NOT DLL FUNCTIONS

void printNotExistMessage() {
	cout << endl;
	cout << "You have to create a Neural Network before save it" << endl;
	cout << endl;
}

vector<double>* Prediction(vector<double> *inputs)
{
	return NN->getPrediction(*inputs);
}
#pragma endregion

#pragma region DLL FUNCTIONS

void NeuralNetworkLib::LoadOrCreateNeuralNetwork(string Path, string filename, vector<int> *topology)
{
	NN = new NeuralNetwork(Path, filename, *topology);

	bool fileExist = NN->FileExists();
	if (!fileExist) {
		NN->saveNN();
	}
	else {
		NN->loadNN();
	}
}

void NeuralNetworkLib::Train(vector<double> *inputs, vector<double> *expectedOutputs)
{
	NN->TrainNeuralNetwork(*inputs, *expectedOutputs);
}

double* NeuralNetworkLib::GetPrediction(vector<double> &inputs)
{
	if (NN == nullptr) {
		printNotExistMessage();
		return nullptr;
	}

	vector<double> *outs = Prediction(&inputs);
	double *outArray = &outs->at(0);

	return outArray;
}

void NeuralNetworkLib::SaveNeuralNetwork()
{
	if (NN != nullptr)
		NN->saveNN();
	else {
		printNotExistMessage();
	}
}

int NeuralNetworkLib::getOutputLayerSize()
{
	if (NN != nullptr)
		return NN->getOutputLayerSize();
	else {
		printNotExistMessage();
	}
}

int NeuralNetworkLib::getInputLayerSize()
{
	if (NN != nullptr)
		return NN->getInputLayerSize();
	else {
		printNotExistMessage();
	}
}

void NeuralNetworkLib::printNN()
{
	if (NN != nullptr) {
		cout << endl;
		NN->printNN("Neural Network");
		cout << endl;
	}
	else {
		printNotExistMessage();
	}
}

#pragma endregion
