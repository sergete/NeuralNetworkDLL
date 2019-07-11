#ifndef _NEURALNETWORK_HPP_
#define _NEURALNETWORK_HPP_

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <assert.h>
#include <sstream>
#include "Layer.h"

using namespace std;

class NeuralNetwork
{
public:
	NeuralNetwork(vector<int> NN_topology);
	NeuralNetwork(const string RouteOfFile, const string Filename);
	NeuralNetwork(const string RouteOfFile, const string Filename, vector<int> NN_topology);
	virtual ~NeuralNetwork();

	inline void setLearningRate(double newLearningRate) { this->learningRate = newLearningRate; }
	inline double getLearningRate() { return this->learningRate; }
	inline int getOutputLayerSize() { return this->layers.at(layers.size()-1)->getLayerSize(); }
	inline int getInputLayerSize() { return this->layers.at(0)->getLayerSize(); }

	void feedForward();
	void backPropagation();

	void insertInputs(vector<double> inputs);
	void insertOutputs(vector<double> outputs);

	void printNN(string message = "Hello");

	//Functions to DLL
	void InitializeNeuralNetwork(const string routeToSaveFile,const string filename, vector<int> topology);
	void TrainNeuralNetwork(vector<double> inputs, vector<double> expectedOutputs);
	vector<double>* getPrediction(vector<double> inputs);
	void saveNN();
	void loadNN();
	bool FileExists();

private:
	//file variables
	string FILE_PATH;
	string FILENAME;
	//

	double learningRate;
	//vector that contains the number of neurons in every layer of the neural network
	vector<int> NN_topology;
	vector<double> inputLayer;
	vector<double> expectedOutputs;
	vector<double> predictionOutput;
	vector<Layer *> layers;

	void InitializeNeuralNetwork(vector<int> NN_topology);
	void updateAllWeights();
	void checkOrCreateFile();
	void loadData();
	void saveData();
	void clearData();
};

#endif //!_NEURALNETWORK_HPP_

