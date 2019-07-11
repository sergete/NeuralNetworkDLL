#include "../stdafx.h"
#include "Layer.h"

Layer::Layer(int size, int nextLayerSize, bool isInputLayer, bool weightRandomFill)
{
	this->layerSize = size;
	this->weightsListSize = nextLayerSize;
	this->isInputLayer = isInputLayer;

	//seed based in current time in seconds
	auto time = chrono::system_clock::now();
	double randomValue = chrono::system_clock::to_time_t(time);
	srand(randomValue);

	for (int i = 0; i < size; i++) {
		//Create all neurons of the layer
		Neurons.push_back(new Neuron(0));
		if (nextLayerSize != 0) {
			weightList.push_back(new vector<double>(nextLayerSize));
			//initialize the array where the new weights will be saved till we update the weights
			newWeightList.push_back(new vector<double>(nextLayerSize));
		}
		else {
			weightList = vector<vector<double> *>();
			newWeightList = vector<vector<double> *>();
		}

		//Create all the weights from the neuron i to the rest of neurons of the next layer
		for (int weightIndex = 0; weightIndex < nextLayerSize; weightIndex++) {
			double value = 0.00;
			if (weightRandomFill) {
				value = rand()/double(RAND_MAX);
			}
			weightList[i]->at(weightIndex) = value;
			newWeightList[i]->at(weightIndex) = -1;
		}
	}
}

Layer::~Layer()
{
}

/// <summary>
/// set value of the selected neuron
/// if is input layer we want that the value, activated value and derived value of the neuron was same
/// </summary>
/// <param name="index"></param>
/// <param name="newValue"></param>
void Layer::setValue(int index, double newValue) {

	if (isInputLayer) {
		Neurons[index]->SetInputValue(newValue);
	}
	else {
		Neurons[index]->SetValue(newValue);
	}
	
}

void Layer::setWeightValue(int neuronIndex, int nextLayerNeuronIndex, double value)
{
	weightList.at(neuronIndex)->at(nextLayerNeuronIndex) = value;
}

/// <summary>
/// sum all the weights of a selected index and returns this sum
/// </summary>
/// <param name="neuronIndex"></param>
/// <returns>the total sum of the weights to the next neuron</returns>
double Layer::GetTotalWeightToNextLayer(int neuronIndex)
{
	double totalSum = 0;

	for (int i = 0; i < this->weightList.size(); i++) {
		totalSum += (this->weightList.at(i)->at(neuronIndex) * this->Neurons.at(i)->getActivatedValue());
	}

	return totalSum;
}

void Layer::updateAllWeights()
{
	for (int neuron = 0; neuron < this->Neurons.size(); neuron++)
	{
		for (int weight = 0; weight < this->weightList.at(neuron)->size(); weight++)
		{
			double *newWeight = &this->newWeightList.at(neuron)->at(weight);
			double *oldWeight = &this->weightList.at(neuron)->at(weight);
			*oldWeight = *newWeight;
			*newWeight = -1;
		}
	}
}
