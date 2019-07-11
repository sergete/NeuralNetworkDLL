#ifndef _LAYER_HPP_
#define _LAYER_HPP_

#include <iostream>
#include <vector>
#include <math.h>
#include <random>
#include <chrono>
#include <ctime>
#include "Neuron.h"

using namespace std;

class Layer
{
public:
	Layer(int size, int nextLayerSize = 0,bool isInputLayer = false, bool randomFill = false);
	virtual ~Layer();

	void setValue(int index, double newValue);
	void setWeightValue(int neuronIndex, int nextLayerNeuronIndex, double value);
	double GetTotalWeightToNextLayer(int neuronIndex);

	inline vector<Neuron *> getNeurons() { return Neurons; }
	inline vector < vector<double> *> getWeights() { return weightList; }
	inline vector < vector<double> *> getNewWeights() { return newWeightList; }
	inline int getLayerSize() { return layerSize; }
	inline int getWeightListSize() { return weightsListSize; }
	//new weights to update the current weights methods
	inline void addNewWeight(int index, vector<double> *weightList) { newWeightList.at(index) = weightList; }
	inline void addNewWeight(int Neuronindex, int posIndex, double newWeight) { newWeightList.at(Neuronindex)->at(posIndex) = newWeight; }
	//Errors methods
	inline void AddErrorsAccumulated(double *value) { errorsAccumulated.push_back(value); }
	void updateAllWeights();

private:
	bool isInputLayer;
	int layerSize;
	int weightsListSize;
	vector<Neuron *> Neurons;
	vector<vector<double> *> weightList;
	vector<vector<double> *> newWeightList;
	//errors from layer + 1 to this layer in back progagation to calculate the new weight
	vector<double *> errorsAccumulated;
};
#endif // !_LAYER_HPP_
