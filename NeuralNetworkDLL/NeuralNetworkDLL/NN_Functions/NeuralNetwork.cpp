#include "../stdafx.h"
#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(vector<int> NN_topology) :FILE_PATH(""), FILENAME("Default.txt")
{
	if (NN_topology.size() < 3) {
		cerr << "Incorrect size of the neural network this must have at least 3 layers: Input layer, hidden layer and output layer" << endl;
		assert(false);
	}

	InitializeNeuralNetwork(NN_topology);
}

NeuralNetwork::NeuralNetwork(const string RouteOfFile, const string Filename) :FILE_PATH(RouteOfFile), FILENAME(Filename)
{
	vector<int> topology;
	topology.push_back(1);
	topology.push_back(1);
	topology.push_back(1);

	InitializeNeuralNetwork(topology);
}

NeuralNetwork::NeuralNetwork(const string RouteOfFile, const string Filename, vector<int> NN_topology):FILE_PATH(RouteOfFile),FILENAME(Filename)
{
	if (NN_topology.size() < 3) {
		cerr << "Incorrect size of the neural network this must have at least 3 layers: Input layer, hidden layer and output layer" << endl;
		assert(false);
	}

	InitializeNeuralNetwork(NN_topology);
}

/// <summary>
/// set the initial values of the neural networks and creates every layer with his layers associated
/// </summary>
/// <param name="NN_topology"></param>
void NeuralNetwork::InitializeNeuralNetwork(vector<int> NN_topology)
{
	this->NN_topology = NN_topology;
	//default learningRate
	this->learningRate = 0.05;

	//initialize all neural network with his layers and neurons
	for (int i = 0; i < NN_topology.size(); i++) {

		//if is layer 0 is input layer
		if (i == 0) {
			layers.push_back(new Layer(NN_topology.at(i), NN_topology.at(i + 1), true, true));
		}
		//is is last layer then is output layer
		else if (i == (NN_topology.size() - 1)) {
			layers.push_back(new Layer(NN_topology.at(i), 0, false, true));
		}
		else
			layers.push_back(new Layer(NN_topology.at(i), NN_topology.at(i + 1), false, true));
	}

	//printNN();
}

NeuralNetwork::~NeuralNetwork()
{
}

/// <summary>
/// Neural network feedForward method to forward propagation
/// it use sigmoid function to activate function
/// </summary>
/// <param name="inputLayer"></param>
void NeuralNetwork::feedForward()
{

	for (int layerIndex = 0; layerIndex < layers.size()-1; layerIndex++) {
		for (int nextNeuronIndex = 0; nextNeuronIndex < layers.at(layerIndex)->getWeightListSize(); nextNeuronIndex++) {
			//returns the total value of the weights in position nextNeuronValue multiplied by all neurons of this size
			double neuronInputValue = layers.at(layerIndex)->GetTotalWeightToNextLayer(nextNeuronIndex);
			//update the next layer neuron in position nextNeuronIndex with the value neuronInputValue
			layers.at(layerIndex + 1)->setValue(nextNeuronIndex, neuronInputValue);
		}
	}
}

/// <summary>
/// back propagation to update all weights of the neural network
/// </summary>
void NeuralNetwork::backPropagation()
{
	vector<double> errors;
	vector<double> nextLayerErrors;

	//from output layer to hidden layer
	for (int currentLayer = (layers.size() - 1); currentLayer > (layers.size() - 2); currentLayer--) {
		for (int outputIndex = 0; outputIndex < layers.at(currentLayer)->getLayerSize(); outputIndex++) {

			Neuron *neuron = layers.at(currentLayer)->getNeurons().at(outputIndex);

			//error calculous (expected output - current exit value)
			double error = this->expectedOutputs.at(outputIndex) - neuron->getActivatedValue();
			double derivedError = neuron->getDerivedValue() * error;
			//the error from output layer to layer - 1 
			errors.push_back(derivedError);

			//It saves the new weights of every neuron of the layer -1 corresponding with the index of the current exit neuron
			for (int i = 0; i < layers.at(currentLayer - 1)->getNeurons().size(); i++) {

				double newWeight = this->layers.at(currentLayer - 1)->getNeurons().at(i)->getActivatedValue() * errors.at(outputIndex);
				newWeight += this->layers.at(currentLayer - 1)->getWeights().at(i)->at(outputIndex);

				//add new weights to the layer - 1 and save it till we update all weights
				layers.at(currentLayer - 1)->addNewWeight(i, outputIndex, newWeight);
			}
		}
	}

	//from hidden layer to hidden layer or input layer
	for (int currentLayer = (layers.size() - 2); currentLayer > 0; currentLayer--) {

		//This help to calculate the index from the errors array to the matrix of weights
		int layerSize = (errors.size() / layers.at(currentLayer)->getLayerSize());
		if (layerSize <= 0)
			layerSize = 1;
		//

		for (int errorIndex = 0; errorIndex < errors.size(); errorIndex++) {

			//For every neuron
			for (int neuronIndex = 0; neuronIndex < layers.at(currentLayer)->getLayerSize(); neuronIndex++) {

				Neuron *neuron = layers.at(currentLayer)->getNeurons().at(neuronIndex);
				
				//translate errors position to weights position because weight position is a matrix and errors just a array				
				double errorIndexToWeightIndex = (neuronIndex % layerSize);
				double weightIndex = errorIndexToWeightIndex + ((neuronIndex / layerSize) * layerSize);
				//

				double error = neuron->getDerivedValue() * this->layers.at(currentLayer)->getWeights().at(neuronIndex)->at(errorIndexToWeightIndex)
								* errors.at(errorIndexToWeightIndex);

				//the error from layer to layer - 1 in case that we have more than 1 hidden layer
				nextLayerErrors.push_back(error);

				for (int i = 0; i < layers.at(currentLayer - 1)->getNeurons().size(); i++) 
				{
					//neuron of layer - 1
					Neuron *previewNeuron = layers.at(currentLayer - 1)->getNeurons().at(i);

					double derivedError = previewNeuron->getActivatedValue() * error;	
					double newWeight = this->layers.at(currentLayer - 1)->getWeights().at(i)->at(neuronIndex) + derivedError;

					//add new weights to the layer - 1 and save it till we update all weights
					layers.at(currentLayer - 1)->addNewWeight(i, neuronIndex, newWeight);
				}
			}

		}
		//Update the errors array to continue with back propagation
		errors.clear();
		errors = nextLayerErrors;
		nextLayerErrors.clear();
	}

	updateAllWeights();
}

/// <summary>
/// once time back propagation ends we have to set all new weights on every layer for every neuron
/// this calls the function of every layer to do it
/// </summary>
void NeuralNetwork::updateAllWeights()
{
	for (int currentLayer = 0; currentLayer < this->layers.size() - 1; currentLayer++) 
	{
		this->layers.at(currentLayer)->updateAllWeights();
	}
}

/// <summary>
/// insert neural networks input values and update the first layer with the entering inputs
/// </summary>
/// <param name="inputs"></param>
void NeuralNetwork::insertInputs(vector<double> inputs)
{
	if (inputs.size() != this->layers[0]->getLayerSize()) {
		cerr << "The inputs size is different of the layer 0 revise the topology or the inputs array" << endl;
		assert(false);
	}

	this->inputLayer = inputs;

	int firstLayersNeuronSize = layers.at(0)->getNeurons().size();

	for (int i = 0; i < firstLayersNeuronSize; i++) {
		layers.at(0)->setValue(i, this->inputLayer.at(i));
	}
}

void NeuralNetwork::insertOutputs(vector<double> outputs)
{
	if (outputs.size() != this->layers[this->layers.size()-1]->getLayerSize()) {
		cerr << "The outputs size is different of the output layer revise the topology or the outputs array" << endl;
		assert(false);
	}

	this->expectedOutputs = outputs;
}

void NeuralNetwork::printNN(string message)
{
	cout <<message << "---------------------------------------------------------------------------------------------"  << endl;
	for (int i = 0; i < layers.size(); i++) {

		cout << "LAYER: " << i << endl;

		for (int j = 0; j < layers.at(i)->getNeurons().size(); j++) {

			cout << "Value: " << layers.at(i)->getNeurons().at(j)->getInputValue() << +
				" ActivatedValue: " << layers.at(i)->getNeurons().at(j)->getActivatedValue() << +
				" DerivedValue: " << layers.at(i)->getNeurons().at(j)->getDerivedValue() << endl;
			//cout << layers.at(i)->getNeurons().at(j)->getInputValue() << endl;
			cout << "Weights: ";

			//for (int k = 0; k < layers.at(i)->getWeights().size(); k++) {
			for (int l = 0; l < layers.at(i)->getWeightListSize(); l++) {
				cout << layers.at(i)->getWeights().at(j)->at(l) << " , ";
			}
			//}
			cout << endl;
		}
		cout << endl;
	}
	cout << "---------------------------------------------------------------------------------------------" << endl;
}

/// <summary>
/// UNDER CONSTRUCTION IS A METHOD FOR DLL 
/// </summary>
/// <param name="routeToSaveFile"></param>
/// <param name="filename"></param>
/// <param name="topology"></param>
void NeuralNetwork::InitializeNeuralNetwork(const string routeToSaveFile, const string filename, vector<int> topology)
{
	//NeuralNetwork *NN = new NeuralNetwork(routeToSaveFile, filename, topology);
}

/// <summary>
/// Check if the file exists, if not exist create a new one
/// </summary>
/// <param name="NN"></param>
void NeuralNetwork::checkOrCreateFile()
{
	string name = FILE_PATH + FILENAME;
	fstream FileToWorkWith;
	FileToWorkWith.open(name, std::fstream::in | std::fstream::out);

	// If file does not exist, Create new file
	if (!FileToWorkWith)
	{
		cout << "Cannot open file, file does not exist. Creating new file.." << endl;
		cout << endl;

		FileToWorkWith.open(name, fstream::in | fstream::out | fstream::trunc);
		FileToWorkWith.close();
	}
	else {
		std::cout << "File exists, loaded from the specified route" << endl;
		cout << endl;
		FileToWorkWith.close();
	}

}

bool NeuralNetwork::FileExists()
{
	string name = FILE_PATH + FILENAME;
	fstream FileToWorkWith;
	FileToWorkWith.open(name, std::fstream::in | std::fstream::out);

	// If file does not exist, Create new file
	if (!FileToWorkWith)
	{		
		FileToWorkWith.close();
		return false;
	}
	else {
		FileToWorkWith.close();
		return true;
	}
}

/// <summary>
/// Load the neural network from the file selected and reinit it with the saved parameters
/// </summary>
void NeuralNetwork::loadData()
{
	string Path = FILE_PATH + FILENAME;
	string line = "";
	//array to save the data
	vector <vector <string> > data;

	ifstream myfile(Path);
	if (myfile.is_open())
	{
		while (getline(myfile, line))
		{
			//to break the string
			istringstream ss(line);
			vector<string> record;

			while (!ss.eof())
			{
				string tempLine;
				if (!getline(ss, tempLine, ' ')) break;
				record.push_back(tempLine);
				//cout << tempLine << '\n';
			}

			data.push_back(record);
		}
		myfile.close();
	}

	//reinit variables
	clearData();

	FILE_PATH = data.at(0).at(0);
	FILENAME = data.at(1).at(0);
	
	//topology set
	for (int i = 0; i < data.at(2).size(); i++) {
		int layerSize = stoi(data.at(2).at(i));
		this->NN_topology.push_back(layerSize);
	}

	InitializeNeuralNetwork(this->NN_topology);

	for (int i = 3; i < data.size(); i+=2) {
		// index of the neural network converted from string to int
		int layerIndex = stoi(data.at(i).at(1));
		int numNeurons = stoi(data.at(i).at(2));
		int numWeightsPerNeurons = stoi(data.at(i).at(3));
		//

		if (layerIndex == (this->NN_topology.size() - 1)) break;

		for (int neuronIndex = 0; neuronIndex < numNeurons; neuronIndex++) {
			for (int weightIndex = 0; weightIndex < numWeightsPerNeurons; weightIndex++) {
				//because array is linear and the array in layers is matrix need a conversion
				int index = numWeightsPerNeurons * neuronIndex + weightIndex;
				double weightValue = stod(data.at(i + 1).at(index));
				layers.at(layerIndex)->setWeightValue(neuronIndex, weightIndex, weightValue);
			}
		}
	}
}

/// <summary>
/// reinit all the variables of the neural network
/// </summary>
void NeuralNetwork::clearData() {
	//file variables
	FILE_PATH = "";
	FILENAME = "";
	//

	double learningRate = 0.05;
	//vector that contains the number of neurons in every layer of the neural network
	this->NN_topology.clear();
	this->inputLayer.clear();
	this->expectedOutputs.clear();
	this->layers.clear();
}

void NeuralNetwork::saveData()
{
	string PATH = FILE_PATH + FILENAME;
	ofstream out(PATH);

	string data = FILE_PATH + "\n";
	data += FILENAME + "\n";

	
	//topology of NN
	for (int i = 0; i < NN_topology.size(); i++) {
		data += to_string(NN_topology.at(i)) + " ";
	}
	data += "\n";

	for (int currentLayer = 0; currentLayer < layers.size(); currentLayer++) {
		data += "LAYER " + to_string(currentLayer) + " " + to_string(this->layers.at(currentLayer)->getLayerSize())+ " " +
			to_string(this->layers.at(currentLayer)->getWeightListSize()) + "\n";

		for (int neuronIndex = 0; neuronIndex < layers.at(currentLayer)->getLayerSize(); neuronIndex++) {
			for (int nextLayerNeuronIndex = 0; nextLayerNeuronIndex < layers.at(currentLayer)->getWeightListSize(); nextLayerNeuronIndex++) {
				data += to_string(this->layers.at(currentLayer)->getWeights().at(neuronIndex)->at(nextLayerNeuronIndex)) + " ";
			}
		}
		data += "\n";
	}

	out << data;
	out.close();

}

void NeuralNetwork::TrainNeuralNetwork(vector<double> inputs, vector<double> expectedOutputs)
{
	insertInputs(inputs);
	insertOutputs(expectedOutputs);
	feedForward();
	backPropagation();
}

vector<double>* NeuralNetwork::getPrediction(vector<double> inputs) {

	if (inputs.size() != this->layers[0]->getLayerSize()) {
		cerr << "The outputs size is different of the output layer revise the topology or the inputs array" << endl;
		assert(false);
	}

	predictionOutput.clear();
	expectedOutputs.clear();

	insertInputs(inputs);
	feedForward();

	for (int i = 0; i < this->layers[this->layers.size()-1]->getLayerSize(); i++) {
		double activatedValue = this->layers[this->layers.size() - 1]->getNeurons().at(i)->getActivatedValue();
		predictionOutput.push_back(activatedValue);
	}

	return &predictionOutput;
}

void NeuralNetwork::saveNN()
{
	checkOrCreateFile();
	saveData();
}

void NeuralNetwork::loadNN()
{
	loadData();
}
