#include "../stdafx.h"
#include "Neuron.h"

Neuron::Neuron()
{
	inputValue = 0;
	activatedValue = 0;
	derivedValue = 0;
}

Neuron::Neuron(double newValue)
{
	inputValue = newValue;
	ActivateFunction();
	DerivativeFunction();
}


Neuron::~Neuron()
{
}

void Neuron::SetValue(double newValue)
{
	inputValue = newValue;
	ActivateFunction();
	DerivativeFunction();
}

void Neuron::SetInputValue(double newValue)
{
	this->inputValue = newValue;
	this->activatedValue = newValue;
	this->derivedValue = newValue;
}

void Neuron::ActivateFunction()
{
	activatedValue = 1 / (1 + exp(-inputValue));
}

void Neuron::DerivativeFunction()
{
	derivedValue = activatedValue * (1 - activatedValue);
}
