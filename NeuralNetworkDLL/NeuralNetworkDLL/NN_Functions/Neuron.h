#ifndef _NEURON_HPP_
#define _NEURON_HPP_

#include <iostream>
#include <math.h>

class Neuron
{
public:
	Neuron();
	Neuron(double newValue);
	~Neuron();

	void SetValue(double newValue);
	void SetInputValue(double newValue);

	inline double getInputValue() { return this->inputValue; }
	inline double getActivatedValue() { return this->activatedValue; }
	inline double getDerivedValue() { return this->derivedValue; }

private:
	double inputValue;
	double activatedValue;
	double derivedValue;

	void ActivateFunction();
	void DerivativeFunction();
};

#endif // !_NEURON_HPP_
