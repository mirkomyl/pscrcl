/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#include <iostream>
#include <limits>

#include "Optimizer.h"

#define OBJEVAL 3

using namespace pscrCL;

const std::string errorLocation = errorMsgBegin + " / ";
const std::string debugLocation = debugMsgBegin + " / ";

void Optimizer::addParam(const OptParam& _param) {

	// Check for duplicated parameter ids
	if(params.find(_param.getId()) != params.end()) {
		throw DuplicatedIdException(_param.getId());
	}

	params.insert(std::pair<pscrCL_opt_id, OptParam>(_param.getId(), _param));

#if FULL_DEBUG
//	std::cout << debugLocation <<
//			"Optimizer::addParam: New parameter added, " <<
//			toString(_param) << std::endl;
#endif
}

void Optimizer::addParams(const Optimizer &optimizer) {
#if FULL_DEBUG
	std::cout << debugLocation <<
			"Optimizer::addParam: Adding " << optimizer.params.size() << 
			" parameters..." << std::endl;
#endif
	
	typename ParamContainer::const_iterator it;
	for(it =  optimizer.params.begin(); it != optimizer.params.end(); it++)
		addParam(it->second);
}

int Optimizer::interped(pscrCL_opt_id _id, const OptValues& _values) const {
	ParamContainer::const_iterator it = params.find(_id);

	if(it != params.end())
		return it->second.interpret(_values(_id));

	throw InvalidIdException(_id);
}

int Optimizer::revInterped(pscrCL_opt_id _id, int _revValue) const {
	ParamContainer::const_iterator it = params.find(_id);

	if(it != params.end())
		return it->second.revInterpret(_revValue);

	throw InvalidIdException(_id);
}

OptValues Optimizer::getDefaultValues() const {
	OptValues values;

	ParamContainer::const_iterator it;
	for(it = params.begin(); it != params.end(); it++)
		values.setValue(it->first, it->second.getDefaultInternalValue());

	return values;
}

OptValues Optimizer::getOptimizedValues(
		AbstractOptimizerHelper& _helper) const {

	// Differential evolution parameters
	const int N = 30;
	const int maxValidIEval = 200;
	const int maxIter = 10000;
	const float initCR = 0.2;
	const float CR = 0.5;
	const float mutCR = 0.05;
	const float F = 1.0;

	// Initialization
	
#if FULL_DEBUG
	std::cout << debugLocation <<
		"Optimizer::getOptimizedValues: Initializing the helper..." <<
		std::endl;
#endif
	
	_helper.initialize();

#if FULL_DEBUG
	std::cout << debugLocation <<
		"Optimizer::getOptimizedValues: Creating initial population..." <<
		std::endl;
#endif
	
	std::vector<std::pair<OptValues, float> > population;

	OptValues defaultValues = getDefaultValues();

#if FULL_DEBUG
	std::cout << debugLocation <<
		"Optimizer::getOptimizedValues: Creating a new individual with " <<
		"params: " << std::endl << getString(defaultValues);
#endif

	population.push_back(
		std::pair<OptValues, float>(
				defaultValues,
				objectFunction(_helper, defaultValues)));
	
#if FULL_DEBUG
	std::cout << debugLocation <<
		"Optimizer::getOptimizedValues: Created a new individual with " <<
		"fitness " << population.back().second << "." << std::endl;
#endif

	for(int i = 1; i < N; i++) {
		OptValues x = defaultValues;
		for(int j = 0; j < x.size(); j++) {
			float r = RandF<float>::i(0.0, 1.0);
			if(r < initCR)
				x[j] += RandInt::i(-5, 5);
		}
			
#if FULL_DEBUG
		std::cout << debugLocation <<
			"Optimizer::getOptimizedValues: Creating a new individual with " <<
			"params: " << std::endl << getString(x);
#endif
	
		float time = objectFunction(_helper, x);
	
#if FULL_DEBUG
		std::cout << debugLocation <<
			"Optimizer::getOptimizedValues: Created a new individual " <<
			"with fitness " << time << "." << std::endl;
#endif
			
		population.push_back(std::pair<OptValues, float>(x, time));
	}

#if FULL_DEBUG
	std::cout << debugLocation <<
		"Optimizer::getOptimizedValues: The main iteration begins..." <<
		std::endl;
#endif
	
	// Main iteration
	int validEval = 0;
	for(int iter = 0; validEval < maxValidIEval && iter < maxIter; iter++) {
		
#if DEBUG
	std::cout << debugLocation <<
		"Optimizer::getOptimizedValues: Iteration " << toString(iter) << ", " <<
		"valid evaluations = " << validEval << std::endl;
#endif

		for(int x_idx = 0; x_idx < (int) population.size(); x_idx++) {
		
			// Select a, b and c
			int a_idx = RandInt::i(0, population.size()-1);
			while(a_idx == x_idx) 
				a_idx = RandInt::i(0, population.size()-1);
			int b_idx = RandInt::i(0, population.size()-1);
			while(b_idx == x_idx || b_idx == a_idx) 
				b_idx = RandInt::i(0, population.size()-1);
			int c_idx = RandInt::i(0, population.size()-1);
			while(c_idx == x_idx || c_idx == a_idx || c_idx == b_idx)
				c_idx = RandInt::i(0, population.size()-1);

			OptValues& x = population[x_idx].first;
			OptValues& a = population[a_idx].first;
			OptValues& b = population[b_idx].first;
			OptValues& c = population[c_idx].first;

			int R = RandInt::i(0, x.size()-1);
			OptValues y = x;
			for(int i = 0; i < (int) y.size(); i++) {
				float r = RandF<float>::i(0.0, 1.0);
				if(r < CR || i == R) {
					y[i] = a[i] + F * (b[i] - c[i]);
				}
				float s = RandF<float>::i(0.0, 1.0);
				if(s < mutCR)
					y[i] += RandInt::i(-5, 5);
			}

#if FULL_DEBUG
		std::cout << debugLocation <<
			"Optimizer::getOptimizedValues: Creating a new individual with " <<
			"params: " << std::endl << getString(y);
#endif
			
			// Check if f(y) < f(x)
			float time = objectFunction(_helper, y);
			if(time < population[x_idx].second)
				population[x_idx] = std::pair<OptValues, float>(y, time);
			
#if FULL_DEBUG
		std::cout << debugLocation <<
			"Optimizer::getOptimizedValues: Created a new individual with " <<
			"fitness " << time << "." << std::endl;
#endif
			
			if(time < 1.0/0.0)
				validEval++;
		}
	}

	_helper.release();

	// Pick the agent that has the lowest cost
	OptValues best = population[0].first;
	float bestTime = population[0].second;

	for(int i = 1; i < (int) population.size(); i++) {
		if(population[i].second < bestTime) {
			best = population[i].first;
			bestTime = population[i].second;
		}
	}

#if DEBUG
	std::cout << debugLocation <<
		"Optimizer::getOptimizedValues: Found the best individual with " <<
		"fitness " << bestTime << ". Params: " << std::endl << getString(best);
#endif
		
	return best;
}

std::string Optimizer::getCompilerArgs(const OptValues& _values) const {
	std::string args = "";

	ParamContainer::const_iterator it;
	for(it = params.begin(); it != params.end(); it++) {
		if(it->second.hasClName()) {
			args += " -D " + it->second.getClName() +
					"=" + toString(interped(it->first, _values));
		}
	}

	return args;
}

std::string Optimizer::getString(const OptValues& _values) const {
	std::string str = "";

	ParamContainer::const_iterator it;
	for(it = params.begin(); it != params.end(); it++) {
		str += "    " + it->second.getName() + " = " +
				toString(interped(it->first, _values)) + "\n";
	}

	return str;
}

Optimizer Optimizer::operator+(const Optimizer& a) const {
	Optimizer ret;
	ret.addParams(*this);
	ret.addParams(a);
	return ret;
}

float Optimizer::objectFunction(
		AbstractOptimizerHelper& _helper,
		const OptValues& _values) const {

	float time = 0.0;
	try {
		if(!_helper.prepare(_values))
			return 1.0/0.0;

		_helper.finalize();

		for(int i = 0; i < OBJEVAL; i++) {
			Timer timer;
			timer.begin();
			_helper.run();
			_helper.finalize();
			timer.end();
			time += timer.getTime();
		}
	} catch(OpenCLError &err) {
		_helper.recover();
		return 1.0/0.0;
	}

	return time/OBJEVAL;
}