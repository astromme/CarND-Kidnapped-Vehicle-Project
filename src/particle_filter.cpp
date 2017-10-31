/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <unordered_map>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	num_particles = 30;
	weights = std::vector<double>(num_particles);
	particles = std::vector<Particle>(num_particles);

	gen = default_random_engine();

	// This line creates a normal (Gaussian) distribution for x
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; ++i) {
		double sample_x, sample_y, sample_theta;
		 sample_x = dist_x(gen);
		 sample_y = dist_y(gen);
		 sample_theta = dist_theta(gen);

		 weights[i] = 1.0;
		 Particle p = Particle();
		 p.id = i;
		 p.x = sample_x;
		 p.y = sample_y;
		 p.theta = sample_theta;
		 p.weight = weights[i];

		 particles[i] = p;
		 //cout << "particle init: " << particles[i].x << "," << particles[i].y  << "," << particles[i].theta << "," << particles[i].weight << "," << particles[i].theta << endl;

	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);


  bool yaw_dot_is_zero = fabs(yaw_rate) < 1e-10;
	double c1;
	if (!yaw_dot_is_zero) {
		c1 = velocity / yaw_rate;
	}

	for (int i=0; i< num_particles; i++) {
		double x_0 = particles[i].x;
		double y_0 = particles[i].y;
		double theta_0 = particles[i].theta;
		double yaw_delta_t = yaw_rate * delta_t;

		//cout << "prediction: old location " << x_0 << "," << y_0  << "," << theta_0 << "," << particles[i].weight << endl;

		if (yaw_dot_is_zero) {
			cout << "yaw_dot_is_zero";
			particles[i].x = x_0 + velocity * delta_t * cos(theta_0) + dist_x(gen);
			particles[i].y = y_0 + velocity * delta_t * sin(theta_0) + dist_y(gen);
			particles[i].theta = theta_0 + dist_theta(gen);
		} else {
			particles[i].x = x_0 + c1 * (sin(theta_0 + yaw_delta_t) - sin(theta_0)) + dist_x(gen);
			particles[i].y = y_0 + c1 * (cos(theta_0) - cos(theta_0 + yaw_delta_t)) + dist_y(gen);
			particles[i].theta = theta_0 + yaw_delta_t + dist_theta(gen);
		}



		// particles[i].x = x_0 + c1 * (sin(theta_0 + yaw_delta_t) - sin(theta_0));// + dist_x(gen);
		// particles[i].y = y_0 + c1 * (cos(theta_0) - cos(theta_0 + yaw_delta_t));// + dist_y(gen);
		// particles[i].theta = theta_0 + yaw_delta_t;// + dist_theta(gen);

		//cout << "prediction: new location " << particles[i].x << "," << particles[i].y  << "," << particles[i].theta << endl;

	}
}

double ParticleFilter::distance(LandmarkObs o1, LandmarkObs o2) {
	return sqrt(pow(o2.x-o1.x, 2) + pow(o2.y-o1.y, 2));
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	if (predicted.size() == 0 || observations.size() == 0) {
		cout << "predicted or observations are zero" << endl;
		return;
	}

	for (int o=0; o<observations.size(); o++) {
		// cout << "obs x,y " << observations[o].x << "," << observations[o].y << endl;

		double closest_distance = distance(predicted[0], observations[o]);
		observations[o].id = predicted[0].id;

		// cout << "initialized closest_distance as " << closest_distance << endl;

		for (int p=1; p<predicted.size(); p++) {
			double current_distance = distance(predicted[p], observations[o]);
			// cout << "considering particle " << predicted[p].x << "," << predicted[p].y << " with distance " << current_distance << "," << closest_distance << endl;
			if (current_distance < closest_distance) {
				closest_distance = current_distance;
				observations[o].id = predicted[p].id;
				// cout << "mu  x,y " << predicted[p].x << "," << predicted[p].y << " with distance " << current_distance << endl;
				// cout << "matching with " << predicted[p].id << endl;
			}
		}
		//cout << endl;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// calculate normalization term
	const double sig_x = std_landmark[0];
	const double sig_y = std_landmark[1];
	const double gauss_norm = (1/(2 * M_PI * sig_x * sig_y));
	const double sig_xx_2 = 2 * pow(sig_x, 2);
	const double sig_yy_2 = 2 * pow(sig_y, 2);

	//cout << "gauss_norm " << gauss_norm << endl;

	for (int p=0; p<particles.size(); p++) {

		const Particle particle = particles[p];
		// cout << "updateWeights-particle: " << particle.x << "," << particle.y  << "," << particle.theta << "," << particle.weight << endl;

		const double cos_theta = cos(particle.theta);
		const double sin_theta = sin(particle.theta);

		// convert each observation into map coordinates
		// //cout << "---------------------Obs Transformations------------------------" << endl;
		std::vector<LandmarkObs> observations_map_coordinate_system;
		for (int o=0; o<observations.size(); o++) {
			LandmarkObs observation = observations[o];
			LandmarkObs mapObservation;

			//cout << "obs(" << observation.x << "," << observation.y << ")-->mapObs(";

			mapObservation.x = particle.x + cos_theta * observation.x - sin_theta * observation.y;
			mapObservation.y = particle.y + sin_theta * observation.x + cos_theta * observation.y;

			//cout << mapObservation.x << "," << mapObservation	.y << ")" << endl;

			observations_map_coordinate_system.push_back(mapObservation);
		}

		//cout << "landarmks" << endl;
		std::vector<LandmarkObs> predicted;
		std::unordered_map<int, LandmarkObs> predicted_dict;
		for (int m=0; m<map_landmarks.landmark_list.size(); m++) {
			LandmarkObs prediction;

			prediction.id = map_landmarks.landmark_list[m].id_i;
			prediction.x = map_landmarks.landmark_list[m].x_f;
			prediction.y = map_landmarks.landmark_list[m].y_f;

			//cout << prediction.x << "," << prediction.y << endl;

			if (dist(prediction.x, prediction.y, particle.x, particle.y) > sensor_range) {
				//cout << "dropping landmark, too far from sensor" << endl;
				continue;
			}

			predicted.push_back(prediction);
			predicted_dict[prediction.id] = prediction;
		}

		dataAssociation(predicted, observations_map_coordinate_system);

		//cout << "---------------------Assosciations------------------------" << endl;
		for (int o=0; o<observations_map_coordinate_system.size(); o++) {
			const LandmarkObs obs = observations_map_coordinate_system[o];
			const LandmarkObs prediction = predicted_dict[obs.id];

			//cout << "mapObs(" << obs.x << "," << obs.y << "); Predicted(" << prediction.x << "," << prediction.y << ") distance: " << distance(obs, prediction) << endl;
		}


		//cout << "---------------------Weights Calc------------------------" << endl;

		std::vector<int> associations(observations_map_coordinate_system.size());
		std::vector<double> sense_x(observations_map_coordinate_system.size());
		std::vector<double> sense_y(observations_map_coordinate_system.size());
		std::vector<double> new_weights;
		//cout << "transformed" << endl;
		for (int o=0; o<observations_map_coordinate_system.size(); o++) {
			const LandmarkObs obs = observations_map_coordinate_system[o];
			const LandmarkObs prediction = predicted_dict[obs.id];
			const double x_obs = obs.x;
			const double y_obs = obs.y;
			const double mu_x = prediction.x;
			const double mu_y = prediction.y;
			//cout << x_obs << "," << y_obs << endl;
			//cout << "obs x,y " << x_obs << "," << y_obs << " with id " << prediction.id << endl;
			//cout << "mu  x,y " << mu_x << "," << mu_y << endl << endl;

			//cout << "mapObs(" << obs.x << "," << obs.y << "); Predicted(" << prediction.x << "," << prediction.y << ") distance: " << distance(obs, prediction) << endl;

			associations[o] = observations_map_coordinate_system[o].id;
			sense_x[o] = x_obs;
			sense_y[o] = y_obs;

			// calculate exponent
			const double exponent = (pow(x_obs - mu_x, 2))/sig_xx_2 + (pow(y_obs - mu_y, 2))/sig_yy_2;
			//cout << "exp: " << exponent << endl;

			// calculate weight using normalization terms and exponent
			const double weight_partial = gauss_norm * exp(-exponent);
			//cout << "weight partial " << weight_partial << endl;
			new_weights.push_back(weight_partial);
		}

		// for (auto i = new_weights.begin(); i != new_weights.end(); ++i)
	  //   std::cout << *i << ' ';
		// cout << endl;

		double weight = std::accumulate(new_weights.begin(), new_weights.end(), 1.0, std::multiplies<double>());

		SetAssociations(particle, associations, sense_x, sense_y);
		// for (auto i = associations.begin(); i != associations.end(); ++i)
	  //   std::cout << *i << ' ';
		// cout << endl;
		particles[p].weight = weight;
		weights[p] = weight;
		//cout << "weight: " << weight << endl << endl;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::discrete_distribution<int> d(weights.begin(), weights.end());
	std::vector<Particle> new_particles(num_particles);
	for (int i=0; i<num_particles; i++) {
		int id = d(gen);
		new_particles[i] = particles[id];
	}
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
