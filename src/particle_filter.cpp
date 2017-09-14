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

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	if(!is_initialized)
	{
		// Define number of particles.
		num_particles = 100;

		// Define random engine "gen"
		default_random_engine gen;

		// Define normal distribution for sensors
		normal_distribution<double> num_x(x, std[0]); // Initialize x
		normal_distribution<double> num_y(y, std[1]); // Initialize y
		normal_distribution<double> num_theta(theta, std[2]); // Initialize theta

		//weights.clear();
		//particles.clear();

		// Initialize Particles
		for (int i = 0; i < num_particles; i++)
		{
			Particle P_;
			P_.id = i;
			P_.x = num_x(gen);
			P_.y = num_y(gen);
			P_.theta = num_theta(gen);
			P_.weight = 1.0;
			particles.push_back(P_);
			weights.push_back(1);
		}

		is_initialized = true;
	}

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// Define random engine "gen"
	default_random_engine gen;

	for (int i = 0; i < num_particles; i++)
	{
		// Predict New State
		if (fabs(yaw_rate) < 0.0001) {
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}
		else {
			particles[i].x += velocity/yaw_rate * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
			particles[i].y += velocity/yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
			particles[i].theta += yaw_rate * delta_t;
		}
		
		// Define normal distribution for sensors.
		normal_distribution<double> num_x(particles[i].x , std_pos[0]);
		normal_distribution<double> num_y(particles[i].y , std_pos[1]);
		normal_distribution<double> num_theta(particles[i].theta , std_pos[2]);

		particles[i].x = num_x(gen);
		particles[i].y = num_y(gen);
		particles[i].theta = num_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int i = 0; i < observations.size(); i++)
	{
		double min_d = numeric_limits<double>::max();

		for (int k = 0; k < predicted.size(); k++)
		{
			double dista_ = dist(observations[i].x, observations[i].y, predicted[k].x, predicted[k].y);
			if (dista_ < min_d)
			{
				min_d = dista_;
				observations[i].id = predicted[k].id;
			}
		}
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
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

	weights.clear();
	for (int i = 0; i < num_particles; i++)
	{
		vector<LandmarkObs> t_observations;
		for (int j = 0; j < observations.size(); j++)
		{
			LandmarkObs t_observations_;
			t_observations_.x = observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta) + particles[i].x;
			t_observations_.y = observations[j].x * sin(particles[i].theta) + observations[j].y * cos(particles[i].theta) + particles[i].y;
			t_observations_.id = j;
			t_observations.push_back(t_observations_);
		}

		vector<LandmarkObs> pred_landmarks;
		for (int k = 0; k < map_landmarks.landmark_list.size(); k++)
		{
			double dist_ = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f);
			if (dist_ <= sensor_range)
			{
				LandmarkObs pred_;
				pred_.x = map_landmarks.landmark_list[k].x_f;
				pred_.y = map_landmarks.landmark_list[k].y_f;
				pred_.id = map_landmarks.landmark_list[k].id_i;
                                pred_landmarks.push_back(pred_);
			}
		}

		dataAssociation(pred_landmarks, t_observations);
		double weight = 1.0;

		for (int k = 0; k < t_observations.size(); k++)
		{
			double x_meas, y_meas, x_mu, y_mu = 0.0;
			x_meas = t_observations[k].x;
			y_meas = t_observations[k].y;

			for (int j = 0; j < pred_landmarks.size(); j++)
			{
				if (pred_landmarks[j].id == t_observations[k].id)
				{
					x_mu = pred_landmarks[j].x;
					y_mu = pred_landmarks[j].y;
				}
			}

			// Calculate 2D Gaussian probability
			double prob_, t_prob, t_prob_ = 0.0;
			t_prob = 1.0/2*M_PI*std_landmark[0]*std_landmark[1];
			t_prob_ = (pow((x_meas - x_mu),2)/2*pow(std_landmark[0],2)) + (pow((y_meas-y_mu),2)/2*pow(std_landmark[1],2));
			prob_ = t_prob * exp(-t_prob_);
			if (prob_ > 0.0)
			{
				weight *= prob_;
			}
		}

		weights.push_back(weight);
		particles[i].weight = weight;
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// Define random engine "gen"
        default_random_engine gen;

	// Define discrete distribution
	discrete_distribution<int> d_(weights.begin(), weights.end());

	// Define a resampled particle vector.
	vector<Particle> resample_P_;

	// Iterate through number of particles and update resampled vector.
	for (int i = 0; i < num_particles; i++)
	{
		resample_P_.push_back(particles[d_(gen)]);
	}
	particles = resample_P_;

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
