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

#include "helper_functions.h"
#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	default_random_engine gen;
	// Creates normal (Gaussian) distributions for x, y and theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; ++i) {
		Particle particle;

		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1.0;
		particles.push_back(particle);
		weights.push_back(1.0);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	normal_distribution<double> dist_x(0.0, std_pos[0]);
	normal_distribution<double> dist_y(0.0, std_pos[1]);
	normal_distribution<double> dist_theta(0.0, std_pos[2]);

	for (int i; i < num_particles; ++i) {

		double theta = particles[i].theta;
		if (fabs(yaw_rate) < 0.00001) {
			particles[i].x += velocity * delta_t * cos(theta);
			particles[i].y += velocity * delta_t * sin(theta);
		}
		else {
			particles[i].x += (velocity / yaw_rate) * (sin(theta + yaw_rate * delta_t) - sin(theta));
			particles[i].y += (velocity / yaw_rate) * (cos(theta) - cos(theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}
		// Adding noise
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
	for(int i = 0; i < observations.size(); i++ ) {
		double min_dist = numeric_limits<double>::max();
		int id;
		for (int j = 0; j < predicted.size(); j++) {
			// Current Distance using helper_functions
			double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			if(distance < min_dist) {
				min_dist = distance;
				id = predicted[j].id;
			}
		}
		// Set observed measurement to landmark ID
		observations[i].id = id;
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
	for (int i = 0; i < num_particles; ++i) {
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;

		vector <LandmarkObs> in_range_landmarks;
		for(int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
			int landmarkID = map_landmarks.landmark_list[j].id_i;
			double landmarkX = map_landmarks.landmark_list[j].x_f;
			double landmarkY = map_landmarks.landmark_list[j].y_f;

			double distance = dist(x, y, landmarkX, landmarkY);

			if (distance <= sensor_range) {
				in_range_landmarks.push_back(LandmarkObs{landmarkID, landmarkX, landmarkY});
			}
		}

		vector<LandmarkObs> transformed_observations;
		for (int k = 0; k < observations.size(); ++k){
			int observationID = observations[k].id;
			double observationX = observations[k].x;
			double observationY = observations[k].y;

			double transformedX =  x + cos(theta) * observationX - sin(theta) * observationY;
			double transformedY =  y + sin(theta) * observationX + cos(theta) * observationY;

			transformed_observations.push_back(LandmarkObs{observationID, transformedX, transformedY});
		}

		dataAssociation(in_range_landmarks, transformed_observations);
		particles[i].weight = 1.0;

		for (int l = 0; l < transformed_observations.size(); ++l) {

			double convertedX = transformed_observations[l].x;
			double convertedY = transformed_observations[l].y;
			int convertedID = transformed_observations[l].id;
			double prX, prY;

			for (int m = 0; m< in_range_landmarks.size(); m++) {
				if (in_range_landmarks[m].id == convertedID) {
					prX = in_range_landmarks[m].x;
					prY = in_range_landmarks[m].y;
				}
			}

			double stdX = std_landmark[0];
			double stdY = std_landmark[1];
			double newW = ( 1/(2*M_PI*stdX*stdY)) * exp( -( pow(prX-convertedX,2)/(2*pow(stdX, 2)) + (pow(prY-convertedY,2)/(2*pow(stdY, 2))) ) );

			particles[i].weight *= newW;
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> new_particles;
	vector<double> weights;
	for(int i = 0; i < num_particles; i++){
		weights.push_back(particles[i].weight);
	}
	double max_weight = *max_element(weights.begin(), weights.end());
	uniform_int_distribution<int> uniintdist(0, num_particles - 1);
	int index = uniintdist(gen);
	uniform_real_distribution<double> unirealdist(0.0, max_weight);
	double beta = 0.0;
	for(int i = 0; i < num_particles; i++){
		beta += unirealdist(gen) * 2.0;
		while (beta > weights[index]){
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		new_particles.push_back(particles[index]);
	}

	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
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
