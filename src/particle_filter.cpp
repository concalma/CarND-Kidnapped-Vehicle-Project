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
#include <map>

#include "particle_filter.h"

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]); // FIXME

    num_particles = 200;
    particles.resize(num_particles);

    for(int i=0; i<num_particles; i++) {
        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
        particles[i].weight = 1;
    }

    is_initialized = true;

    
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    // from model update:
    // Xf = x0 + 
    //
    default_random_engine gen;
    
    for( auto& p : particles ) {
        if(fabs(yaw_rate) > 0.00001) {
            p.x = p.x + (velocity/yaw_rate) * ( sin(p.theta + yaw_rate*delta_t) - sin(p.theta) );
            p.y = p.y + (velocity/yaw_rate) * ( cos(p.theta) - cos(p.theta + yaw_rate*delta_t) );
            p.theta = p.theta + yaw_rate*delta_t;
        } else {
            p.x = p.x + velocity * delta_t * cos(p.theta);
            p.y = p.y + velocity * delta_t * sin(p.theta);
            //p.theta = p.theta;
        }
            
        normal_distribution<double> dist_x(p.x, std_pos[0]);
        normal_distribution<double> dist_y(p.y, std_pos[1]);
        normal_distribution<double> dist_theta(p.theta, std_pos[2]);

        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for( LandmarkObs &obs : observations ) {
        double d, mindist = std::numeric_limits<double>::max();
        for( auto lm : predicted ) {
            if( (d=dist(lm.x,lm.y, obs.x, obs.y)) < mindist ) { 
                obs.id = lm.id;
                mindist = d;
            }
        }
    }
}

#define POW2(x) ((x)*(x))

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
    //
    //  STEPS to take
    //  1. for every particle
    //    2. xform obs from car coordinates to map coordinates
    //    3. data association between particle observe part and map features
    //    4. we can calculate prob(x,y) per measurement, then calculate weight
    

    
    weights.clear();
    for( auto &p : particles ) {

        // 2. xform from car coordinates to map coordinates
        vector<LandmarkObs> xformed_obs;
        for( LandmarkObs o : observations ) {
            LandmarkObs lo;
            lo.x = p.x + o.x * cos(p.theta) - o.y * sin(p.theta);
            lo.y = p.y + o.x * sin(p.theta) + o.y * cos(p.theta);
            xformed_obs.push_back(lo);
        }

        // curate map features inside range of sensor 'sensor_range'
        vector<LandmarkObs> predicted_obs;
        for( auto lm : map_landmarks.landmark_list ) {
            if( dist(p.x,p.y, lm.x_f, lm.y_f) < sensor_range ) {
                LandmarkObs lo;
                lo.x = lm.x_f;
                lo.y = lm.y_f;
                lo.id = lm.id_i;
                predicted_obs.push_back(lo);
            }
        }

        
        map<int,LandmarkObs> obsmap; // mapping id landmarks to landmarks for O(1) lookup later
        for( LandmarkObs &lm : predicted_obs ) {
            obsmap[lm.id] = lm;
        }

        // 3. data association and map features. This updates the id field in 'xformed_obs' with closest one
        dataAssociation( predicted_obs, xformed_obs);

        // 4. updating weights
        double sx=std_landmark[0], sy=std_landmark[1];
        double sx2=sx*sx, sy2=sy*sy;
        p.weight = 1;

        for( auto const &obs : xformed_obs ) {
            LandmarkObs lm = obsmap[obs.id]; // associated landmark to this measurement
            // calculate multivariate gaussian probability density function
            double prob = (1/(2*M_PI*sx*sy)) * exp( -(  POW2(lm.x-obs.x)/(2*sx2) +  POW2(lm.y-obs.y)/(2*sy2) ) );    
            p.weight *= prob;
            //printf("%f,", prob  );
        }
        //printf("final weight: %f\n", p.weight);
        weights.push_back(p.weight);
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    //
    std::discrete_distribution<int> discrete_dist(weights.begin(), weights.end());
    std::vector<Particle> new_parts;

    for(int i = 0; i < num_particles; i++)
    {
        auto idx = discrete_dist(gen);
        new_parts.push_back(particles[idx]);
    }
    particles = new_parts;

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
