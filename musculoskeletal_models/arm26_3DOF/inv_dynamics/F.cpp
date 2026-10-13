#include <OpenSim/Simulation/Model/Model.h>
#include <OpenSim/Simulation/SimbodyEngine/PinJoint.h>
#include <OpenSim/Simulation/SimbodyEngine/WeldJoint.h>
#include <OpenSim/Simulation/SimbodyEngine/PlanarJoint.h>
#include <OpenSim/Simulation/SimbodyEngine/Joint.h>
#include <OpenSim/Simulation/SimbodyEngine/SpatialTransform.h>
#include <OpenSim/Simulation/SimbodyEngine/CustomJoint.h>
#include <OpenSim/Common/LinearFunction.h>
#include <OpenSim/Common/PolynomialFunction.h>
#include <OpenSim/Common/MultiplierFunction.h>
#include <OpenSim/Common/Constant.h>
#include <OpenSim/Simulation/Model/SmoothSphereHalfSpaceForce.h>
#include <OpenSim/Simulation/SimulationUtilities.h>
#include "SimTKcommon/internal/recorder.h"

#include <iostream>
#include <iterator>
#include <random>
#include <cassert>
#include <algorithm>
#include <vector>
#include <fstream>

using namespace SimTK;
using namespace OpenSim;

constexpr int n_in = 2; 
constexpr int n_out = 1; 
constexpr int nCoordinates = 3; 
constexpr int NX = nCoordinates*2; 
constexpr int NU = nCoordinates; 
constexpr int NR = 24; 

template<typename T> 
T value(const Recorder& e) { return e; }; 
template<> 
double value(const Recorder& e) { return e.getValue(); }; 

template<typename T>
int F_generic(const T** arg, T** res) {

	// Definition of model.
	OpenSim::Model* model;
	model = new OpenSim::Model();

	// Definition of bodies.
	OpenSim::Body* base;
	base = new OpenSim::Body("base", 0.00000000000000000000, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Inertia(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000, 0., 0., 0.));
	model->addBody(base);

	OpenSim::Body* r_humerus;
	r_humerus = new OpenSim::Body("r_humerus", 1.86457199999999989615, Vec3(0.00000000000000000000, -0.18049599999999998978, 0.00000000000000000000), Inertia(0.01481000000000000025, 0.00455100000000000036, 0.01319299999999999987, 0., 0., 0.));
	model->addBody(r_humerus);

	OpenSim::Body* r_ulna_radius_hand;
	r_ulna_radius_hand = new OpenSim::Body("r_ulna_radius_hand", 1.53431500000000009543, Vec3(0.00000000000000000000, -0.18147900000000000142, 0.00000000000000000000), Inertia(0.01928099999999999953, 0.00157099999999999991, 0.02006199999999999997, 0., 0., 0.));
	model->addBody(r_ulna_radius_hand);

	// Definition of joints.
	OpenSim::WeldJoint* offset;
	offset = new OpenSim::WeldJoint("offset", model->getGround(), Vec3(0.00000000000000000000, 0.80000000000000004441, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *base, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	model->addJoint(offset);

	SpatialTransform st_r_shoulder;
	st_r_shoulder[0].setCoordinateNames(OpenSim::Array<std::string>("r_shoulder_elev", 1, 1));
	st_r_shoulder[0].setFunction(new LinearFunction(1.0000, 0.0000));
	st_r_shoulder[0].setAxis(Vec3(-0.05889802000000000232, 0.00229999999999999996, 0.99826135999999998649));
	st_r_shoulder[1].setFunction(new Constant(0.00000000000000000000));
	st_r_shoulder[1].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_r_shoulder[2].setCoordinateNames(OpenSim::Array<std::string>("r_shoulder_plane", 1, 1));
	st_r_shoulder[2].setFunction(new LinearFunction(1.0000, 0.0000));
	st_r_shoulder[2].setAxis(Vec3(0.99826135999999998649, -0.00000000000000000000, 0.05889802000000000232));
	st_r_shoulder[3].setFunction(new Constant(0.00000000000000000000));
	st_r_shoulder[3].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_r_shoulder[4].setFunction(new Constant(0.00000000000000000000));
	st_r_shoulder[4].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_r_shoulder[5].setFunction(new Constant(0.00000000000000000000));
	st_r_shoulder[5].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	OpenSim::CustomJoint* r_shoulder;
	r_shoulder = new OpenSim::CustomJoint("r_shoulder", *base, Vec3(-0.01754500000000000157, -0.00700000000000000015, 0.17000000000000001221), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *r_humerus, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), st_r_shoulder);
	model->addJoint(r_shoulder);

	SpatialTransform st_r_elbow;
	st_r_elbow[0].setCoordinateNames(OpenSim::Array<std::string>("r_elbow_flex", 1, 1));
	st_r_elbow[0].setFunction(new LinearFunction(1.0000, 0.0000));
	st_r_elbow[0].setAxis(Vec3(0.04940001000000000114, 0.03660001000000000226, 0.99810825000000003016));
	st_r_elbow[1].setFunction(new Constant(0.00000000000000000000));
	st_r_elbow[1].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_r_elbow[2].setFunction(new Constant(0.00000000000000000000));
	st_r_elbow[2].setAxis(Vec3(0.99810825000000003016, 0.00000000000000000000, -0.04940001000000000114));
	st_r_elbow[3].setFunction(new Constant(0.00000000000000000000));
	st_r_elbow[3].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_r_elbow[4].setFunction(new Constant(0.00000000000000000000));
	st_r_elbow[4].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_r_elbow[5].setFunction(new Constant(0.00000000000000000000));
	st_r_elbow[5].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	OpenSim::CustomJoint* r_elbow;
	r_elbow = new OpenSim::CustomJoint("r_elbow", *r_humerus, Vec3(0.00610000000000000039, -0.29039999999999999147, -0.01230000000000000017), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *r_ulna_radius_hand, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), st_r_elbow);
	model->addJoint(r_elbow);

	// Definition of contacts.
	// Initialize system.
	SimTK::State* state;
	state = new State(model->initSystem());

	// Read inputs.
	std::vector<T> x(arg[0], arg[0] + NX);
	std::vector<T> u(arg[1], arg[1] + NU);

	// States and controls.
	T ua[NU];
	Vector QsUs(NX);
	/// States
	for (int i = 0; i < NX; ++i) QsUs[i] = x[i];
	/// Controls
	/// OpenSim and Simbody have different state orders.
	auto indicesOSInSimbody = getIndicesOpenSimInSimbody(*model);
	for (int i = 0; i < NU; ++i) ua[i] = u[indicesOSInSimbody[i]];

	// Set state variables and realize.
	model->setStateVariableValues(*state, QsUs);
	model->realizeVelocity(*state);

	// Compute residual forces.
	/// Set appliedMobilityForces (# mobilities).
	Vector appliedMobilityForces(nCoordinates);
	appliedMobilityForces.setToZero();
	/// Set appliedBodyForces (# bodies + ground).
	Vector_<SpatialVec> appliedBodyForces;
	int nbodies = model->getBodySet().getSize() + 1;
	appliedBodyForces.resize(nbodies);
	appliedBodyForces.setToZero();
	/// Set gravity.
	Vec3 gravity(0);
	gravity[1] = -9.80659999999999953957;
	/// Add weights to appliedBodyForces.
	for (int i = 0; i < model->getBodySet().getSize(); ++i) {
		model->getMatterSubsystem().addInStationForce(*state,
		model->getBodySet().get(i).getMobilizedBodyIndex(),
		model->getBodySet().get(i).getMassCenter(),
		model->getBodySet().get(i).getMass()*gravity, appliedBodyForces);
	}
	/// Add contact forces to appliedBodyForces.
	/// knownUdot.
	Vector knownUdot(nCoordinates);
	knownUdot.setToZero();
	for (int i = 0; i < nCoordinates; ++i) knownUdot[i] = ua[i];
	/// Calculate residual forces.
	Vector residualMobilityForces(nCoordinates);
	residualMobilityForces.setToZero();
	model->getMatterSubsystem().calcResidualForceIgnoringConstraints(*state,
			appliedMobilityForces, appliedBodyForces, knownUdot, residualMobilityForces);

	/// Body origins.
	Vec3 base_or = base->getPositionInGround(*state);
	Vec3 r_humerus_or = r_humerus->getPositionInGround(*state);
	Vec3 r_ulna_radius_hand_or = r_ulna_radius_hand->getPositionInGround(*state);

	/// Ground reaction forces.
	Vec3 GRF_r(0), GRF_l(0);

	/// Ground reaction moments.
	Vec3 GRM_r(0), GRM_l(0);
	Vec3 normal(0, 1, 0);

	/// Outputs.
	/// Residual forces (OpenSim and Simbody have different state orders).
	auto indicesSimbodyInOS = getIndicesSimbodyInOpenSim(*model);
	for (int i = 0; i < NU; ++i) res[0][i] =
			value<T>(residualMobilityForces[indicesSimbodyInOS[i]]);
	/// Ground reaction forces.
	for (int i = 0; i < 3; ++i) res[0][i + 3] = value<T>(GRF_r[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 6] = value<T>(GRF_l[i]);
	/// Ground reaction moments.
	for (int i = 0; i < 3; ++i) res[0][i + 9] = value<T>(GRM_r[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 12] = value<T>(GRM_l[i]);
	/// Body origins.
	for (int i = 0; i < 3; ++i) res[0][i + 15] = value<T>(base_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 18] = value<T>(r_humerus_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 21] = value<T>(r_ulna_radius_hand_or[i]);

	return 0;
}

int main() {
	Recorder x[NX];
	Recorder u[NU];
	Recorder tau[NR];
	for (int i = 0; i < NX; ++i) x[i] <<= 0;
	for (int i = 0; i < NU; ++i) u[i] <<= 0;
	const Recorder* Recorder_arg[n_in] = { x,u };
	Recorder* Recorder_res[n_out] = { tau };
	F_generic<Recorder>(Recorder_arg, Recorder_res);
	double res[NR];
	const Recorder symbolic_zero = x[0] - x[0];
	for (int i = 0; i < NR; ++i) (Recorder_res[0][i] + symbolic_zero) >>= res[i];
	Recorder::stop_recording();
	return 0;
}
