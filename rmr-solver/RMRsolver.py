"""
Based on original method by I. Belli and Python adaptation from I. Beck.

Author:             FJ van Melis
Created on:         September 13th 2024.
Last updated on:    December 2nd 2024.

PURPOSE:
Class for preparing a Rapid Muscle Redundancy (RMR) solver for use, by providing an OpenSim model and configuring settings.

USAGE:
See runRMRsolver().py for an example on how to use this class!

_init_: 
    Set up RMR solver by providing OpenSim model and optionally change solver and constraint settings.
setObjective: 
    Set up RMR solver objective (otherwise default is selected upon first solve).
info(): 
    Print RMR solver info to terminal.
solve(): 
    Solve for a single time instant by provide kinematic state (preferred use with MotionAnalysis class).

"""
from typing import Any, List, Dict, Tuple, Optional, Union
import opensim
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
import scipy.optimize as sopt
from datetime import datetime

import utilsRMRsolver as utilsRMR
import utilsObjectives as utilsObj

class RMRsolver:
    def __init__(self, model: opensim.Model, **settingsOverwrite) -> None:
        """
        Sets up RMR solver for selected OpenSim model. Change settings from default by providing keyword arguments.

        :Parameters:
        model: ``opensim.Model``
            OpenSim model to use.

        constrainActDyn: ``bool`` (default = True)
            Toggle muscle activation dynamics constraint.

        constrainGHjoint: ``bool`` (default = False)
            Toggle glenohumeral joint reaction force constraint (requires suitable Delft Shoulder Elbow Model).

        outputJoint: ``list[str]`` (default = empty)
            Names of the joints for which to output reaction forces (expressed in ground frame)

        visualize: ``bool`` (default = False)
            Toggle external window with visualization of the model movement during solving.
        """
        self.model = model

        # Set scipy.optimize.minimize() settings:
        self.solverSettings = dict(maxiter=10000,
                                   ftol= 1E-6,
                                   disp= False,
                                   eps= 1E-8)
        
        # Set default settings and overwrite if applicable:
        self.settings = dict(actuatorMuscleBounds=[0,1], # activation bounds
                        actuatorMuscleGuess=0.1, # activation initial guess
                        actuatorReserveBounds=[-600,600],
                        actuatorReserveGuess=0,
                        actuatorDelta=[0.001,0.005], # tolerance on acceleration constraint
                        solveInfo=False, # print solve info to terminal
                        constrainActDyn=True, # constrain activation dynamics
                        tauAct=0.01, # activation time constant
                        tauDeact=0.04, # de-activation time constant
                        constrainGHjoint=False, # constrain glenohumeral joint
                        visualize=False, # visualize motion during solving
                        outputJoint=[], # list of joint names for which to output force
                        prescribedForceIndex=[])  # list of indexes of the prescribed forces, whose value will be provided at runtime

        self.settings.update(settingsOverwrite)

        # Set up coordinates:
        self.coordinateSet = self.model.getCoordinateSet()
        self.coordinateNum = self.coordinateSet.getSize()
        self.coordinates = []
        self.coordinatesFree = []
        self.coordinateNames = []
        self.coordinateFreeNames = []
        self.coordinateLockedNames = []
        self.coordinateLockedNum = 0

        for i in range(self.coordinateNum):
            coord = self.coordinateSet.get(i)
            self.coordinates.append(coord)
            coordName = coord.getName()
            isLocked = coord.get_locked()

            self.coordinateNames.append(coordName)

            if isLocked:
                self.coordinateLockedNames.append(coordName)
                self.coordinateLockedNum += 1
            else:
                self.coordinateFreeNames.append(coordName)
                self.coordinatesFree.append(coord)
        
        self.coordinateFreeNum = self.coordinateNum - self.coordinateLockedNum
        
        # Set up muscles:
        muscleSet = self.model.getMuscles()
        self.muscleNum = muscleSet.getSize()
        self.muscles = []
        self.muscleNames = []
        self.muscleForceMax = []

        for i in range(self.muscleNum):
            self.muscles.append(opensim.Millard2012EquilibriumMuscle.safeDownCast(muscleSet.get(i)))
            self.muscleNames.append(self.muscles[i].getName())
            self.muscles[i].set_ignore_tendon_compliance(True)
            self.muscles[i].set_ignore_activation_dynamics(True)
            self.muscleForceMax.append(self.muscles[i].getMaxIsometricForce())

        # Set up the functions that allow to input the value of prescribed forces at solving time
        # If the user provides an index that does not correspond to a PrescribedForce object, raise an error.
        self.prescribedForces = []
        self.prescribedForceNames = []
        self.prescribedForcesUpdFunctions = []
        for i in range(len(self.settings['prescribedForceIndex'])):
            prescribed_force = model.getForceSet().get(self.settings['prescribedForceIndex'][i])

            # check that we are dealing with a prescribed force
            if prescribed_force.getConcreteClassName() == 'PrescribedForce':
                self.prescribedForces.append(prescribed_force)
                self.prescribedForceNames.append(prescribed_force.getName())

                force_fnc_upd = opensim.PrescribedForce.safeDownCast(prescribed_force).updForceFunctions()
                torque_fnc_upd = opensim.PrescribedForce.safeDownCast(prescribed_force).updTorqueFunctions()

                self.prescribedForcesUpdFunctions.append([opensim.Constant.safeDownCast(force_fnc_upd.get(0)),      # force_x
                                                        opensim.Constant.safeDownCast(force_fnc_upd.get(1)),      # force_y
                                                        opensim.Constant.safeDownCast(force_fnc_upd.get(2)),      # force_z
                                                        opensim.Constant.safeDownCast(torque_fnc_upd.get(0)),     # torque_x
                                                        opensim.Constant.safeDownCast(torque_fnc_upd.get(1)),     # torque_y
                                                        opensim.Constant.safeDownCast(torque_fnc_upd.get(2))])    # torque_z
            else:
                raise RuntimeError(f"{prescribed_force.getName()} is not of PrescribedForce type. Double-check that the specified index correspond to a PrescribedForce")
        
        # Initialize system:
        if self.settings['visualize']:
            self.model.setUseVisualizer(True)

        self.model.finalizeConnections()
        self.state = self.model.initSystem()

        if self.settings['visualize']:
            # set solid (black) background for 
            self.model.getVisualizer().getSimbodyVisualizer().setBackgroundType(self.model.getVisualizer().getSimbodyVisualizer().SolidColor)
            self.model.getVisualizer().getSimbodyVisualizer().setBackgroundColor(opensim.Vec3(0, 0, 0))

        # Set up joints:
        outputJoint = self.settings['outputJoint']
        if not len(outputJoint) == 0:
            jointSet = self.model.getJointSet()
            self.joints = []
            for name in outputJoint:
                self.joints.append( jointSet.get(name))
            self.jointNum = len(self.joints)
        else:
            self.jointNum = 0

        if self.settings['constrainGHjoint']:
            jointSet = self.model.getJointSet()
            self.jointGH = jointSet.get('GlenoHumeral')
            self.jointGHmaxAngle = utilsRMR.get_glenoid_status(self.model, self.state)[0]
        else:
            self.jointGHmaxAngle = 0

        # Set up coordinate actuators:
        actuatorSet = self.model.getActuators()
        self.actuatorNum = actuatorSet.getSize()
        self.actuators = []
        self.actuatorNames = []
        self.actuatorMuscleIndex = [] # indices of actuators which are muscles

        for i in range(self.actuatorNum):
            self.actuatorNames.append(actuatorSet.get(i).getName())
            self.actuators.append(opensim.ScalarActuator.safeDownCast(actuatorSet.get(i)))
            if self.actuatorNames[i] in self.muscleNames: # only allow overwrite for muscle actuators
                self.actuatorMuscleIndex.append(i)
                self.actuators[i].overrideActuation(self.state, True)
        
        # Set up bounds for actuator controls:
        self.xBounds = []
        for i in range(self.actuatorNum):
            if i in self.actuatorMuscleIndex:
                self.xBounds.append( (self.settings['actuatorMuscleBounds'][0], self.settings['actuatorMuscleBounds'][1]) )
            else:
                self.xBounds.append( (self.settings['actuatorReserveBounds'][0], self.settings['actuatorReserveBounds'][1]) )
        
        # Set up initial settings:
        self.x0 = np.concatenate((self.settings['actuatorMuscleGuess'] * np.ones(self.muscleNum),
                                  self.settings['actuatorReserveGuess'] * np.ones(self.actuatorNum - self.muscleNum)))
        self.xDelta = self.settings['actuatorDelta']
        self.init = True

        # Fetch optimal forces reserve actuators:
        self.actuatorReserveOptimalForce = np.zeros(self.actuatorNum - self.muscleNum)

        for i, act in enumerate(self.actuators[self.muscleNum:]):
            self.actuatorReserveOptimalForce[i] = act.getOptimalForce()

        self.hasObjective = False

    def setObjective(self, objective: utilsObj.ActSquared) -> None:
        """
        Initialize objective function.
        """
        self.objective = objective
        weightNum = len( objective.getWeight())
        if not weightNum == self.actuatorNum:
            raise AssertionError(f"Number of weights of the objective is not equal to the number of actuators ({weightNum} and {self.actuatorNum})")

        self.hasObjective = True



    def info(self) -> None:
        """ 
        Print basic info of the RMR solver to terminal. 
        """

        print(">>> INFO: RMR SOLVER REPORT\n")

        print(f"Number of coordinates = {self.coordinateNum}")
        print(f"Number of free coordinates = {self.coordinateFreeNum}")
        print(f"Number of locked coordinates = {self.coordinateLockedNum}")
        print(f"Number of muscles = {self.muscleNum}")
        print(f"Number of actuators = {self.actuatorNum}")

    
        print("\n--- COORDINATES ---")
        for i, item in enumerate(self.coordinateNames):
            print(f"{i}. {item}")

        print("\n--- MUSCLES ---")
        for i, item in enumerate(self.muscleNames):
            print(f"{i}. {item}")

        print("\n--- ACTUATORS ---")
        for i, item in enumerate(self.actuatorNames):
            print(f"{i}. {item}")

        print("\n--- JOINTS ---")
        jointSetTotal = self.model.getJointSet()
        jointNumTotal = jointSetTotal.getSize()
        for i in range(jointNumTotal):
            print(f"{i}. {jointSetTotal.get(i).getName()}")
        


    def solve(self, time: float, position: npt.ArrayLike, speed: npt.ArrayLike, acceleration: npt.ArrayLike, EMG: Optional[npt.ArrayLike] = None, indexes_prescribed_forces: Optional[List] = None, values_prescribed_forces: Optional[npt.ArrayLike] = None) -> Tuple[npt.NDArray, npt.NDArray, Dict]:
        """ 
        Solve for state and return activations.
        Inputs are:
            * time: current time of the simulation
            * position: vector containing the position of each free coordinate in the model at the given time
            * speed: vector containing the speed of each free coordinate
            * acceleration: vector containing the acceleration of each free coordinate
            * EMG: EMG-based activation for the selected muscles. Their activation will try to track this reference
                    as specified in the cost function
            * values_prescribed_force: force and torque values to be applied at the PrescribedForce objects. They should be 6xN,
                                       with N being the number of PrescribedForces to be updated, and each column carrying the 
                                       values for the force (x,y,z) and torques (x,y,z) in this order. The coordinate frame in
                                       which this wrench is applied is the one specified when creating the PrescribedForce.
                                       When N>1, the order in which the PrescribedForce objects are treated depends on how they
                                       are stored in self.prescribedForces
        """

        timeStart = datetime.now()

        # Set default objective if it is missing:
        if not self.hasObjective:
            raise RuntimeError("No objective was set for RMR solver")

        if EMG is None:
            EMG = 0

        # Update activation bounds if enabled (constraint 2):
        if self.settings['constrainActDyn'] and not self.init:
            timeStep = time - self.timePrev
            
            for i in range(self.muscleNum):
                lb = np.max((self.x0[i] - self.x0[i] * (0.5 + 1.5 * self.x0[i]) * timeStep / self.settings['tauDeact'], 0))
                ub = np.min((self.x0[i]  + (1 - self.x0[i] ) * timeStep / (self.settings['tauAct'] * (0.5 + 1.5 * self.x0[i] )), 1))
            
                self.xBounds[self.actuatorMuscleIndex[i]] = (lb, ub)

        self.timePrev = time
        self.init = False # next time trigger activation bounds update

        # Update state:
        self.state.setTime(time)

        for i, coord in enumerate(self.coordinatesFree):
            coord.setValue(self.state, position[i], False)
            coord.setSpeedValue(self.state, speed[i])

        self.model.assemble(self.state)
        self.model.equilibrateMuscles(self.state)
        self.model.realizeVelocity(self.state)
        modelControls = self.model.getControls(self.state)

        # Get muscle force multipliers:
        muscleForceActiveMax = []
        muscleForcePassiveMax = []
        muscleMomentArm = []

        for i, muscle in enumerate(self.muscles):
            ForceLen = muscle.getActiveForceLengthMultiplier(self.state)
            ForceVel = muscle.getForceVelocityMultiplier(self.state)
            ForcePas = muscle.getPassiveForceMultiplier(self.state)
            cosPenn = muscle.getCosPennationAngle(self.state)

            muscleForceActiveMax.append(ForceLen * ForceVel * self.muscleForceMax[i] * cosPenn)
            muscleForcePassiveMax.append(ForcePas * self.muscleForceMax[i] * cosPenn)
            
            # TODO: retrieving the muscleMomentArm appears useless..
            for j, coord in enumerate(self.coordinatesFree):
                muscleMomentArm.append([])
                muscleMomentArm[j].append(muscle.computeMomentArm(self.state, coord))

        # apply the prescribed forces to the model
        if values_prescribed_forces is not None:        # check that there are values, otherwise the default ones specified at model creation would be used
            values_prescribed_forces = np.asarray(values_prescribed_forces)
            if values_prescribed_forces.ndim ==1:
                values_prescribed_forces = values_prescribed_forces[:,np.newaxis]
            assert values_prescribed_forces.shape[0] == 6, ("Incorrect size for PrescribedForce values.\n"
                                                            "Please provide a 6-element column for every force (f_x, f_y, f_z, tau_x, tau_y, tau_z)")
            if len(self.prescribedForces)>0:            # check that the model contains prescribed forces
                assert len(self.prescribedForces) == values_prescribed_forces.shape[1], ("The number of PrescribedForce elements known to the solver is different than the input received.\n"
                                                                                         "Provide 6 input values for every PrescribedForce elements.")
                
                for i in range(values_prescribed_forces.shape[1]):
                    self.prescribedForcesUpdFunctions[i][0].setValue(values_prescribed_forces[0, i])    # force_x
                    self.prescribedForcesUpdFunctions[i][1].setValue(values_prescribed_forces[1, i])    # force_y
                    self.prescribedForcesUpdFunctions[i][2].setValue(values_prescribed_forces[2, i])    # force_z
                    self.prescribedForcesUpdFunctions[i][3].setValue(values_prescribed_forces[3, i])    # torque_x
                    self.prescribedForcesUpdFunctions[i][4].setValue(values_prescribed_forces[4, i])    # torque_y
                    self.prescribedForcesUpdFunctions[i][5].setValue(values_prescribed_forces[5, i])    # torque_z

            else:
                raise RuntimeError("No prescribed force was specified when initializing the solver, but force values are specified")
        
        # Set up acceleration constraint (constraint 1):
        @dataclass
        class paramsAcc:
            _model: Any = self.model
            _state: Any = self.state
            _actuators: list = field(default_factory=list)
            _actuatorNames: list = field(default_factory=list)
            _coordinates: list = field(default_factory=list)
            _coordinatesFree: list = field(default_factory=list)
            _coordinateNames: list = field(default_factory=list)
            _coordinateFreeNames: list = field(default_factory=list)
            _coordinateNum: int = self.coordinateNum
            _coordinateFreeNum: int = self.coordinateFreeNum
            _muscles: list = field(default_factory=list)
            _muscleNames: list = field(default_factory=list)
            _muscleNum: int = self.muscleNum
            _useMuscles: bool = True
            _useControls: bool = True
            _forceActive: list = field(default_factory=list)
            _forcePassive: list = field(default_factory=list)
            _modelControls: Any = modelControls

        paramsAcc._actuators = self.actuators
        paramsAcc._actuatorNames = self.actuatorNames
        paramsAcc._coordinates = self.coordinates
        paramsAcc._coordinatesFree = self.coordinatesFree
        paramsAcc._coordinateNames = self.coordinateNames
        paramsAcc._coordinateFreeNames = self.coordinateFreeNames
        paramsAcc._muscles = self.muscles
        paramsAcc._muscleNames = self.muscleNames
        paramsAcc._forceActive = muscleForceActiveMax
        paramsAcc._forcePassive = muscleForcePassiveMax

        if self.settings['constrainGHjoint']:
            paramsAcc._GHjoint = self.jointGH

            constrAcc_A, constrAcc_b, constrGH_A, JRFnoAct = utilsRMR.constructAccelerationConstraint_GH(paramsAcc, acceleration, self.actuatorNum)

            # Set up GH-JRF constraint (constraint 3):
            Vec_H2GC = utilsRMR.get_glenoid_status(self.model, self.state)[1]

            @dataclass
            class paramsGH:
                _vectorGlenoid: np.array = field(default_factory=np.array)
                _JRFnoAct: np.array = field(default_factory=np.array)
                _A: np.array = field(default_factory=np.array)
                _maxAngle: float = self.jointGHmaxAngle
            paramsGH._vectorGlenoid = Vec_H2GC
            paramsGH._JRFnoAct = JRFnoAct
            paramsGH._A = constrGH_A

            constraintGH = sopt.NonlinearConstraint(lambda x: utilsRMR.GHjointConstraint(x, paramsGH), -1.0, 0.0)
        else:
            constrAcc_A, constrAcc_b = utilsRMR.constructAccelerationConstraint(paramsAcc, acceleration, self.actuatorNum)

        # Employ solver: (retry for higher tolerance xDelta if optimization is unsuccessful)
        for xDelta in self.xDelta:
            xDeltaUsed = xDelta # save xDelta used for printing/plotting

            # Finalize constraints:
            constrAcc_b_lb, constrAcc_b_ub = utilsRMR.addTolerance(vector=constrAcc_b, tolerance= xDelta)
            
            constraintAcc = sopt.LinearConstraint(A= constrAcc_A, lb=constrAcc_b_lb, ub=constrAcc_b_ub, keep_feasible=False)

            if self.settings['constrainGHjoint']:
                constraints = [constraintAcc, constraintGH]
            else:
                constraints = [constraintAcc]

            # Solve:
            result = sopt.minimize(self.objective, 
                                    x0= self.x0,
                                    args= EMG,
                                    method='SLSQP', 
                                    constraints=constraints, 
                                    bounds=self.xBounds,
                                    options=self.solverSettings)
            if result.success:
                break

            self.x0 = result.x # better initial guess for next try
        
        # Get results:
        timeElapsed = datetime.now() - timeStart 
        self.x0 = result.x # better initial guess for next time instant
        activation = result.x
        
        # activeAcc = utilsRMR.getInducedAcceleration(activation, paramsAcc)
        # noActAcc = utilsRMR.getInducedAcceleration(np.zeros(self.actuatorNum), paramsAcc)
        # constraintViolation = activeAcc - noActAcc - constrAcc_b

        # Print solver log if enabled
        if self.settings['solveInfo']:
            constraintViolation = constrAcc_A.dot(activation) - constrAcc_b
            constraintViolationRelative = constraintViolation/constrAcc_b/xDeltaUsed

            print("Acceleration constraint violation: (>=1: no violation)")
            for i in range(len(constraintViolationRelative)):
                print(f"{constraintViolationRelative[i]:5.2f}")
                
            if result.success:
                print(">>> INFO: Optimization succeeded:")
            else:
                print(">>> INFO: Optimization failed:")
                print(result.message)

            print(f"Final cost = {result.fun}")
            print(f"xDelta = {xDeltaUsed}")
            print(f"Time elapsed = {timeElapsed}")
        
        # Acquire joint reaction forces for joints specified in outputJoints:
        if not self.jointNum == 0:
            # Initialize the muscles to produce the required forces
            forceTotal = muscleForceActiveMax * activation[:self.muscleNum] + muscleForcePassiveMax
            for i, muscle in enumerate(self.muscles):
                muscle.setOverrideActuation(self.state, forceTotal[i])

            # Initialize the CoordinateActuators to produce the required effect
            for i, act in enumerate(self.actuators):
                if not self.actuatorNames[i] in self.muscleNames:
                    act.setControls(opensim.Vector(1, activation[i]), modelControls)
            
            # Realize the model to the acceleration stage:
            self.model.realizeVelocity(self.state)
            self.model.setControls(self.state, modelControls)
            self.model.realizeAcceleration(self.state)
            
            jointForces = np.zeros([self.jointNum, 6])
            for i, joint in enumerate(self.joints):
                jointForces[i,:3] = joint.calcReactionOnParentExpressedInGround(self.state).get(0).to_numpy() # moments
                jointForces[i,3:] = joint.calcReactionOnParentExpressedInGround(self.state).get(1).to_numpy() # forces
        else:
            jointForces = 0

        # Acquire GH-JRF magnitude and relative angle:
        if self.settings['constrainGHjoint']:
            GH_forceVector = np.matmul(constrGH_A, result.x) + JRFnoAct
            GH_forceMagnitude = np.linalg.norm(GH_forceVector)

            cosTheta = np.max((np.min((np.dot(Vec_H2GC, GH_forceVector) / (np.linalg.norm(Vec_H2GC) * np.linalg.norm(GH_forceVector)), 1)), -1))
            GH_forceRelativeAngle = np.rad2deg( np.arccos(cosTheta) )
        else:
            GH_forceRelativeAngle = 0
            GH_forceMagnitude = 0

        # Pack info:
        info = dict(success=result.success,
                    message=result.message,
                    cost=result.fun,
                    tolerance=xDeltaUsed,
                    timeElapsed=timeElapsed.total_seconds(),
                    GHjointMaxAngle=self.jointGHmaxAngle,
                    GHjointAngle=GH_forceRelativeAngle,
                    GHjointForce=GH_forceMagnitude)
        
        # Visualize if enabled:
        if self.settings['visualize']:
            self.model.getVisualizer().show(self.state)
        
        # Convert reserve actuator 'activation' to forces
        activation[self.muscleNum:] = activation[self.muscleNum:] * self.actuatorReserveOptimalForce
        
        return activation, jointForces, info