
import os
import pinocchio as pin
import crocoddyl
import numpy as np
from os.path import dirname, join
from pinocchio.robot_wrapper import RobotWrapper

def getModelPath(subpath, printmsg=False):
    source = os.path.abspath('')  # top level directory
    if (printmsg): print("using %s as modelPath" % source)
    return source

class RobotLoader(object):
    path = ''
    urdf_filename = ''
    srdf_filename = ''
    urdf_subpath = 'urdf'
    srdf_subpath = 'srdf'
    ref_posture = 'half_sitting'
    has_rotor_parameters = False
    free_flyer = False
    verbose = False

    def __init__(self):
        urdf_path = join(self.path, self.urdf_subpath, self.urdf_filename)
        self.model_path = getModelPath(urdf_path, self.verbose)
        self.urdf_path = join(self.model_path, urdf_path)
        self.robot = RobotWrapper.BuildFromURDF(self.urdf_path, [join(self.model_path, '../..')],
                                                pin.JointModelFreeFlyer() if self.free_flyer else None)
                                                # pin.JointModelPZ())

        if self.srdf_filename:
            self.srdf_path = join(self.model_path, self.path, self.srdf_subpath, self.srdf_filename)
            self.robot.q0 = readParamsFromSrdf(self.robot.model, self.srdf_path, self.verbose,
                                               self.has_rotor_parameters, self.ref_posture)

            if pin.WITH_HPP_FCL and pin.WITH_HPP_FCL_BINDINGS:
                # Add all collision pairs
                self.robot.collision_model.addAllCollisionPairs()

                # Remove collision pairs per SRDF
                pin.removeCollisionPairs(self.robot.model, self.robot.collision_model, self.srdf_path, False)

                # Recreate collision data since the collision pairs changed
                self.robot.collision_data = self.robot.collision_model.createData()
        else:
            self.srdf_path = None
            self.robot.q0 = pin.neutral(self.robot.model)

        if self.free_flyer:
            self.addFreeFlyerJointLimits()

    def addFreeFlyerJointLimits(self):
        ub = self.robot.model.upperPositionLimit
        ub[:7] = 1
        self.robot.model.upperPositionLimit = ub
        lb = self.robot.model.lowerPositionLimit
        lb[:7] = -1
        self.robot.model.lowerPositionLimit = lb

    @property
    def q0(self):
        warnings.warn("`q0` is deprecated. Please use `robot.q0`", FutureWarning, 2)
        return self.robot.q0

def readParamsFromSrdf(model, SRDF_PATH, verbose=False, has_rotor_parameters=True, referencePose='half_sitting'):
    if has_rotor_parameters:
        pin.loadRotorParameters(model, SRDF_PATH, verbose)
    model.armature = np.multiply(model.rotorInertia.flat, np.square(model.rotorGearRatio.flat))
    pin.loadReferenceConfigurations(model, SRDF_PATH, verbose)
    q0 = pin.neutral(model)
    if referencePose is not None:
        q0 = model.referenceConfigurations[referencePose].copy()
    return q0


class FOREloader(RobotLoader):
    path = 'robot_description_package'
    urdf_filename = "robot_simplified.urdf"
    srdf_filename = "robot_simplified.srdf"
    ref_posture = "standing"
    free_flyer = True


ROBOTS = {
    'fore': FOREloader,
}

def load(name):
    """Load a robot by its name"""
    inst = ROBOTS[name]()
    return inst.robot
