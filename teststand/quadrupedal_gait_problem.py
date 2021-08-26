import pinocchio as pin
import crocoddyl
import numpy as np

class SimpleQuadrupedalGaitProblem:
    
    def __init__(self, rmodel, feet, stateWeights, defaultState): # for ANYmal: feet = [lfFoot, rfFoot, lhFoot, rhFoot]
        self.rmodel = rmodel
        self.rdata = rmodel.createData()
        self.state = crocoddyl.StateMultibody(self.rmodel)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        # Getting the frame id for all the legs
        self.feet = []
        self.stateWeights = stateWeights
        for foot in feet:
            self.feet +=[self.rmodel.getFrameId(foot)]
#         self.lfFootId = self.feet[0]
#         self.rfFootId = self.feet[1]
#         self.lhFootId = self.feet[2]
#         self.rhFootId = self.feet[3]

        # Defining default state
        # self.rmodel.defaultState = defaultState
        q0 = self.rmodel.referenceConfigurations["standing"]
        self.rmodel.defaultState = np.concatenate([q0, np.zeros(self.rmodel.nv)])
        self.firstStep = True
        # Defining the friction coefficient and normal
        self.mu = 0.7
        self.Rsurf = np.eye(3)
        
    def createJumpingProblem(self, x0, jumpHeight, jumpLength, timeStep, groundKnots, flyingKnots):
        q0 = x0[:self.rmodel.nq]
        pin.forwardKinematics(self.rmodel, self.rdata, q0)
        pin.updateFramePlacements(self.rmodel, self.rdata)
        FootPos0 = []
        comRef = 0
        for foot in self.feet:
            FootPos0 += [self.rdata.oMf[foot].translation]
            FootPos0[-1][2] = 0
#             comRef += FootPos0[-1]
#         comRef /= 4
        comRef = pin.centerOfMass(self.rmodel, self.rdata, q0) # assumption is that the intitial com is correct
        
#         rfFootPos0 = self.rdata.oMf[self.feet[1]].translation
#         rhFootPos0 = self.rdata.oMf[self.feet[3]].translation
#         lfFootPos0 = self.rdata.oMf[self.feet[0]].translation
#         lhFootPos0 = self.rdata.oMf[self.feet[2]].translation
# #         df = jumpLength[2] - rfFootPos0[2]
#         rfFootPos0[2] = 0.
#         rhFootPos0[2] = 0.
#         lfFootPos0[2] = 0.
#         lhFootPos0[2] = 0.
        
#         comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        

        loco3dModel = []
        takeOff = [
            self.createSwingFootModel(
                timeStep,
                self.feet,
            ) for k in range(groundKnots)
        ]
        flyingUpPhase = [
            self.createSwingFootModel(
                timeStep, [],
                np.array([0., 0., jumpHeight]) * (k + 1) / flyingKnots + comRef)
                # np.array([jumpLength[0], jumpLength[1], jumpLength[2] + jumpHeight]) * (k + 1) / flyingKnots + comRef)
            for k in range(flyingKnots)
        ]
        flyingDownPhase = []
        for k in range(flyingKnots):
            flyingDownPhase += [self.createSwingFootModel(timeStep, [])]

#          f0 = jumpLength
#         footTask = [
#             crocoddyl.FramePlacement(self.lfFootId, pin.SE3(np.eye(3), lfFootPos0 + f0)),
#             crocoddyl.FramePlacement(self.rfFootId, pin.SE3(np.eye(3), rfFootPos0 + f0)),
#             crocoddyl.FramePlacement(self.lhFootId, pin.SE3(np.eye(3), lhFootPos0 + f0)),
#             crocoddyl.FramePlacement(self.rhFootId, pin.SE3(np.eye(3), rhFootPos0 + f0))
#         ]
#         landingPhase = [
#             self.createFootSwitchModel([self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId], footTask, False)
#         ]
#         f0[2] = df
        landed = [
            self.createSwingFootModel(timeStep, self.feet,
                                      comTask=comRef) for k in range(groundKnots)
        ]
        loco3dModel += takeOff
        loco3dModel += flyingUpPhase
        loco3dModel += flyingDownPhase
#         loco3dModel += landingPhase
        loco3dModel += landed

        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem
    
    def createSwingFootModel(self, timeStep, supportFootIds, comTask=None, swingFootTask=None):
        """ Action model for a swing foot phase.

        :param timeStep: step duration of the action model
        :param supportFootIds: Ids of the constrained feet
        :param comTask: CoM task
        :param swingFootTask: swinging foot task
        :return action model for a swing foot phase
        """
        # Creating a 3D multi-contact model, and then including the supporting
        # foot
        nu = self.actuation.nu
        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        for i in supportFootIds:
            # supportContactModel = crocoddyl.ContactModel3D(self.state, i, np.array([0., 0., 0.]), nu,
                                                        #    np.array([0., 50.]))
            initial_foot_placement = self.rdata.oMf[i]                                                        
            contact_location = crocoddyl.FrameTranslation(i, initial_foot_placement.translation)
            supportContactModel = crocoddyl.ContactModel2D(self.state, contact_location, 
                                                            nu, np.array([0., 50]))
            # supportContactModel = crocoddyl.ContactModel2D(self.state, i, np.array([0., 0.]), nu,
            #                                              np.array([0., 50.]))
            contactModel.addContact(self.rmodel.frames[i].name + "_contact", supportContactModel)

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, nu)
        if isinstance(comTask, np.ndarray):
            comResidual = crocoddyl.ResidualModelCoMPosition(self.state, comTask, nu)
            comTrack = crocoddyl.CostModelResidual(self.state, comResidual)
            costModel.addCost("comTrack", comTrack, 1e6)
        # for i in supportFootIds:
            # coneResidual = crocoddyl.ResidualModelContactControlGrav(self.state, nu) # add cost without the cone 
            # cone = crocoddyl.FrictionCone(self.Rsurf, self.mu, 2, False)
            # coneResidual = crocoddyl.ResidualModelContactFrictionCone(self.state, i, cone, nu)
            # coneActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub))
            # frictionCone = crocoddyl.CostModelResidual(self.state, coneActivation, coneResidual)
            # frictionCone = crocoddyl.CostModelResidual(self.state, coneResidual)
            # costModel.addCost(self.rmodel.frames[i].name + "_frictionCone", frictionCone, 1e6)
        if swingFootTask is not None:
            for i in swingFootTask:
                frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(self.state, i[0], i[1].translation,
                                                                                   nu)
                footTrack = crocoddyl.CostModelResidual(self.state, frameTranslationResidual)
                costModel.addCost(self.rmodel.frames[i[0]].name + "_footTrack", footTrack, 1e6)

        stateResidual = crocoddyl.ResidualModelState(self.state, self.rmodel.defaultState, nu)
        stateActivation = crocoddyl.ActivationModelWeightedQuad(self.stateWeights**2)
        ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
        stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)
        ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)
        costModel.addCost("stateReg", stateReg, 1e1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-1)

        lb = np.concatenate([self.state.lb[1:self.state.nv + 1], self.state.lb[-self.state.nv:]])
        ub = np.concatenate([self.state.ub[1:self.state.nv + 1], self.state.ub[-self.state.nv:]])
        stateBoundsResidual = crocoddyl.ResidualModelState(self.state, nu)
        stateBoundsActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lb, ub))
        stateBounds = crocoddyl.CostModelResidual(self.state, stateBoundsActivation, stateBoundsResidual)
        costModel.addCost("stateBounds", stateBounds, 1e3)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contactModel,
                                                                     costModel, 0., True)
        model = crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)
        return model