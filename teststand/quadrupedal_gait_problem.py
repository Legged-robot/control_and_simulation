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
        # self.lfFootId = self.feet[0]
        # self.rfFootId = self.feet[1]
        # self.lhFootId = self.feet[2]
        # self.rhFootId = self.feet[3]

        # Defining default state
        self.rmodel.defaultState = defaultState
        # q0 = self.rmodel.referenceConfigurations["standing"]
        # self.rmodel.defaultState = np.concatenate([q0, np.zeros(self.rmodel.nv)])
        self.firstStep = True
        # Defining the friction coefficient and normal
        self.mu = 0.7
        self.Rsurf = np.eye(3)
        
    def createJumpingProblem(self, x0, jumpHeight, jumpLength, timeStep, groundKnots, flyingKnots, flyingKnotsRepeat):
        q0 = x0[:self.rmodel.nq]
        pin.forwardKinematics(self.rmodel, self.rdata, q0)
        pin.updateFramePlacements(self.rmodel, self.rdata)
        # FootPos0 = []
        # stateRef = 0
        # for foot in self.feet:
        #     FootPos0 += [self.rdata.oMf[foot].translation]
        #     FootPos0[-1][2] = 0
        #     stateRef += FootPos0[-1]
        # stateRef /= 4
        comRef = pin.centerOfMass(self.rmodel, self.rdata, q0) # assumption is that the intitial com is correct
        stateRef = x0 
        print("Com reference and state reference: ", comRef, stateRef)
        # rfFootPos0 = self.rdata.oMf[self.feet[1]].translation
        # rhFootPos0 = self.rdata.oMf[self.feet[3]].translation
        # lfFootPos0 = self.rdata.oMf[self.feet[0]].translation
        # lhFootPos0 = self.rdata.oMf[self.feet[2]].translation
        # df = jumpLength[2] - rfFootPos0[2]
        # rfFootPos0[2] = 0.
        # rhFootPos0[2] = 0.
        # lfFootPos0[2] = 0.
        # lhFootPos0[2] = 0.
        
        # stateRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        

        loco3dModel = []
        takeOff = [
            self.createSwingFootModel(
                timeStep,
                supportFootIds = self.feet,
                stateTask = stateRef
            ) for k in range(groundKnots)
        ]
        flyingUpPhase = [
            self.createSwingFootModel(
                timeStep, 
                supportFootIds = [],
                stateTask = np.concatenate(([x0[0] + jumpHeight * (k + 1) / flyingKnots], x0[1:])),
                comTask = np.array([jumpHeight, 0., 0.]) * (k + 1) / flyingKnots + comRef
                # np.array([jumpLength[0], jumpLength[1], jumpLength[2] + jumpHeight]) * (k + 1) / flyingKnots + stateRef)
            ) for k in np.repeat(range(flyingKnots),flyingKnotsRepeat)
        ]
        flyingDownPhase = [
            self.createSwingFootModel(
                timeStep, 
                supportFootIds = [],
                stateTask = stateRef,
                comTask = comRef
            ) for k in np.repeat(range(flyingKnots),flyingKnotsRepeat)
        ]
        #  f0 = jumpLength
        # footTask = [
        #     crocoddyl.FramePlacement(self.lfFootId, pin.SE3(np.eye(3), lfFootPos0 + f0)),
        #     crocoddyl.FramePlacement(self.rfFootId, pin.SE3(np.eye(3), rfFootPos0 + f0)),
        #     crocoddyl.FramePlacement(self.lhFootId, pin.SE3(np.eye(3), lhFootPos0 + f0)),
        #     crocoddyl.FramePlacement(self.rhFootId, pin.SE3(np.eye(3), rhFootPos0 + f0))
        # ]
        # landingPhase = [
        #     self.createFootSwitchModel([self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId], footTask, False)
        # ]
        # f0[2] = df
        landed = [
            self.createSwingFootModel(
                timeStep, 
                supportFootIds = self.feet,
                stateTask = stateRef,
                comTask = comRef
            ) for k in range(groundKnots)
        ]
        loco3dModel += takeOff
        loco3dModel += flyingUpPhase
        loco3dModel += flyingDownPhase
        # loco3dModel += landingPhase
        loco3dModel += landed

        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem
    
    def createSwingFootModel(self, timeStep, supportFootIds, stateTask=None, comTask = None, swingFootTask=None):
        """ Action model for a swing foot phase.

        :param timeStep: step duration of the action model
        :param supportFootIds: Ids of the constrained feet
        :param stateTask: State task
        :param swingFootTask: swinging foot task
        :return action model for a swing foot phase
        """
        # Creating a 3D multi-contact model, and then including the supporting
        # foot
        print("stateTask",stateTask)
        nu = self.actuation.nu
        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        for i in supportFootIds:
            # supportContactModel = crocoddyl.ContactModel3D(self.state, i, np.array([0., 0., 0.]), nu,
            #                                                np.array([0., 50.]))
            # oMf_i = self.rdata.oMf[i] 
            # print(oMf_i.translation[0])
            supportContactModel = crocoddyl.ContactModel2D(self.state, i, np.array([0., 0.]), nu,
                                                           np.array([80., 4.]))
            contactModel.addContact(self.rmodel.frames[i].name + "_contact", supportContactModel)

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, nu)

        # State Task 
        if isinstance(stateTask, np.ndarray):
            stateResidual = crocoddyl.ResidualModelState(self.state, stateTask, nu)
            stateActivation = crocoddyl.ActivationModelWeightedQuad(self.stateWeights**2)
            stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)
            costModel.addCost("stateReg", stateReg, 1.1e4)

        # Com Task to ensure better jumping motion
        if isinstance(comTask,np.ndarray):
            comResidual = crocoddyl.ResidualModelCoMPosition(self.state, comTask, nu)
            comTrack = crocoddyl.CostModelResidual(self.state, comResidual)
            costModel.addCost("comTrack", comTrack, 2e2)

        # Contact 
        for i in supportFootIds:
            cone = crocoddyl.FrictionCone(self.Rsurf, self.mu, 4, False)
            coneResidual = crocoddyl.ResidualModelContactFrictionCone(self.state, i, cone, nu)
            coneActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub))
            frictionCone = crocoddyl.CostModelResidual(self.state, coneActivation, coneResidual)
            costModel.addCost(self.rmodel.frames[i].name + "_frictionCone", frictionCone, 4.5e3)

        # Walking (not yet implemented)
        if swingFootTask is not None:
            for i in swingFootTask:
                frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(self.state, i[0], i[1].translation,
                                                                                   nu)
                footTrack = crocoddyl.CostModelResidual(self.state, frameTranslationResidual)
                costModel.addCost(self.rmodel.frames[i[0]].name + "_footTrack", footTrack, 1e6)

        ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)

        # print("Effort limits pinocchio", self.state.pinocchio.effortLimit)
        
        # Control bounds and regularization
        ctrlBoundsActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(-1.0 * self.state.pinocchio.effortLimit[1:], self.state.pinocchio.effortLimit[1:]))
        ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlBoundsActivation, ctrlResidual)
        costModel.addCost("ctrlReg", ctrlReg, 10e2)

        # State bounds and regularization
        lb = np.concatenate([self.state.lb[1:self.state.nv + 1], self.state.lb[-self.state.nv:]])
        ub = np.concatenate([self.state.ub[1:self.state.nv + 1], self.state.ub[-self.state.nv:]])
        stateBoundsResidual = crocoddyl.ResidualModelState(self.state, nu)
        stateBoundsActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lb, ub))
        stateBounds = crocoddyl.CostModelResidual(self.state, stateBoundsActivation, stateBoundsResidual)
        costModel.addCost("stateBounds", stateBounds, 4e2)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contactModel,
                                                                     costModel, 0., True)
        model = crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)
        return model