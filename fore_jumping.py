import pinocchio as pin
import crocoddyl
import numpy as np
from teststand_quadrupedal_gait_problem import SimpleQuadrupedalGaitProblem
from teststand_robot_loader import load
from teststand_plotting_tools import plotSolutionOneLeg

PRISMATIC = False
DISPLAY = True
PLOT = True
fore = load('fore')
print("Number of degrees of freedom: ",fore.model.nq)
# fore.initViewer(loadModel=True)
# fore.display(fore.q0)
from crocoddyl.utils.quadruped import plotSolution

# Defining the initial state of the robot
q0 = fore.model.referenceConfigurations['standing'].copy()
v0 = pin.utils.zero(fore.model.nv)
x0 = np.concatenate([q0, v0])
lfFoot = 'LF_FOOTPOINT'

stateWeights = np.array([0.] * 3 + [500.] * 3 + [0.01] * (fore.model.nv - 6) + [10.] * 6 + [1.] *
                        (fore.model.nv - 6))  #    For freeflyer joint

# stateWeights = np.array([2.] * 1 + [0.01] * (fore.model.nv - 1) + [10.] * 1 + [1.] *
#                         (fore.model.nv - 1))    #    For prismatic joint

gait = SimpleQuadrupedalGaitProblem(fore.model, [lfFoot], stateWeights)

# Setting up all tasks
GAITPHASES = {
    'jumping': {
        'jumpHeight': 0.15,
        'jumpLength': [0.0, 0.3, 0.],
        'timeStep': 1e-2,
        'groundKnots': 10,
        'flyingKnots': 20
    }
}

cameraTF = [2., 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]

value = GAITPHASES['jumping']
ddp = crocoddyl.SolverFDDP(
    gait.createJumpingProblem(x0, value['jumpHeight'], value['jumpLength'], value['timeStep'],
                              value['groundKnots'], value['flyingKnots']))

# Show the solution itterations
callbacks = []
if PLOT:
    callbacks += [crocoddyl.CallbackLogger()]
    
callbacks += [crocoddyl.CallbackVerbose()]
if DISPLAY:
    display = crocoddyl.GepettoDisplay(fore, 4, 4, cameraTF, frameNames=[lfFoot])
    callbacks += [crocoddyl.CallbackDisplay(display)]
    
ddp.setCallbacks(callbacks)

xs = [fore.model.defaultState] * (ddp.problem.T + 1)
us = ddp.problem.quasiStatic([fore.model.defaultState] * ddp.problem.T)
solution = ddp.solve(xs, us, 10, False, 0.1)
print(solution)

if DISPLAY:
    # Display the entire motion
    display = crocoddyl.GepettoDisplay(fore, frameNames=[lfFoot])
    display.displayFromSolver(ddp)  

if PLOT and not PRISMATIC:
    log = ddp.getCallbacks()[0]
#     plotSolution(ddp, figIndex=1, show=False) # error expected since the function expects 4 legs
    plotSolutionOneLeg(ddp)
    title = list(GAITPHASES.keys())[0] + " (phase " + str(0) + ")"
    crocoddyl.plotConvergence(log.costs, 
                              log.u_regs,
                              log.x_regs,
                              log.grads,
                              log.stops,
                              log.steps,
                              figTitle=title,
                              figIndex=3,
                              show=True)