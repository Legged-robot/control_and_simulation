from numpy.core.shape_base import block
import pinocchio as pin
import crocoddyl
import numpy as np
import matplotlib.pyplot as plt
from teststand.quadrupedal_gait_problem import SimpleQuadrupedalGaitProblem
from teststand.robot_loader import load
from teststand.plotting_tools import plotSolutionOneLeg, plotConvergence
from teststand.logging_tools import CallbackLogger, writeOneLegDataToFile
# from crocoddyl.utils.quadruped import plotSolution

PRISMATIC = True
DISPLAY = True
PLOT = True
WRITE_TO_FILE = True
TIME_STRETCH_FACTOR = 4.
CONTINIOUS_DISPLAY = True
# fore.initViewer(loadModel=True)
# fore.display(fore.q0)

if PRISMATIC:
    fore = load('fore_prismatic')
else: 
    fore = load('fore_freeflyer')

print("Number of degrees of freedom: ",fore.model.nq)

# Defining the initial state of the robot
q0 = fore.q0.copy()
v0 = pin.utils.zero(fore.model.nv)
x0 = np.concatenate([q0, v0])
# x0[0]+=0.001
print("x0[q0, v0] state:", x0)
print("Default robot state joint:", fore.model.referenceConfigurations["standing"])
lfFoot = 'LF_FOOTPOINT'

if PRISMATIC:
    stateWeights = np.array([10.] * 1 + [.1] * (fore.model.nv - 1) + [10.] * 1 + [1.] *
                            (fore.model.nv - 1))    #    For prismatic joint
    # stateWeights = np.array([0.] * 1 + [0.0] * (fore.model.nv - 1) + [0.] * 1 + [0.] *
    #                         (fore.model.nv - 1))    #    For prismatic joint
else:
    stateWeights = np.array([0.] * 3 + [500.] * 3 + [0.01] * (fore.model.nv - 6) + [10.] * 6 + [1.] *
                            (fore.model.nv - 6))  #    For freeflyer joint

gait = SimpleQuadrupedalGaitProblem(fore.model, [lfFoot], stateWeights, x0)

# Setting up all tasks
GAITPHASES = {
    'jumping': {
        'jumpHeight': 0.1,
        'jumpLength': [0., 0., 0.],
        'timeStep': 9.5e-3,
        'groundKnots': 10,
        'flyingKnots': 16,
        'flyingKnotsRepeat': 1
    }
}

cameraTF = [2., 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]

value = GAITPHASES['jumping']
ddp = crocoddyl.SolverFDDP(
    gait.createJumpingProblem(x0, value['jumpHeight'], value['jumpLength'], value['timeStep'],
                              value['groundKnots'], value['flyingKnots'], value['flyingKnotsRepeat']))

# Show the solution itterations
callbacks = []
if PLOT:
    callbacks += [CallbackLogger()]

# iter     cost         stop         grad         xreg         ureg       step    ||ffeas||
# 
# cost - Total cost
# stop - Value computed by stoppingCriteria()
# grad - LQ approximation of the expected improvement.
# xreg, ureg - Current state and control regularization values.      
# step - Current applied step-length. 
# ||ffeas|| - Return the feasibility of the dynamic constraints ||ffeas||inf,1 of the current guess
callbacks += [crocoddyl.CallbackVerbose()]

if DISPLAY:
    # display = crocoddyl.GepettoDisplay(fore, 1, 1, cameraTF, frameNames=[lfFoot])
    display = crocoddyl.GepettoDisplay(fore, 1, 1, frameNames=[lfFoot])
    callbacks += [crocoddyl.CallbackDisplay(display)]
    
ddp.setCallbacks(callbacks)

xs = [fore.model.defaultState] * (ddp.problem.T + 1)
us = ddp.problem.quasiStatic([fore.model.defaultState] * ddp.problem.T)
# xs = []
# us = []
solution = ddp.solve(xs, us, 1000, False, value['timeStep']/2)
print(solution)



# if PLOT and not PRISMATIC:
if WRITE_TO_FILE:
    writeOneLegDataToFile(ddp, file_name = 'oneLegData.txt', w_torques = False)

if PLOT:
    log = ddp.getCallbacks()[0]
#     plotSolution(ddp, figIndex=1, show=False) # error expected since the function expects 4 legs
    plotSolutionOneLeg(ddp)
    title = list(GAITPHASES.keys())[0] + " (phase " + str(0) + ")"
    plotConvergence(costs       = log.costs, 
                    muLM        = log.u_regs,
                    muV         = log.x_regs,
                    gamma       = log.grads, 
                    theta       = log.stops,
                    alpha       = log.steps,
                    figTitle    = title,
                    figIndex    = 3,
                    show        = True  )

if DISPLAY:
    # Display the entire motion
    display = crocoddyl.GepettoDisplay(fore, frameNames=[lfFoot])
    while(CONTINIOUS_DISPLAY):
        get_char = input("Waiting for keyboard input to display solver solution in GUI. \n Optionaly enter the time stretch factor (number): ")
        try:
            factor = float(get_char)
            print("Displaying with stretch factor: ",factor )
        except ValueError:
            factor = TIME_STRETCH_FACTOR
            print("Displaying with default value for stretch factor: ", TIME_STRETCH_FACTOR)    
        display.displayFromSolver(ddp, factor=factor)

input("Press key to exit the program")
 