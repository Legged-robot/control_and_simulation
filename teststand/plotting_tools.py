from numpy.core.fromnumeric import shape
import pinocchio as pin
import crocoddyl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator, AutoMinorLocator

def plotSolutionOneLeg(solver, dt, bounds = True):
    rmodel = solver.problem.runningModels[0].state.pinocchio
    xs, us = solver.xs, solver.us
    data_coef = [-1. ,1. ,-1.] # data adjustments if on real robot direction is reverced

    if bounds:
        us_lb, us_ub = [], []
        xs_lb, xs_ub = [], []
    models = solver.problem.runningModels.tolist() + [solver.problem.terminalModel]
    for m in models:
        us_lb += [m.u_lb]
        us_ub += [m.u_ub]
        xs_lb += [m.state.lb]
        xs_ub += [m.state.ub]

    # Getting the state and control trajectories
    nx, nq, nu = xs[0].shape[0], rmodel.nq, us[0].shape[0]
    X = [0.] * nx
    U = [0.] * nu
    if bounds:
        U_LB = [0.] * nu
        U_UB = [0.] * nu
        X_LB = [0.] * nx
        X_UB = [0.] * nx
    for i in range(nx):
        X[i] = [np.asscalar(x[i]) for x in xs]
        if bounds:
            X_LB[i] = [np.asscalar(x[i]) for x in xs_lb]
            X_UB[i] = [np.asscalar(x[i]) for x in xs_ub]
    for i in range(nu):
        U[i] = [np.asscalar(u[i]) if u.shape[0] != 0 else 0 for u in us]
        if bounds:
            U_LB[i] = [np.asscalar(u[i]) if u.shape[0] != 0 else np.nan for u in us_lb]
            U_UB[i] = [np.asscalar(u[i]) if u.shape[0] != 0 else np.nan for u in us_ub]

    # Plotting the joint positions, velocities and torques
    plt.figure()
    plt.suptitle('Simulated jumping motion - joints trajectory')
    legJointNames = ['Hip', 'Knee', 'Ankle']
    colors = ['royalblue','orange','mediumseagreen']
    # LF foot
    plt.subplot(1, 3, 1)
    # plt.title('Joint position')
    [plt.plot([i*dt for i in range(len(X[k]))], data_coef[i]*np.array(X[k]), label = legJointNames[i]) for i, k in enumerate(range(1, 4))]
    if bounds:
        [plt.plot([i*dt for i in range(len(X_LB[k]))], X_LB[k], linestyle='dashed', color = colors[i]) for i, k in enumerate(range(1, 4))]
        [plt.plot([i*dt for i in range(len(X_UB[k]))], X_UB[k], linestyle='dashed', color = colors[i]) for i, k in enumerate(range(1, 4))]
    plt.ylabel('position [rad]')
    plt.xlabel('time [s]')
    plt.legend(loc='upper left')
    plt.subplot(1, 3, 2)
    # plt.title('Joint velocity')
    [plt.plot([i*dt for i in range(len(X[k]))], data_coef[i]*np.array(X[k]), label=legJointNames[i], color = colors[i]) for i, k in enumerate(range(nq + 1, nq + 4))]
    if bounds:
        [plt.plot([i*dt for i in range(len(X_LB[k]))], X_LB[k], linestyle='dashed', color = colors[i]) for i, k in enumerate(range(nq + 1, nq + 4))]
        [plt.plot([i*dt for i in range(len(X_UB[k]))], X_UB[k], linestyle='dashed', color = colors[i]) for i, k in enumerate(range(nq + 1, nq + 4))]
    plt.ylabel('velocity [rad/s]')
    plt.xlabel('time [s]')
    plt.legend(loc='upper left')
    plt.subplot(1, 3, 3)
    # plt.title('Joint torque')
    [plt.plot([i*dt for i in range(len(U[k]))], data_coef[i]*np.array(U[k]), label=legJointNames[i], color = colors[i]) for i, k in enumerate(range(0, 3))]
    if bounds:
        [plt.plot([i*dt for i in range(len(U_LB[k]))], U_LB[k], linestyle='dashed', color = colors[i]) for i, k in enumerate(range(0, 3))]
        [plt.plot([i*dt for i in range(len(U_UB[k]))], U_UB[k], linestyle='dashed', color = colors[i]) for i, k in enumerate(range(0, 3))]
    plt.ylabel('torque  [N*m]')
    plt.xlabel('time [s]')
    plt.legend(loc='upper left')
    
    us_lb, us_ub = [], []
    xs_lb, xs_ub = [], []        
    
    plt.figure()
    plt.suptitle('CoM position and body height analysis')
    rdata = rmodel.createData()
    Cx = []
    Cy = []
    Cz = []
    for x in xs:
        q = x[:rmodel.nq]
        c = pin.centerOfMass(rmodel, rdata, q)
        Cx.append(np.asscalar(c[0]))
        Cy.append(np.asscalar(c[1]))
        Cz.append(np.asscalar(c[2]))

    plt.subplot(1, 3, 1)
    plt.plot(Cx, Cy)
    plt.title('CoM xy plane')
    plt.xlabel('x position [m]')
    plt.ylabel('y position [m]')
    plt.grid(True)
    plt.subplot(1, 3, 2)
    plt.plot(Cx, Cz)
    plt.title('CoM xz plane')
    plt.xlabel('x position [m]')
    plt.ylabel('z position [m]')
    plt.grid(True)
    plt.subplot(1, 3, 3)
    plt.plot([i*dt for i in range(len(X[0]))], X[0])
    plt.title('Body height through time')
    plt.ylabel('z position [m]')
    plt.xlabel('time [s]')
    plt.grid(True)
    # plt.show()
    return
    
def plotSolutionOneLegSeparate(solver, dt, bounds = True):
    rmodel = solver.problem.runningModels[0].state.pinocchio
    xs, us = solver.xs, solver.us
    plot_m = 3
    data_coef = [-1. ,1. ,-1.] # data adjustments if on real robot direction is reverced
    
    if bounds:
        us_lb, us_ub = [], []
        xs_lb, xs_ub = [], []
        models = solver.problem.runningModels.tolist() + [solver.problem.terminalModel]
        for m in models:
            us_lb += [m.u_lb]
            us_ub += [m.u_ub]
            xs_lb += [m.state.lb]
            xs_ub += [m.state.ub]
    
    colorst = ['orange','brown','purple'] 

    # Getting the state and control trajectories
    nx, nq, nu = xs[0].shape[0], rmodel.nq, us[0].shape[0]
    X = [0.] * nx
    U = [0.] * nu
    if bounds:
        U_LB = [0.] * nu
        U_UB = [0.] * nu
        X_LB = [0.] * nx
        X_UB = [0.] * nx
    for i in range(nx):
        X[i] = [np.asscalar(x[i]) for x in xs]
        if bounds:
            X_LB[i] = [np.asscalar(x[i]) for x in xs_lb]
            X_UB[i] = [np.asscalar(x[i]) for x in xs_ub]
    for i in range(nu):
        U[i] = [np.asscalar(u[i]) if u.shape[0] != 0 else 0 for u in us]
        if bounds:
            U_LB[i] = [np.asscalar(u[i]) if u.shape[0] != 0 else np.nan for u in us_lb]
            U_UB[i] = [np.asscalar(u[i]) if u.shape[0] != 0 else np.nan for u in us_ub]

    fig, axs = plt.subplots(plot_m, plot_m, sharex='col')
    legJointNames = ['Hip', 'Knee', 'Ankle']
    colorsb = ['royalblue','orange','mediumseagreen']
    labels = ["Position","Velocity","Torque"]

    # Ploting the positions
    for i, k in enumerate(range(1, 4)):
        axs[0, i].plot([i*dt for i in range(len(X[k]))], data_coef[i]*np.array(X[k]), label = labels[0], color = colorst[0])
        if bounds:
            axs[0, i].plot([i*dt for i in range(len(X_LB[k]))], X_LB[k], linestyle='dashed', color = colorst[0])
            axs[0, i].plot([i*dt for i in range(len(X_UB[k]))], X_UB[k], linestyle='dashed', color = colorst[0]) 
        axs[0, i].title.set_text('{} Actuator'.format(legJointNames[i]))
        axs[0, i].set_ylabel('rad')
        # axs[0, i].set_ylabel('position [rad]')
        axs[0, i].grid(True)
        axs[0, i].legend(loc='upper left')
    # Plotting the joint positions, velocities and torques
    
    for i, k in enumerate(range(nq + 1, nq + 4)):
        axs[1, i].plot([i*dt for i in range(len(X[k]))], data_coef[i]*np.array(X[k]), label = labels[1], color = colorst[1])
        if bounds:
            axs[1, i].plot([i*dt for i in range(len(X_LB[k]))], X_LB[k], linestyle='dashed', color = colorst[1])
            axs[1, i].plot([i*dt for i in range(len(X_UB[k]))], X_UB[k], linestyle='dashed', color = colorst[1]) 
        axs[1, i].set_ylabel('rad/s')
        # axs[1, i].set_ylabel('velocity [rad/s]')
        axs[1, i].grid(True)
        axs[1, i].legend(loc='upper left')

    for i, k in enumerate(range(0, 3)):
        axs[2, i].plot([i*dt for i in range(len(U[k]))], data_coef[i]*np.array(U[k]), label = labels[2], color = colorst[2])
        if bounds:
            axs[2, i].plot([i*dt for i in range(len(U_LB[k]))], U_LB[k], linestyle='dashed', color = colorst[2])
            axs[2, i].plot([i*dt for i in range(len(U_UB[k]))], U_UB[k], linestyle='dashed', color = colorst[2]) 
        axs[2, i].set_ylabel('N*m')
        # axs[2, i].set_ylabel('torque  [N*m]')
        axs[2, i].grid(True)
        axs[2, i].legend(loc='upper left')  
        axs[2, i].set_xlabel('time [s]')
    
    fig.suptitle('Simulated jumping motion - joints trajectory')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0)

    fig, axs = plt.subplots(1, 3)
    fig.suptitle('CoM position and body height analysis')
    rdata = rmodel.createData()
    Cx = []
    Cy = []
    Cz = []
    for x in xs:
        q = x[:rmodel.nq]
        c = pin.centerOfMass(rmodel, rdata, q)
        Cx.append(np.asscalar(c[0]))
        Cy.append(np.asscalar(c[1]))
        Cz.append(np.asscalar(c[2]))

    axs[0].plot(Cx, Cy, label = 'CoM y position')
    axs[0].title.set_text('CoM xy-plane')
    axs[0].set_xlabel('x position [m]')
    axs[0].set_ylabel('m')
    axs[0].grid(True)
    axs[0].xaxis.set_major_locator(MultipleLocator(0.01))
    axs[0].legend(loc='upper left')
    axs[0].xaxis.set_minor_locator(AutoMinorLocator(5))
    axs[0].grid(which='minor', color='#CCCCCC', linestyle=':')
    axs[0].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    axs[1].plot(Cx, Cz, label = 'CoM z position')
    axs[1].title.set_text('CoM xz-plane')
    axs[1].set_xlabel('x position [m]')
    axs[1].set_ylabel('m')
    axs[1].grid(True)
    axs[1].xaxis.set_major_locator(MultipleLocator(0.01))
    axs[1].legend(loc='upper left')
    axs[1].xaxis.set_minor_locator(AutoMinorLocator(5))
    axs[1].grid(which='minor', color='#CCCCCC', linestyle=':')
    axs[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    axs[2].plot([i*dt for i in range(len(X[0]))], X[0], label = 'Body z position')
    axs[2].title.set_text('Body height through time')
    axs[2].legend(loc='upper left')
    axs[2].set_ylabel('m')
    axs[2].set_xlabel('time [s]')
    axs[2].grid(True)
    axs[2].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    
    # plt.show()
    return
        
def plotConvergence(costs, muLM, muV, gamma, theta, alpha, figIndex=1, show=True, figTitle=""):
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.figure(figIndex, figsize=(6.4, 8))

    # Plotting the total cost sequence
    plt.subplot(611)
    plt.ylabel("cost")
    plt.plot(costs)
    plt.title(figTitle, fontsize=14)

    # Ploting mu sequences
    plt.subplot(612)
    plt.ylabel("muLM")
    plt.plot(muLM)
    plt.subplot(613)
    plt.ylabel("muV")
    plt.plot(muV)


    # Plotting the gradient sequence (gamma and theta)
    plt.subplot(614)
    plt.ylabel("gamma")
    plt.plot(gamma)
    plt.subplot(615)
    plt.ylabel("theta")
    plt.plot(theta)


    # Plotting the alpha sequence
    plt.subplot(616)
    plt.ylabel("alpha")
    ind = np.arange(len(alpha))
    plt.bar(ind, alpha)
    plt.xlabel("iteration")
    if show:
        plt.ion()
        plt.show()