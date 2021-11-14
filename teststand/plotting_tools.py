from numpy.core.fromnumeric import shape
import pinocchio as pin
import crocoddyl
import numpy as np
import matplotlib.pyplot as plt

def plotSolutionOneLeg(solver, bounds = True):
    rmodel = solver.problem.runningModels[0].state.pinocchio
    xs, us = solver.xs, solver.us
    
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
    legJointNames = ['HAA', 'HFE', 'KFE']
    colors = ['royalblue','orange','mediumseagreen']
    # LF foot
    plt.subplot(1, 3, 1)
    plt.title('joint position [rad]')
    [plt.plot(X[k], label = legJointNames[i]) for i, k in enumerate(range(1, 4))]
    if bounds:
        [plt.plot(X_LB[k], linestyle='dashed', color = colors[i]) for i, k in enumerate(range(1, 4))]
        [plt.plot(X_UB[k], linestyle='dashed', color = colors[i]) for i, k in enumerate(range(1, 4))]
    plt.ylabel('LF')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.title('joint velocity [rad/s]')
    [plt.plot(X[k], label=legJointNames[i], color = colors[i]) for i, k in enumerate(range(nq + 1, nq + 4))]
    if bounds:
        [plt.plot(X_LB[k], linestyle='dashed', color = colors[i]) for i, k in enumerate(range(nq + 1, nq + 4))]
        [plt.plot(X_UB[k], linestyle='dashed', color = colors[i]) for i, k in enumerate(range(nq + 1, nq + 4))]
    plt.ylabel('LF')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.title('joint torque [Nm]')
    [plt.plot(U[k], label=legJointNames[i], color = colors[i]) for i, k in enumerate(range(0, 3))]
    if bounds:
        [plt.plot(U_LB[k], linestyle='dashed', color = colors[i]) for i, k in enumerate(range(0, 3))]
        [plt.plot(U_UB[k], linestyle='dashed', color = colors[i]) for i, k in enumerate(range(0, 3))]
    plt.ylabel('LF')
    plt.legend()
    
    us_lb, us_ub = [], []
    xs_lb, xs_ub = [], []        
    
    plt.figure()
    plt.suptitle('CoM and body positions')
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
    plt.title('CoM position xy')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid(True)
    plt.subplot(1, 3, 2)
    plt.plot(Cx, Cz)
    plt.title('CoM position xz')
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.grid(True)
    plt.subplot(1, 3, 3)
    plt.plot(X[0])
    plt.title('Body height')
    plt.ylabel('z')
    plt.grid(True)
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