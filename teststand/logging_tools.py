from crocoddyl.libcrocoddyl_pywrap import CallbackAbstract
import numpy as np


def writeOneLegDataToFile(solver, file_name = 'oneLegData.txt',  w_positions = True, w_velocities = True, w_torques = True):
    f = open(file_name, "w")   
    print("Writing data to a file: ",file_name)

    rmodel = solver.problem.runningModels[0].state.pinocchio
    xs, us = solver.xs, solver.us
    

    # Getting the state and control trajectories
    nx, nq, nu = xs[0].shape[0], rmodel.nq, us[0].shape[0]
    X = [0.] * nx
    U = [0.] * nu
    for i in range(nx):
        X[i] = [np.asscalar(x[i]) for x in xs]
    for i in range(nu):
        U[i] = [np.asscalar(u[i]) if u.shape[0] != 0 else 0 for u in us]

    # LF foot
    f.write("#define TIME_INDEXES {}\n".format(len(X[0])))
    if (w_positions):
        f.write("float pos[][TIME_INDEXES] = {\n") #format in c
        k_indx = range(1, 4)
        for i, k in enumerate(k_indx, 1):
            f.write("\t\t\t\t{")
            [f.write("{:.2f}, ".format(x)) for x in X[k][:-1]]  
            f.write("{:.2f} ".format(X[k][-1]))   #write last one without comma after it
            f.write("},\n") if k != k_indx[-1] else  f.write("}\n")
        f.write("\t\t\t\t};\n")

# int Matrix[][X] = { {1,2,3,4,...,X},
#                     {3,4,5,6,...,X},
#                     {1,9,2,8,...,X},
#                     {9,8,7,6,...,X} };

    if (w_velocities):
        f.write("float vel[][TIME_INDEXES] = {\n") #format in c
        k_indx = range(nq + 1, nq + 4)
        for i, k in enumerate(k_indx, 1):
            f.write("\t\t\t\t{")
            [f.write("{:.2f}, ".format(x)) for x in X[k][:-1]]  
            f.write("{:.2f} ".format(X[k][-1]))   #write last one without comma after it
            f.write("},\n") if k != k_indx[-1] else  f.write("}\n")
        f.write("\t\t\t\t};\n")

    if (w_torques):
        f.write("float torques[][TIME_INDEXES] = {\n") #format in c
        k_indx = range(0, 3)
        for i, k in enumerate(k_indx, 1):
            f.write("\t\t\t\t{")
            [f.write("{:.2f}, ".format(u)) for u in U[k][:-1]]  
            f.write("{:.2f} ".format(U[k][-1]))   #write last one without comma after it
            f.write("},\n") if k != k_indx[-1] else  f.write("}\n")
        f.write("\t\t\t\t};\n")

    print("Done")
    f.close()
    return

class CallbackLogger(CallbackAbstract):
    def __init__(self):
        CallbackAbstract.__init__(self)
        self.xs = []
        self.us = []
        self.fs = []
        self.steps = []
        self.iters = []
        self.costs = []
        self.u_regs = []
        self.x_regs = []
        self.stops = []
        self.grads = []

    def __call__(self, solver):
        import copy
        # costs = dict()
        # for i in range(solver.problem.T):
        #     for name,data in solver.problem.runningDatas[i]:
        #         if name in costs:
        #             costs[name]+=data.cost
        #         else:
        #             costs[name]=0
        # print("Printing individial costs")
        # for key in costs.keys():
        #     print(key,costs[key])
        self.xs = copy.copy(solver.xs)
        self.us = copy.copy(solver.us)
        self.fs.append(copy.copy(solver.fs))
        self.steps.append(solver.stepLength)
        self.iters.append(solver.iter)
        self.costs.append(solver.cost)
        self.u_regs.append(solver.u_reg)
        self.x_regs.append(solver.x_reg)
        self.stops.append(solver.stop)
        self.grads.append(solver.d[0])