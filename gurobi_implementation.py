from gurobipy import *
import random
import numpy as np
from matplotlib import pyplot as plt
import time
import itertools



''' Problem Parameters '''
# Number of Machines
n_machine = 2


# Number of Jobs
n_job = 2


# A large number for constraint linearization
M = 100

# A large number for processing time of those machines that cannot process the operation
Mt = 100

# Maximum process time for an operation
maxt = 3


# Number of Operations for each job
n_oper = [(random.randint(2,3)) for i in range(n_job)]
n_op = sum(n_oper)



# Machines that can be used to process each operation
mach2oper = [(random.randint(1,n_machine)) for i in range(n_op)]
A = [[] for i in range(n_job)]
jj = 0
for i in range(n_job):
    for j in range(n_oper[i]):
        aa = random.sample(range(n_machine), mach2oper[jj])
        aa.sort()
        A[i].insert(j,aa)
        jj = jj + 1


# Process time for each machine
t = [[] for i in range(n_job)]
jj = 0
for i in range(n_job):
    for j in range(n_oper[i]):
        t[i].insert(j,[Mt]*n_machine)
        jj = jj + 1

for i in range(n_job):
    for j in range(n_oper[i]):
        for k in A[i][j]:
            t[i][j][k] = random.randint(1,maxt)



''' Gurobi Implementation '''

# Create Optimization model
model = Model("JSP")


''' Decision Variables'''
y = model.addVars(n_job,n_op,n_job,n_op,vtype=GRB.BINARY,name="y") # y[i,j,i',j']=1 if the same machine perform job i operation j after job i' operation j' where i'>i
c = model.addVars(n_job,n_op,n_machine,vtype=GRB.CONTINUOUS,lb=0,name="c") # represent the time when job i operation j is finished
x = model.addVars(n_job,n_op,n_machine,vtype=GRB.BINARY,name="x") # x[m,i,j]=1 if machine m process job i operation j
makespan = model.addVar(lb=0,ub=float('inf'),vtype=GRB.CONTINUOUS,name="makespan")

# Set variables x to zero in specific condition
# for i in range(n_op):
#         y[i,i] = 0


''' Constraints '''

for i in range(n_job):
    for j in range(n_oper[i]-1):
        for k in A[i][j+1]:
            # Precedence Constraint
            model.addConstr(quicksum(c[i,j,kk] for kk in A[i][j]) - (c[i,j+1,k] - t[i][j+1][k]) <= M*(1-x[i,j+1,k]))



for i in range(n_job):
    for j in range(n_oper[i]):
        for ii in range(n_job):
            if ii > i:
                for jj in range(n_oper[ii]):
                    avail_mach = list(set(A[i][j]) & set(A[ii][jj]))
                    for k in avail_mach:
                        # Order conflict resolution constraint on each machine
                        model.addConstr(c[i,j,k] - t[i][j][k] - c[ii,jj,k] >= -M*(1-y[i,j,ii,jj]) - M*(2-x[i,j,k]-x[ii,jj,k]))
                        model.addConstr(c[ii,jj,k] - t[ii][jj][k] - c[i,j,k] >= -M*y[i,j,ii,jj] - M*(2-x[i,j,k]-x[ii,jj,k]))


for i in range(n_job):
    for j in range(n_oper[i]):
        # Each operation is done by exactly one machine
        model.addConstr(quicksum(x[i,j,k] for k in A[i][j]) == 1)


for i in range(n_job):
    for k in A[i][0]:
        # Start time of each job is greater than zero
        model.addConstr(M*(1-x[i,0,k]) + c[i,0,k] - t[i][0][k] >= 0)


for i in range(n_job):
    for j in range(n_oper[i]):
        for k in A[i][j]:
            # If operation i is not assign to machine m, than z[m,i]=0
            model.addConstr(c[i,j,k] - M*x[i,j,k] <= 0)


for i in range(n_job):
    # Makespan definition
    model.addConstr(makespan >= quicksum(c[i,n_oper[i]-1,k] for k in A[i][n_oper[i]-1]))





''' Objective Function '''
# Minimize the makespan
model.setObjective(makespan,GRB.MINIMIZE)


#model.tune()
model.setParam("TimeLimit", 3600.0)


'''Solve the problem'''
model.optimize()



