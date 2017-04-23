vars = Variables()
vars.Add('DEBUG', '', 1)
vars.Add('FULL_DEBUG', '', 0)

env = Environment(variables = vars)
env.Append(CPPDEFINES={
	'DEBUG' : '${DEBUG}', 
	'FULL_DEBUG' : '${FULL_DEBUG}'})

#env.Append(CPPPATH=['/usr/local/cuda-5.5/include'])
#env.Append(CPPPATH=['/usr/local/cuda/include'])

env.Append(CCFLAGS = ['-Wall', '-O2'])

sources = Split("""
Boundaries.cpp
CommandQueue.cpp
cl/source.cpp
common.cpp
DeviceContextHelper.cpp
DeviceInformation.cpp
eigen/EigenContainer.cpp
Guide.cpp
KernelHelper.cpp
KernelLaunch.cpp
L1DeviceContext.cpp
L1OptimizerHelper.cpp
L1SolverContext.cpp
L2DeviceContext.cpp
L2OptimizerHelper.cpp
L2SolverContext.cpp
L3DeviceContext.cpp
L3OptimizerHelper.cpp
MatrixContainer.cpp
optimizer/Optimizer.cpp
optimizer/OptParam.cpp
optimizer/OptValues.cpp
""")

object_list = env.Object(source = sources)

env.Command("cl/common.cl.dat", "cl/common.cl", "cd cl/ && xxd -i common.cl > common.cl.dat")
env.Command("cl/l3_kernel.cl.dat", "cl/l3_kernel.cl", "cd cl/ && xxd -i l3_kernel.cl > l3_kernel.cl.dat")
env.Command("cl/lx_kernel.cl.dat", "cl/lx_kernel.cl", "cd cl/ && xxd -i lx_kernel.cl > lx_kernel.cl.dat")

Library('pscrCL', object_list)