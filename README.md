# pscrCL

A OpenCL-based PSCR solver.

Mirko Myllykoski: On GPU-accelerated fast direct solvers and their applications in image denoising, University of Jyväskylä, 2015, https://jyx.jyu.fi/dspace/handle/123456789/46733

Mirko Myllykoski, Tuomo Rossi, Jari Toivanen: On solving separable block tridiagonal linear systems using a GPU implementation of radix-4 PSCR method (draft article), https://mmyllykoski.files.wordpress.com/2016/05/sha3_noheader.pdf

CAUTION: This implementation has not been tested since 2015.

## API

See `L1SolverContext.h`, `L2SolverContext.h`, and `pscrCL.h`.

## Example

```
// Allocate factor matrices
double a1Diag[n1], a1OffDiag[n1], m1Diag[n1];
double a2Diag[n2], a2OffDiag[n2], m2Diag[n2];
for(int i = 0; i < n1; i++) {
		a1Diag[i] = 2.0;
		a1OffDiag[i] = -1.0;
		m1Diag[i] = 1;
}
for(int i = 0; i < n2; i++) {
		a2Diag[i] = 2.0;
		a2OffDiag[i] = -1.0;
		m2Diag[i] = 1;
}

// Use double precicion real numbers
pscrCL::PscrCLMode mode(PSCRCL_PREC_DOUBLE | PSCRCL_NUM_REAL);

// Add a OpenCL context (cl::Context) into a vector
std::vector<cl::Context> contexts;
contexts.push_back(context);

// Add a OpenCL device (cl::Device) into a vector
std::vector<cl::Device> devices;
devices.push_back(device);

// Use default configuration
std::vector<pscrCL::OptValues> optValues = pscrCL::L2SolverContext::getDefaultValues(devices, n1, mode);

// Initialize solver (ldf is the right-hand side leading dimension)
pscrCL::L2SolverContext solver(contexts, devices, optValues, a1Diag, a1OffDiag, m1Diag, 0, a2Diag, a2OffDiag, m2Diag, 0, n1, n2, n1, ldf, mode);

// Add a OpenCL queue (cl::CommandQueue) into a vector
std::vector<pscrCL::CommandQueue> queues;
queues.push_back(queue);

// Allocate and initialize device-side data structures
solver->allocate(queues);

// Allocate a temporary device-side buffer
solver->allocateTmp();

// Solve system [A_1 (x) M_2 + M_1 (x) A_2 + ch * M_1 (x) M_2] u = f
double ch = 0.2;
solver->run(queues, devMemF, 1, &ch);
```
