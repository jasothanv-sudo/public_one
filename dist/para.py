"""
para.py
=======
User-editable parameters for the Sod shock-tube solver.

Everything a user is likely to want to change lives here.  `main.py` and
`roe_mod.py` read these values, so there is no need to touch the solver
code to run a different test case, refine the mesh, or change the output.

The default values reproduce the classic Sod (1978) shock-tube problem
and match the original Fortran implementation.
"""

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
GAMMA = 1.4          # ratio of specific heats (ideal diatomic gas)
R_GAS = 287.0        # specific gas constant [J/(kg*K)] -- used for temperature

# ---------------------------------------------------------------------------
# Computational domain and mesh
# ---------------------------------------------------------------------------
X_MIN = 0.0          # left end of the physical domain
X_MAX = 1.0          # right end of the physical domain
N_CELLS = 400        # number of interior cells (finite-volume cells)
X_DIAPHRAGM = 0.5    # initial location of the diaphragm / discontinuity

# ---------------------------------------------------------------------------
# Initial (Riemann) states:  primitive variables [rho, u, p]
#   left  state applies for x <  X_DIAPHRAGM
#   right state applies for x >= X_DIAPHRAGM
# ---------------------------------------------------------------------------
RHO_L, U_L, P_L = 1.000, 0.0, 1.0     # left  state (high pressure)
RHO_R, U_R, P_R = 0.125, 0.0, 0.1     # right state (low  pressure)

# ---------------------------------------------------------------------------
# Time integration
# ---------------------------------------------------------------------------
CFL = 0.75           # Courant number (must be < 1 for stability)
T_END = 0.20         # physical end time of the simulation
MAX_STEPS = 100000   # safety cap on the number of time steps

# Multi-stage (low-storage Runge-Kutta) coefficients.
# These reproduce the 6-stage scheme used in the reference Fortran code.
# Replace with e.g. [1.0] for plain forward-Euler, or a classic RK set.
RK_STAGES = [0.0742, 0.1393, 0.2198, 0.3302, 0.5182, 1.0000]

# ---------------------------------------------------------------------------
# Roe solver options
# ---------------------------------------------------------------------------
# Harten-Hyman entropy fix smooths the eigenvalues near sonic points and
# removes the small non-physical "kink" that a plain Roe solver leaves in
# the expansion fan.  Set to 0.0 to recover the plain Roe flux.
ENTROPY_FIX_EPS = 0.10

# ---------------------------------------------------------------------------
# Output / plotting
# ---------------------------------------------------------------------------
OUTPUT_FILE = "sod_output.dat"   # solution dump (plain text, column data)
SHOW_EXACT = True                # overlay the analytic Riemann solution
SAVE_FIGURE = True               # write the figure to disk
FIGURE_FILE = "sod_result.png"   # figure filename
SHOW_FIGURE = True               # pop up an interactive matplotlib window
