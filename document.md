# DHARA — Compressible Module Documentation

The compressible module solves the compressible Navier–Stokes (and, as a
limit, Euler) equations in 2D and 3D. Everything a user configures lives in a
single `para.py`; the solver core lives in the `DHARA/` package.

---

## 1. Installation Guide

### 1.1 System Requirements

**Minimum (CPU-only)**
- 64-bit OS: Linux (recommended), macOS, or Windows (WSL2 recommended).
- Python **3.8+** (3.10 recommended).
- 4 GB RAM minimum (8 GB+ recommended for larger 2D grids).
- ~2 GB disk for the environment, plus room for simulation output.

**Optional (GPU acceleration)**
- NVIDIA GPU (CUDA Compute Capability ≥ 3.5) with a working driver.
- CUDA Toolkit (§1.4) and CuPy (§1.5).

The backend is chosen at runtime by the `device` flag in `para.py`:
`'CPU'` uses NumPy, `'GPU'` uses CuPy. GPU components are only needed for
`device = 'GPU'`.

### 1.2 Install Conda and Create an Environment

An isolated Conda environment is recommended. Install **Miniconda**:

```bash
# Linux (x86_64)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Restart your shell (or `source ~/.bashrc`). Other installers are on the
[Miniconda page](https://docs.conda.io/en/latest/miniconda.html). Then:

```bash
conda create -n dhara_env python=3.10 -y
conda activate dhara_env
```

### 1.3 Install the Core Python Libraries

From the project root (folder containing `requirements.txt`):

```bash
pip install -r requirements.txt
```

This installs `numpy`, `scipy`, `h5py`, and `imageio` — enough to run the
compressible solver on **CPU**.

### 1.4 Install the CUDA Toolkit (GPU only)

Skip if running CPU-only.

1. Check the driver and its maximum supported CUDA version:
   ```bash
   nvidia-smi
   ```
2. Download and install a matching (or lower) CUDA Toolkit from the
   [NVIDIA CUDA downloads page](https://developer.nvidia.com/cuda-downloads).
3. Verify:
   ```bash
   nvcc --version
   ```

Note the CUDA version (e.g. 11.x or 12.x) for the next step.

### 1.5 Install CuPy (GPU only)

Install the CuPy build matching your CUDA version:

```bash
pip install cupy-cuda12x   # for CUDA 12.x
pip install cupy-cuda11x   # for CUDA 11.x
```

Verify GPU detection:

```bash
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount(), 'GPU(s) detected')"
```

### 1.6 (Optional) Install mpi4py for Parallel Runs

Only needed when `is_parallel = True` in `para.py`:

```bash
conda install -c conda-forge openmpi -y   # an MPI implementation
pip install mpi4py                         # Python bindings
```

Parallel runs are launched with `mpirun` (see §2.7).

---

## 2. Running a Simulation

All behaviour is controlled by `para.py` in your chosen case folder
(`compressible/2d/` or `compressible/3d/`). Edit it, save, then run from that
folder:

```bash
cd compressible/2d      # or compressible/3d
python main.py
```

`main.py` reads every variable in `para.py`, builds the problem, and advances
it in time. You normally never edit `main.py`.

### 2.1 Simulation Details

| Parameter | Options | Description |
|-----------|---------|-------------|
| `device` | `'CPU'`, `'GPU'` | Compute backend (NumPy vs CuPy). |
| `is_parallel` | `True`, `False` | Enable MPI domain decomposition (needs `mpi4py`). |
| `device_rank` | `0`, `1`, … | GPU index for a serial GPU run. |
| `output_dir` | `'output/test'` | Output path, relative to the case folder. |
| `dimension` | `2`, `3` | Spatial dimension; must match the case folder. |
| `problem` | `'hydro'`, `'euler'`, … | Governing equations. Compressible NS uses `'hydro'`. |
| `sub_problem` | `'TG_vortex'`, `'sod'`, … | Specific test case (valid options listed in `para.py`). |
| `layout_type` | `'collocated'`, `'nodal'`, `'staggered'` | Variable arrangement (default `collocated`). |
| `spacing_type` | `'uniform'`, `'non-uniform'` | Uniform or clustered (tangent-hyperbolic) grid. |
| `beta` | `1`, `1.2`, … | Clustering strength (non-uniform grids only). |
| `profile` | `'static'`, `'custom'`, `'continue'` | Initial-condition source. |
| `reconstruction` | `linear`, `cweno3`, `weno3/5/7`, `teno5/7` | Spatial reconstruction scheme. |
| `z_smoother` | `True`, `False` | Enable the Z-smoother in reconstruction. |
| `viscous_term_order` | `2`, `4`, `6` | Order of accuracy for viscous terms. |
| `scheme` | `'RK2'`, `'RK3'` | Runge–Kutta time-integration order. |
| `imex` | `True`, `False` | Use the IMEX time-integration scheme. |
| `data_type` | `'float64'`, `'float32'` | Array precision (`float32` is faster/lighter on GPU). |

### 2.2 Time-Stepping

| Parameter | Example | Description |
|-----------|---------|-------------|
| `tinit` | `0` | Start time (restart time for `profile = 'continue'`). |
| `tfinal` | `1` | Stop time. |
| `dt` | `2.5e-3` | Fixed step size (used when `cfl_condition = False`). |
| `cfl_condition` | `True`, `False` | If `True`, step size is set adaptively and `dt` is ignored. |
| `cfl_cons` | `0.5` | CFL constant for adaptive stepping. |

### 2.3 Grid / Domain

Lists have one entry per dimension.

| Parameter | 2D | 3D | Description |
|-----------|----|----|-------------|
| `L_min` | `[-np.pi, -np.pi]` | `[-np.pi, -np.pi, -np.pi]` | Lower domain bounds. |
| `L_max` | `[np.pi, np.pi]` | `[np.pi, np.pi, np.pi]` | Upper domain bounds. |
| `N` | `[256, 256]` | `[64, 64, 64]` | Grid points per direction (resolution). |

`np` (NumPy) is imported at the top of `para.py`, so `np.pi` is valid.

### 2.4 Control (Physical) Parameters — hydro

| Parameter | Example | Description |
|-----------|---------|-------------|
| `gamma` | `1.4` | Ratio of specific heats. |
| `Ma` | `1.25` | Mach number. |
| `mu_varying` | `False` | Temperature-dependent viscosity (Sutherland's law) if `True`. |
| `Re` | `1600` | Reynolds number. |
| `forcing` | `False` | Enable a turbulence forcing term. |
| `forced_vrms` | `1` | Target RMS velocity for forcing. |
| `zeta` | `2/3` | Compressibility parameter for forcing. |
| `rad_cooling` | `False` | Enable a radiative-cooling source term. |
| `alpha_cooling` | `1e-2` | Cooling-rate parameter. |

### 2.5 Output / Saving

| Parameter | Example | Description |
|-----------|---------|-------------|
| `print_out_terminal` | `True` | Print running time-instance to the terminal. |
| `save_fields` | `True` | Save field snapshots (`.h5`) to `output_dir/fields/`. |
| `save_fields_dt` | `2.5` | Interval between saved field files. |
| `save_global_quantities` | `True` | Save integrated quantities over time. |
| `save_global_quantities_dt` | `1e-2` | Sampling interval (keep `≥ dt`). |
| `save_global_file` | `1` | Interval for writing the global-quantities file. |
| `save_plots` | `False` | Save field plots during the run. |
| `save_plots_tinit` | `0` | Time at which plot-saving begins. |
| `save_plots_dt` | `5e-1` | Interval between saved plots. |
| `N_dump_fields` | `10` / `50` | Snapshots grouped per plotting file (3D `para.py`). |

### 2.6 CPU vs GPU (Quick Reference)

- **CPU:** `device = 'CPU'` — only core libraries (§1.3) needed.
- **Serial GPU:** `device = 'GPU'`, `is_parallel = False`, pick GPU with `device_rank`.
- **Speed vs precision:** on GPU, `data_type = 'float32'` roughly halves memory and speeds up runs at reduced accuracy.

### 2.7 Running in Parallel (optional)

With `is_parallel = True`, launch with MPI:

```bash
mpirun -np 4 python main.py
```

Requires `mpi4py` (§1.6). In parallel GPU runs, each rank is assigned a GPU by
`rank % (GPUs per node)`.

---

## 3. Example Run — 512 × 512 Taylor–Green Vortex on GPU

This section runs the 2D compressible Taylor–Green vortex at 512 × 512 on a
single GPU (device rank 0), then examines the output it produces.

---

### Part 1 — Running the Simulation

**Step 1** — Open `compressible/2d/para.py`.

**Step 2** — Set these lines (leave the rest at default):

```python
# --- Simulation details ---
device      = 'GPU'          # run on the GPU
is_parallel = False          # single-device (serial) run
device_rank = 0              # use GPU rank 0

dimension   = 2
problem     = 'hydro'        # compressible Navier–Stokes
sub_problem = 'TG_vortex'    # Taylor–Green vortex
profile     = 'static'       # built-in analytic initial condition

reconstruction = 'teno5'     # 5th-order TENO reconstruction
scheme         = 'RK3'       # 3rd-order Runge–Kutta
data_type      = 'float64'   # use 'float32' for a faster, lighter run

# --- Grid / domain ---
L_min = [-np.pi, -np.pi]     # domain: [-π, π] × [-π, π]
L_max = [ np.pi,  np.pi]
N     = [512, 512]           # 512 × 512 resolution

# --- Time stepping ---
tinit  = 0
tfinal = 10                  # adjust to taste
dt     = 2.5e-3
cfl_condition = False        # fixed step; True lets CFL choose dt

# --- Output ---
save_fields    = True
save_fields_dt = 2.5         # write a field snapshot every 2.5 time units

# --- Physics ---
gamma = 1.4
Ma    = 1.25
Re    = 1600
```

**Step 3** — Run:

```bash
cd compressible/2d
python main.py
```

`main.py` builds a `Hydro2D` problem, sets the analytic Taylor–Green field, and
advances it with RK3 until `tfinal`. The total wall-clock time prints at the
end.

**Tips:** for a quicker first test use `N = [256, 256]` or a smaller `tfinal`;
`data_type = 'float32'` halves GPU memory; change `device_rank` to pick a
different GPU.

---

### Part 2 — Results

#### 3.1 How the Output Is Saved

With the settings above, the run writes two kinds of HDF5 output into
`output_dir` (here `output/test/`).

**Field snapshots — `fields/2D_<time>.h5`.** Whenever the simulation time
reaches a multiple of `save_fields_dt` (2.5), `DataIO` calls the problem's
`save_fields`, which writes one file per snapshot. With `tfinal = 10` this
produces five files:

```
output/test/fields/2D_0.00.h5     # initial condition (t = 0)
output/test/fields/2D_2.50.h5
output/test/fields/2D_5.00.h5
output/test/fields/2D_7.50.h5
output/test/fields/2D_10.00.h5    # final state (t = 10)
```

The number in each filename is the snapshot time. Every file contains the full
spatial fields at that instant:

| Dataset | Meaning |
|---------|---------|
| `rho` | density, 2D array of shape `(Nx, Ny)` |
| `p` | pressure |
| `ux`, `uy` | velocity components |
| `x`, `y` | 1D coordinate arrays |
| `t` | the snapshot time (scalar) |

**Global quantities — `glob_0.00.h5`.** In parallel, `DataIO` records
domain-integrated scalar diagnostics over the whole run and writes them to a
single file. The `0.00` in the name is `tinit` (the run's start time), not a
timestep. Each dataset is a 1D time series sampled every
`save_global_quantities_dt`, with the sample times stored in `t`:

| Dataset | Meaning |
|---------|---------|
| `M_T` | total mass |
| `I_T` | total internal energy |
| `E_T` | total energy |
| `K_T` | total kinetic energy |
| `V_rms` | root-mean-square velocity |
| `Re` | Reynolds number |
| `Ma_max` | maximum Mach number |
| `Ma_avg` | volume-averaged Mach number |
| `t` | sample times |

The field files are used for spatial plots (a snapshot of the flow); the global
file is used for time-history plots (how integrated quantities evolve). The
figures below were produced from these two sources.

> The image paths in this section assume the PNG files sit in the same folder
> as this document; adjust the paths if you store them elsewhere.

#### 3.2 Final Velocity Field

![Final velocity field at t = 10](https://github.com/user-attachments/assets/ab4f45ad-4860-4efb-8b3b-663b4a089359
)

*Velocity magnitude $|u| = \sqrt{u_x^2 + u_y^2}$ at $t = 10$, produced from
`2D_10.00.h5`, with the velocity vectors overlaid in white.*

By $t = 10$ the smooth initial Taylor–Green pattern has evolved into a finer
cellular structure while preserving the fourfold symmetry of the initial
condition and boundary conditions. Bright (yellow) bands are high-speed shear
regions; dark (purple) spots mark vortex cores and stagnation points, where the
speed drops toward zero. The white arrows trace the circulation, curling around
each vortex core — the signature of the decaying vortex array.

#### 3.3 Energy Budget vs Time

![Kinetic, internal and total energy vs time](https://github.com/user-attachments/assets/eb38bdce-5552-4117-bbf2-1d1502f653b1)

*Total kinetic (`K_T`), internal (`I_T`) and total (`E_T`) energy vs time, from
`glob_0.00.h5`.*

The total energy `E_T` (green) stays essentially flat, confirming that the
conservative scheme conserves total energy over the run. Kinetic energy `K_T`
(blue) falls from about 10, and internal energy `I_T` (orange) rises by a
matching amount — viscous dissipation steadily converts kinetic energy into
heat. The wavy modulation on the two curves is the compressible (acoustic)
exchange between kinetic and internal energy through pressure work; the net
trend remains a monotonic decay of kinetic energy.

#### 3.4 Kinetic Energy Decay

![Total kinetic energy vs time](https://github.com/user-attachments/assets/04008c1a-3ab7-49a7-9552-4e3c22aee554
)

*Total kinetic energy `K_T` vs time (a zoom of the blue curve above).*

Starting near 9.85, the kinetic energy decays to roughly 5.5–6 by $t = 10$. The
oscillations riding on the decay envelope — local peaks near $t \approx 2.5,\
5.8,\ 9$ and troughs near $t \approx 4.5,\ 7.7$ — are acoustic oscillations
inherent to the compressible dynamics: energy sloshes back and forth between
kinetic and internal forms via pressure work while viscosity drains the overall
kinetic energy.

#### 3.5 Mass Conservation

![Total mass vs time](https://github.com/user-attachments/assets/392e49ff-830b-4409-b92e-69a5b778b415
)

*Total mass `M_T` vs time, from `glob_0.00.h5`.*

Note the axis scale: values are offset by `3.94784176e1` with ticks in units of
`1e-11`. The total mass therefore stays at $\approx 39.4784176$ and varies by
only about $10^{-11}$ across the entire run — conservation to roughly twelve
significant figures. The faint downward slope is at the level of
floating-point round-off accumulation, not a physical loss, which confirms that
the finite-volume formulation conserves mass to machine precision.
---

## 4. Theory

### 4.1 Governing Equations

The compressible module (`problem = 'hydro'`) solves the **compressible
Navier–Stokes equations** in non-dimensional form. With state vector
$\mathbf{q} = (\rho,\ \rho u,\ \rho v,\ E)$:

$$
\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{u}) = 0
$$

$$
\frac{\partial (\rho \mathbf{u})}{\partial t}
  + \nabla \cdot (\rho \mathbf{u}\otimes\mathbf{u} + p\,\mathbf{I})
  = \nabla \cdot \boldsymbol{\tau} + \mathbf{f}
$$

$$
\frac{\partial E}{\partial t} + \nabla \cdot \big[(E + p)\mathbf{u}\big]
  = \nabla \cdot (\boldsymbol{\tau}\cdot\mathbf{u}) + \nabla\cdot(K\nabla T)
$$

Closure relations (as implemented):

- **Equation of state:** $p = (\gamma - 1)\left(E - \tfrac{1}{2}\rho|\mathbf{u}|^2\right)$, and $p = R_{\text{gas}}\,\rho\,T$ with $R_{\text{gas}} = \dfrac{1}{\gamma\,\mathrm{Ma}^2}$.
- **Viscous stress (Newtonian):** $\tau_{xx} = \mu\!\left(\tfrac{4}{3}\partial_x u - \tfrac{2}{3}\partial_y v\right)$, $\tau_{xy} = \mu\!\left(\partial_y u + \partial_x v\right)$.
- **Transport coefficients:** $\mu = \dfrac{1}{\mathrm{Re}}$, $K = \dfrac{1}{\mathrm{Ma}^2(\gamma-1)\,\mathrm{Re}\,\mathrm{Pr}}$.
- $\mathbf{f}$ is an optional forcing term (`forcing = True`).

Dropping the viscous terms recovers the **Euler equations**
(`problem = 'euler'`), on which the hydro solver is built.

### 4.2 Equations Used in the Example Simulation (in order)

For the 512 × 512 Taylor–Green vortex (`sub_problem = 'TG_vortex'`,
`profile = 'static'`):

**(1) Gas constant:**
$$R_{\text{gas}} = \frac{1}{\gamma\,\mathrm{Ma}^2}$$

**(2) Analytic initial fields:**
$$u = \sin x \,\cos y, \qquad v = -\cos x \,\sin y$$
$$p = R_{\text{gas}} + \tfrac{1}{16}\big(\cos 2x + \cos 2y\big), \qquad \rho = \frac{p}{R_{\text{gas}}}\ (T_0 = 1)$$

**(3) Conserved variables:**
$$\rho u = \rho\,u,\qquad \rho v = \rho\,v,\qquad E = \frac{p}{\gamma - 1} + \tfrac{1}{2}\rho\,(u^2 + v^2)$$

**(4) Inviscid flux vectors:**
$$\mathbf{F}_x = \big(\rho u,\ \rho u^2 + p,\ \rho u v,\ u(E+p)\big),\quad
\mathbf{F}_y = \big(\rho v,\ \rho u v,\ \rho v^2 + p,\ v(E+p)\big)$$

**(5) KT central numerical flux** from reconstructed states $\mathbf{q}^{\pm}$ and wave speeds $a$:
$$\mathbf{H} = \tfrac{1}{2}\big(\mathbf{F}(\mathbf{q}^+) + \mathbf{F}(\mathbf{q}^-)\big) - \tfrac{1}{2}\,a\,(\mathbf{q}^+ - \mathbf{q}^-), \qquad a = \sqrt{\gamma p/\rho} + |u_n|$$

**(6) Viscous stresses** added to momentum/energy:
$$\tau_{xx} = \mu\!\left(\tfrac{4}{3}\partial_x u - \tfrac{2}{3}\partial_y v\right),\quad
\tau_{xy} = \mu\!\left(\partial_y u + \partial_x v\right),\quad \mu = \frac{1}{\mathrm{Re}}$$

**(7) Time advance** (SSP-RK3): $\mathbf{q}$ is updated stage-by-stage from the
combined convective + diffusive residual until $t = t_{\text{final}}$.

---

## 5. The DHARA Package — Compressible-Relevant Files

`DHARA/` is the core library imported by every case (`from DHARA import *`).
Only the files on the compressible path are described below.

### 5.1 Top Level

- **`__init__.py`** — package entry point; exposes the public API, including the
  `Hydro2D`/`Hydro3D` and `Euler2D`/`Euler3D` classes used here. Makes
  `from DHARA import *` work.
- **`exceptions.py`** — `check_run_permission()` prevents overwriting an
  existing `output_dir` unless `profile = 'continue'`, and aborts a continue-run
  if no prior output exists.

### 5.2 `init_cond/` — Initial Conditions

- **`init2d.py`, `init3d.py`** — build non-static initial fields, used by
  `main.py` when `profile` is `'custom'` or `'continue'`.

### 5.3 `src/univ/` — Universal Machinery

- **`problem_setup.py`** — `ProblemSetup` base class: allocates conserved-variable
  arrays (`q`, `q_copy`), real-vs-ghost slicing, CFL/time-step state, and
  device/parallel flags. Every problem inherits from it.
- **`evolution.py`** — `Time_Evolution`: the time-marching loops
  (`time_advance_rk2`, `time_advance_rk3`) that call the problem's single-step
  solver and trigger periodic saving.
- **`data_io.py`** — `DataIO`: creates the output directory, saves field `.h5`
  files and global-quantity time series, and loads the per-problem `glob.py`
  and `plot.py` helpers.
- **`grids/grid2d.py`, `grids/grid3d.py`** — grid classes: select the backend
  (NumPy/CuPy from `device`, GPU/MPI setup), build uniform or tangent-hyperbolic
  grids, allocate scratch arrays, and fill ghost cells / boundaries.

### 5.4 `src/problem/` — Compressible Solvers

- **`euler/euler2d.py`, `euler/euler3d.py`** — inviscid compressible Euler
  solvers: conserved↔primitive conversions, pressure, Euler flux vectors, wave
  speeds, CFL step, and boundary conditions per sub-problem.
- **`hydro/hydro2d.py`, `hydro/hydro3d.py`** — the compressible Navier–Stokes
  solvers used here. Inherit from the matching `Euler` and `VisFlux` classes and
  add viscosity/conductivity, temperature, diffusion terms, optional forcing,
  and the single-step drivers.
- **`hydro/compute2d/glob.py`, `hydro/compute3d/glob.py`** — compute
  global/integrated diagnostics, loaded on demand by `DataIO`.
- **`hydro/plot2d/plot.py`, `hydro/plot3d/plot.py`** — generate plots, loaded on
  demand by `DataIO`.

### 5.5 `src/lib/` — Numerical Methods (compressible path)

**Convective flux — `convflux/`**
- **`KTSD.py`** — Kurganov–Tadmor semi-discrete central flux (the scheme used by
  the hydro/euler solvers).

**Reconstruction — `reconstruction/`**
- **`weno.py`** — WENO/TENO smoothness indicators and polynomial
  reconstructions (orders 3/5/7, Z-smoother variants).
- **`reconstruction.py`** — dispatcher selecting the scheme from `para.py`, plus
  slope limiters (van Albada, minmod) and linear reconstruction.

**Viscous flux — `visflux/`**
- **`visflux_2d.py`, `visflux_3d.py`** — viscous stress and heat-flux terms for
  the 2D/3D compressible Navier–Stokes equations.
- **`dervFV.py`, `dervFV4.py`, `dervFV6.py`** — finite-volume derivative
  operators at 2nd/4th/6th order for the viscous terms.
