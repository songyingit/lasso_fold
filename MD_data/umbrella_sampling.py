"""
Biased Simulation, Umbrella Sampling, for Lasso Peptide

"""

import sys
import os
import numpy as np
import mdtraj as md
from itertools import combinations

# OpenMM imports
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from simtk.openmm import CustomCVForce, RMSDForce
from parmed.amber import LoadParm
from parmed.openmm import RestartReporter
from mdtraj.reporters import DCDReporter


def calculate_native_contacts(traj, native):
    """
    Calculate fraction of native contacts (Q) for lasso peptide structure
    
    Parameters:
    -----------
    traj : mdtraj.Trajectory
        Current trajectory frame
    native : mdtraj.Trajectory
        Reference native structure
        
    Returns:
    --------
    q : float
        Fraction of native contacts
    """
    BETA_CONST = 50  # 1/nm - steepness of sigmoidal function
    LAMBDA_CONST = 1.8  # scaling factor for native distance
    NATIVE_CUTOFF = 0.45  # nanometers - cutoff for native contact definition
    
    # Select heavy atoms only
    heavy = native.topology.select_atom_indices('heavy')
    
    # Define native contacts: heavy atom pairs >3 residues apart
    heavy_pairs = np.array(
        [(i, j) for (i, j) in combinations(heavy, 2)
         if abs(native.topology.atom(i).residue.index - 
                native.topology.atom(j).residue.index) > 3])
    
    # Compute distances in native state
    heavy_pairs_distances = md.compute_distances(native[0], heavy_pairs)[0]
    
    # Filter pairs within native cutoff
    native_contacts = heavy_pairs[heavy_pairs_distances < NATIVE_CUTOFF]
    
    # Compute distances for current trajectory
    r = md.compute_distances(traj, native_contacts)
    r0 = md.compute_distances(native[0], native_contacts)
    
    # Calculate Q using continuous sigmoidal function
    q = np.mean(1.0 / (1 + np.exp(BETA_CONST * (r - LAMBDA_CONST * r0))), axis=1)
    
    return q


# =============================================================================
# Simulation Parameters
# =============================================================================

# Platform settings
platform = Platform.getPlatformByName('CUDA')
platformProperties = {'CudaPrecision': 'single'}

# Force field parameters (CHARMM36m)
nonbondedMethod = PME
nonbondedCutoff = 1.0 * nanometers
ewaldErrorTolerance = 0.0005
constraints = HBonds
rigidWater = True
constraintTolerance = 0.000001

# Thermodynamic conditions
temperature = 300 * kelvin
pressure = 1.0 * atmospheres
friction = 2.8284 / picosecond

# Umbrella sampling parameters
WINDOW_NUMBER = 0  # Current umbrella window
REPLICA_NUMBER = 0  # Replica index for this window
FORCE_CONSTANT = 20 * kilocalories_per_mole / angstroms**2  # Harmonic restraint

# Simulation time parameters
dt = 0.004 * picoseconds
n_steps = 5000000
report_interval = 25000  
barostat_interval = 25


# =============================================================================
# File Paths
# =============================================================================

BASE_DIR = ''
REF_DIR = f'{BASE_DIR}/ref'
PRMTOP_DIR = f'{BASE_DIR}/prmtop'
NCRST_DIR = f'{BASE_DIR}/ncrst'
OUTPUT_DIR = f'{BASE_DIR}/dcd'

# Reference structure (pre-folded lasso peptide)
native_structure = f'{REF_DIR}/lasso-nobond-min.ncrst'
native_topology = f'{REF_DIR}/lasso-nobond-HMR.prmtop'

# Starting structure for this window
input_prmtop = f'{PRMTOP_DIR}/lasso_US_window_{WINDOW_NUMBER}_rep_{REPLICA_NUMBER}.prmtop'
input_ncrst = f'{NCRST_DIR}/lasso_US_window_{WINDOW_NUMBER}_rep_{REPLICA_NUMBER}.ncrst'

# Output files
output_dcd = f'{OUTPUT_DIR}/lasso_US_window_{WINDOW_NUMBER}_rep_{REPLICA_NUMBER}.dcd'
output_log = f'{OUTPUT_DIR}/lasso_US_window_{WINDOW_NUMBER}_state_reporter_rep_{REPLICA_NUMBER}.log'
output_chk = f'{OUTPUT_DIR}/lasso_US_window_{WINDOW_NUMBER}_state_checker_rep_{REPLICA_NUMBER}.chk'
output_restart = f'{OUTPUT_DIR}/lasso_US_window_{WINDOW_NUMBER}_restart_rep_{REPLICA_NUMBER}.ncrst'


# =============================================================================
# Load Structures
# =============================================================================

print(f"Loading reference structure: {native_structure}")
native = md.load(native_structure, top=native_topology)

print(f"Loading starting structure for window {WINDOW_NUMBER}, replica {REPLICA_NUMBER}")
parm = LoadParm(input_prmtop, input_ncrst)
topology = parm.topology
positions = parm.positions


# =============================================================================
# Build OpenMM System
# =============================================================================

print("Creating OpenMM system...")
system = parm.createSystem(
    nonbondedMethod=nonbondedMethod,
    nonbondedCutoff=nonbondedCutoff,
    constraints=constraints,
    rigidWater=rigidWater,
    ewaldErrorTolerance=ewaldErrorTolerance
)

# Add Monte Carlo barostat for NPT ensemble
system.addForce(MonteCarloBarostat(pressure, temperature, barostat_interval))

# Add RMSD-based umbrella sampling restraint
print(f"Adding umbrella sampling restraint (k = {FORCE_CONSTANT})...")
heavy_atoms = native.topology.select_atom_indices('heavy')
rmsd_force = RMSDForce(positions, heavy_atoms)
cv_force = CustomCVForce("k*(RMSD^2)")  # Harmonic potential: U = k*(RMSD-RMSD0)^2
cv_force.addCollectiveVariable('RMSD', rmsd_force)
cv_force.addGlobalParameter("k", FORCE_CONSTANT)
system.addForce(cv_force)


# =============================================================================
# Setup Simulation
# =============================================================================

print("Setting up simulation...")
integrator = LangevinIntegrator(temperature, friction, dt)
integrator.setConstraintTolerance(constraintTolerance)

simulation = Simulation(topology, system, integrator, platform, platformProperties)
simulation.context.setPositions(positions)

if parm.box_vectors is not None:
    simulation.context.setPeriodicBoxVectors(*parm.box_vectors)

simulation.context.setVelocitiesToTemperature(temperature)


# =============================================================================
# Add Reporters
# =============================================================================

print("Adding reporters...")
simulation.reporters.append(DCDReporter(output_dcd, report_interval))
simulation.reporters.append(StateDataReporter(
    output_log,
    report_interval,
    totalSteps=n_steps,
    step=True,
    speed=True,
    progress=True,
    elapsedTime=True,
    remainingTime=True,
    potentialEnergy=True,
    temperature=True,
    volume=True,
    density=True,
    separator='\t'
))
simulation.reporters.append(CheckpointReporter(output_chk, report_interval))
simulation.reporters.append(RestartReporter(
    output_restart,
    reportInterval=report_interval,
    netcdf=True
))


# =============================================================================
# Run Umbrella Sampling Simulation
# =============================================================================

print(f"Window: {WINDOW_NUMBER}, Replica: {REPLICA_NUMBER}")
print(f"Total steps: {n_steps} ({n_steps * dt.value_in_unit(nanoseconds):.2f} ns)")
print(f"Reporting interval: {report_interval} steps\n")

atom_indices = native.topology.select("backbone")

for frame in range(int(n_steps / report_interval)):
    simulation.step(report_interval)
    
    # Get current state
    state = simulation.context.getState(getPositions=True, getEnergy=True)
    current_positions = state.getPositions(asNumpy=True)
    
    # Create mdtraj trajectory for analysis
    current_positions_array = np.array(
        current_positions.value_in_unit(current_positions.unit),
        dtype=np.float32
    )
    current_traj = md.Trajectory(current_positions_array, parm.topology)
    
    # Align to native structure
    current_traj_aligned = current_traj.superpose(native, atom_indices=atom_indices)
    
    # Calculate RMSD of heavy atoms
    current_heavy = current_traj_aligned.xyz[0, heavy_atoms, :]
    reference_heavy = native.xyz[0, heavy_atoms, :]
    rmsd = np.sqrt(np.mean(np.square(current_heavy - reference_heavy)))
    
    # Calculate fraction of native contacts
    q = calculate_native_contacts(current_traj, native)
    
    print(f"Frame {frame}: RMSD = {rmsd:.4f} nm, Q = {q[0]:.4f}")
