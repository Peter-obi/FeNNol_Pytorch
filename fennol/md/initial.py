import sys
import os
import io
import argparse
import time
from pathlib import Path
import math

import numpy as np
import torch
from typing import Optional, Callable
from collections import defaultdict
from functools import partial

from ..models import FENNIX
from ..utils.io import last_xyz_frame
from ..utils.periodic_table import PERIODIC_TABLE_REV_IDX, ATOMIC_MASSES
from ..utils.atomic_units import AtomicUnits as au
from ..utils.input_parser import parse_input
from .thermostats import get_thermostat

from copy import deepcopy

def load_model(simulation_parameters):
    """
    Load a FENNIX model from a file specified in the simulation parameters.

    Args:
        simulation_parameters (Dict[str, Any]): A dictionary containing simulation parameters,
            including the path to the model file and optional graph configuration.

    Returns:
        FENNIX: The loaded FENNIX model.
    """
    model_file = simulation_parameters.get("model_file")
    model_file = Path(str(model_file).strip())
    if not model_file.exists():
        raise FileNotFoundError(f"model file {model_file} not found")
    else:
        graph_config = simulation_parameters.get("graph_config", {})
        model = FENNIX.load(model_file, graph_config=graph_config)
        print(f"# model_file: {model_file}")

    if "energy_terms" in simulation_parameters:
        model.energy_terms = simulation_parameters["energy_terms"]
        print("# energy terms:", model.energy_terms)
    return model

def load_system_data(simulation_parameters, fprec):
    """
    Load system data from an XYZ file and prepare it for simulation.

    Args:
        simulation_parameters (Dict[str, Any]): A dictionary containing simulation parameters,
            including the path to the XYZ file and other system-specific settings.
        fprec (torch.dtype): The floating-point precision to use for tensor operations.

    Returns:
        Tuple[Dict[str, Any], Dict[str, torch.Tensor]]: A tuple containing two dictionaries:
            1. system_data: Contains general system information (e.g., atom count, temperature).
            2. conformation: Contains structural information (e.g., coordinates, species).
    """
    system_name = str(simulation_parameters.get("system", "system")).strip()
    indexed = simulation_parameters.get("xyz_input/indexed", True)
    box_info = simulation_parameters.get("xyz_input/box_info", False)
    box_info = simulation_parameters.get("xyz_input/has_comment_line", box_info)
    xyzfile = Path(simulation_parameters.get("xyz_input/file", system_name + ".xyz"))
    if not xyzfile.exists():
        raise FileNotFoundError(f"xyz file {xyzfile} not found")
    system_name = str(simulation_parameters.get("system", xyzfile.stem)).strip()
    symbols, coordinates, _ = last_xyz_frame(
        xyzfile, indexed=indexed, box_info=box_info
    )
    if not symbols:  
        raise IndexError("No atoms found in the XYZ file")
    coordinates = torch.tensor(coordinates, dtype=fprec)
    species = torch.tensor([PERIODIC_TABLE_REV_IDX[s] for s in symbols], dtype=torch.long)
    nat = species.shape[0]

    mass_amu = torch.tensor(ATOMIC_MASSES, dtype=fprec)[species]
    deuterate = simulation_parameters.get("deuterate", False)
    if deuterate:
        print("# Replacing all hydrogens with deuteriums")
        mass_amu[species == 1] *= 2.0
    mass = mass_amu * (au.MPROT * (au.FS / au.BOHR) ** 2)

    temperature = np.clip(simulation_parameters.get("temperature", 300.0), 1.0e-6, None)
    kT = temperature / au.KELVIN

    system_data = {
        "name": system_name,
        "nat": nat,
        "symbols": symbols,
        "species": species,
        "mass": mass,
        "temperature": temperature,
        "kT": kT,
    }

    cell = simulation_parameters.get("cell", None)
    if cell is not None:
        cell = torch.tensor(cell, dtype=fprec).reshape(3, 3)
        reciprocal_cell = torch.inverse(cell)
        volume = torch.det(cell)
        print("# cell matrix:")
        for l in cell:
            print("# ", l)
        totmass_amu = mass_amu.sum()
        dens = totmass_amu / 6.02214129e-1 / volume
        print("# density: ", dens.item(), " g/cm^3")
        pscale = au.KBAR / (3.0 * volume / au.BOHR**3)
        minimum_image = simulation_parameters.get("minimum_image", True)
        estimate_pressure = simulation_parameters.get("estimate_pressure", False)

        crystal_input = simulation_parameters.get("xyz_input/crystal", False)
        if crystal_input:
            coordinates = torch.matmul(coordinates, cell)

        pbc_data = {
            "cell": cell,
            "reciprocal_cell": reciprocal_cell,
            "volume": volume,
            "pscale": pscale,
            "minimum_image": minimum_image,
            "estimate_pressure": estimate_pressure,
        }
    else:
        pbc_data = None
    system_data["pbc"] = pbc_data

    nbeads = simulation_parameters.get("nbeads", None)
    if nbeads is not None:
        nbeads = int(nbeads)
        print("# nbeads: ", nbeads)
        system_data["nbeads"] = nbeads
        coordinates = coordinates.repeat(nbeads, 1, 1).reshape(-1, 3)
        species = species.repeat(nbeads, 1).reshape(-1)
        bead_index = torch.arange(nbeads, dtype=torch.long).repeat_interleave(nat)
        natoms = torch.full((nbeads,), nat, dtype=torch.long)

        eigmat = torch.zeros((nbeads, nbeads))
        for i in range(nbeads - 1):
            eigmat[i, i] = 2.0
            eigmat[i, i + 1] = -1.0
            eigmat[i + 1, i] = -1.0
        eigmat[-1, -1] = 2.0
        eigmat[0, -1] = -1.0
        eigmat[-1, 0] = -1.0
        omk, eigmat = torch.linalg.eigh(eigmat)
        omk[0] = 0.0
        omk = nbeads * kT * torch.sqrt(omk) / au.FS
        for i in range(nbeads):
            if eigmat[i, 0] < 0:
                eigmat[i] *= -1.0
        eigmat = eigmat.to(dtype=fprec)
        system_data["omk"] = omk
        system_data["eigmat"] = eigmat
    else:
        bead_index = torch.full((nat,), 0, dtype=torch.long)
        natoms = torch.tensor([nat], dtype=torch.int32)

    conformation = {
        "species": species,
        "coordinates": coordinates,
        "batch_index": bead_index,
        "natoms": natoms,
    }
    if cell is not None:
        conformation["cells"] = cell.unsqueeze(0)
        conformation["reciprocal_cells"] = reciprocal_cell.unsqueeze(0)

    return system_data, conformation

def initialize_preprocessing(simulation_parameters, model, conformation, system_data):
    """
    Initialize preprocessing for the simulation based on the model and system configuration.

    Args:
        simulation_parameters (Dict[str, Any]): A dictionary containing simulation parameters.
        model (FENNIX): The FENNIX model to be used for simulation.
        conformation (Dict[str, torch.Tensor]): The initial system conformation.
        system_data (Dict[str, Any]): System-specific data.

    Returns:
        Tuple[Dict[str, Any], Dict[str, torch.Tensor]]: A tuple containing:
            1. preproc_state: The updated preprocessing state.
            2. conformation: The potentially modified system conformation.
    """
    nblist_verbose = simulation_parameters.get("nblist_verbose", False)
    nblist_skin = simulation_parameters.get("nblist_skin", -1.0)
    pbc_data = system_data.get("pbc", None)

    preproc_state = deepcopy(model.preproc_state)
    
    if "layers_state" not in preproc_state:
        preproc_state["layers_state"] = [{}]
    
    layers_state = preproc_state["layers_state"]
    for st in layers_state:
        if pbc_data is not None:
            st["minimum_image"] = pbc_data["minimum_image"]
        if nblist_skin >= 0:
            st["nblist_skin"] = nblist_skin
        if "nblist_mult_size" in simulation_parameters:
            st["nblist_mult_size"] = simulation_parameters["nblist_mult_size"]
        if "nblist_add_neigh" in simulation_parameters:
            st["add_neigh"] = simulation_parameters["nblist_add_neigh"]

    preproc_state["check_input"] = True
    preproc_state, conformation = model.preprocessing(preproc_state, conformation)

    # Ensure layers_state is preserved after preprocessing
    preproc_state["layers_state"] = layers_state

    preproc_state["check_input"] = False

    if nblist_verbose:
        graphs_keys = list(model._graphs_properties.keys())
        print("# graphs_keys: ", graphs_keys)
        print("# nblist state:", preproc_state)

    if simulation_parameters.get("print_model", False):
        print(model.summarize(example_data=conformation))

    return preproc_state, conformation

def initialize_system(conformation, vel, model, system_data, fprec):
    """
    Initialize the system for simulation by computing initial energies and forces.

    Args:
        conformation (Dict[str, torch.Tensor]): The initial system conformation.
        vel (torch.Tensor): Initial velocities of the particles.
        model (FENNIX): The FENNIX model to be used for force and energy calculations.
        system_data (Dict[str, Any]): System-specific data.
        fprec (torch.dtype): The floating-point precision to use for tensor operations.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing initialized system properties including:
            - ek: Kinetic energy
            - epot: Potential energy
            - vel: Velocities
            - coordinates: Particle positions
            - forces: Forces acting on particles
    """
    print("# Computing initial energy and forces")
    e, f, _ = model._energy_and_forces(model.variables, conformation)
    f = f.cpu().numpy()
    epot = torch.mean(e)
    ek = 0.5 * torch.sum(system_data["mass"].unsqueeze(1) * vel**2)

    system = {}
    system["ek"] = ek
    system["epot"] = epot
    system["vel"] = vel.to(dtype=fprec)
    if "nbeads" in system_data:
        nbeads = system_data["nbeads"]
        coordinates = conformation["coordinates"].reshape(nbeads, -1, 3)
        eigx = torch.zeros_like(coordinates)
        eigx[0] = coordinates[0]
        system["coordinates"] = eigx
        system["forces"] = torch.einsum(
            "in,i...->n...", system_data["eigmat"], torch.tensor(f.reshape(nbeads, -1, 3))
        ) * (1.0 / nbeads**0.5)
    else:
        system["coordinates"] = conformation["coordinates"]
        system["forces"] = torch.tensor(f)

    return system
