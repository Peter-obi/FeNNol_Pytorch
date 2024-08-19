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

from ..utils.io import (
    write_arc_frame,
    last_xyz_frame,
    write_xyz_frame,
    write_extxyz_frame,
    human_time_duration,
)
from ..utils.periodic_table import PERIODIC_TABLE_REV_IDX, ATOMIC_MASSES
from ..utils.atomic_units import AtomicUnits as au
from ..utils.input_parser import parse_input
from .initial import load_model, load_system_data, initialize_preprocessing
from .integrate import initialize_dynamics

from copy import deepcopy

def minmaxone(x, name=""):
    """
    Calculate and print the minimum, maximum, and root mean square of a tensor.

    Args:
        x (torch.Tensor): The input tensor.
        name (str, optional): A name to identify the tensor in the output. Defaults to "".

    Prints the min, max, and RMS values of the tensor.
    """
    if x.numel() == 0:
        print(f"{name} is empty")
        return
    print(name, x.min().item(), x.max().item(), torch.sqrt(torch.mean(x**2)).item())

def wrapbox(x, cell, reciprocal_cell):
    """
    Wrap atomic coordinates into the primary simulation box.

    This function is used in periodic boundary conditions to ensure all
    atoms are within the primary simulation cell.

    Args:
        x (torch.Tensor): Atomic coordinates.
        cell (torch.Tensor): The simulation cell matrix.
        reciprocal_cell (torch.Tensor): The reciprocal of the simulation cell matrix.

    Returns:
        torch.Tensor: The wrapped coordinates.
    """
    q = torch.matmul(x, reciprocal_cell.t())
    q = q - torch.floor(q)
    return torch.matmul(q, cell)

def main():
    """
    This function parses command-line arguments, reads the simulation parameters
    from a file, sets up the simulation environment (including device and precision),
    and calls the dynamic() function to run the simulation.

    Command-line usage:
    python dynamic.py path/to/parameter_file.txt

    The parameter file should contain all necessary settings for the simulation.
    """
    sys.stdout = io.TextIOWrapper(
        open(sys.stdout.fileno(), "wb", 0), write_through=True
    )
    parser = argparse.ArgumentParser(prog="TinkerIO")
    parser.add_argument("param_file", type=Path, help="Parameter file")
    args = parser.parse_args()
    simulation_parameters = parse_input(args.param_file)

    device: str = simulation_parameters.get("device", "cpu")
    if device.startswith("cuda") or device.startswith("gpu"):
        if ":" in device:
            num = device.split(":")[-1]
            os.environ["CUDA_VISIBLE_DEVICES"] = num
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = "cuda"

    enable_x64 = simulation_parameters.get("double_precision", False)
    fprec = torch.float64 if enable_x64 else torch.float32
    torch.set_default_dtype(fprec)

    dynamic(simulation_parameters, device, fprec)

def dynamic(simulation_parameters, device, fprec):
    """
    Run a molecular dynamics simulation based on the provided parameters.

    This function sets up the simulation system, initializes the dynamics,
    and runs the main simulation loop. It handles output writing, periodic
    calculations, and summary reporting.

    Args:
        simulation_parameters (dict): A dictionary containing all simulation settings.
        device (str): The device to run the simulation on ('cpu' or 'cuda').
        fprec (torch.dtype): The floating-point precision to use (torch.float32 or torch.float64).

    The function doesn't return anything but writes output to files and the console.
    """
    tstart_dyn = time.time()
    t0 = time.time()
    t1 = t0
    tperstep = 0
    ek = 0  # Initialize ek

    model = load_model(simulation_parameters)
    model.to(device)

    system_data, conformation = load_system_data(simulation_parameters, fprec)
    nat = system_data["nat"]

    preproc_state, conformation = initialize_preprocessing(
        simulation_parameters, model, conformation, system_data
    )

    random_seed = simulation_parameters.get(
        "random_seed", np.random.randint(0, 2**32 - 1)
    )
    print(f"# random_seed: {random_seed}")
    torch.manual_seed(random_seed)

    step, update_conformation, dyn_state, system = initialize_dynamics(
        simulation_parameters, system_data, conformation, model, fprec, torch.Generator(device=device).manual_seed(random_seed)
    )

    dt = dyn_state["dt"]
    nsteps = int(simulation_parameters.get("nsteps", 1))  # Default to 1 if not provided
    start_time = 0.0
    start_step = 0

    tperstep = 0  

    Tdump = simulation_parameters.get("tdump", 1.0 / au.PS) * au.FS
    ndump = max(1, int(Tdump / dt))  # Ensure ndump is at least 1
    system_name = system_data["name"]
    
    traj_format = simulation_parameters.get("traj_format", "arc").lower()
    if traj_format == "xyz":
        traj_file = system_name + ".traj.xyz"
        write_frame = write_xyz_frame
    elif traj_format == "extxyz":
        traj_file = system_name + ".traj.extxyz"
        write_frame = write_extxyz_frame
    elif traj_format == "arc":
        traj_file = system_name + ".arc"
        write_frame = write_arc_frame
    else:
        raise ValueError(
            f"Unknown trajectory format '{traj_format}'. Supported formats are 'arc', 'xyz', and 'extxyz'"
        )

    estimate_pressure = dyn_state.get("estimate_pressure", False)
    if estimate_pressure and fprec == torch.float64:
        volume = system_data["pbc"]["volume"]
        coordinates = conformation["coordinates"]
        cell = conformation["cells"][0]
        ek = system.get("ek", torch.tensor(0.0))  # Use get() with a default value
        Pkin = (2 * au.KBAR) * ek / ((3.0 / au.BOHR**3) * volume)
        e, f, vir_t, _ = model._energy_and_forces_and_virial(
            model.variables, conformation
        )
        Pvir = -(torch.trace(vir_t[0]) * au.KBAR) / ((3.0 / au.BOHR**3) * volume)
        vstep = volume * 0.000001
        scalep = ((volume + vstep) / volume) ** (1.0 / 3.0)
        cellp = cell * scalep
        reciprocal_cell = torch.inverse(cellp)
        sysp = model.preprocess(
            **{
                **conformation,
                "coordinates": coordinates * scalep,
                "cells": cellp[None, :, :],
                "reciprocal_cells": reciprocal_cell[None, :, :],
            }
        )
        ep, _ = model._total_energy(model.variables, sysp)
        scalem = ((volume - vstep) / volume) ** (1.0 / 3.0)
        cellm = cell * scalem
        reciprocal_cell = torch.inverse(cellm)
        sysm = model.preprocess(
            **{
                **conformation,
                "coordinates": coordinates * scalem,
                "cells": cellm[None, :, :],
                "reciprocal_cells": reciprocal_cell[None, :, :],
            }
        )
        em, _ = model._total_energy(model.variables, sysm)
        Pvir_fd = -(ep[0] * au.KBAR - em[0] * au.KBAR) / (2.0 * vstep / au.BOHR**3)
        print(
            f"# Initial pressure: {Pkin+Pvir:.3f} (virial); {Pkin+Pvir_fd:.3f} (finite difference) ; Pkin: {Pkin:.3f} ; Pvir: {Pvir:.3f} ; Pvir_fd: {Pvir_fd:.3f}"
        )

    def check_nan(system):
        """
    Check if the system contains any NaN values in velocities or coordinates.

    Args:
        system (dict): A dictionary containing the system state, including 'vel' and 'coordinates'.

    Returns:
        bool: True if NaN values are found, False otherwise.
    """
        return torch.isnan(system["vel"]).any() or torch.isnan(system["coordinates"]).any()

    if system_data["pbc"] is not None:
        cell = system_data["pbc"]["cell"]
        reciprocal_cell = system_data["pbc"]["reciprocal_cell"]
        do_wrap_box = simulation_parameters.get("wrap_box", False)
    else:
        cell = None
        reciprocal_cell = None
        do_wrap_box = False

    per_atom_energy = simulation_parameters.get("per_atom_energy", True)
    energy_unit_str = simulation_parameters.get("energy_unit", "kcal/mol")
    print("# Energy unit: ", energy_unit_str)
    energy_unit = au.get_multiplier(energy_unit_str)
    atom_energy_unit = energy_unit
    atom_energy_unit_str = energy_unit_str
    if per_atom_energy:
        atom_energy_unit /= nat
        atom_energy_unit_str = f"{energy_unit_str}/atom"
        print("# Printing Energy per atom")
    print(
        f"# Initial potential energy: {system['epot']*atom_energy_unit}; kinetic energy: {system['ek']*atom_energy_unit}"
    )
    f = system["forces"]
    minmaxone(torch.abs(f * energy_unit), "# forces min/max/rms:")

    print_timings = simulation_parameters.get("print_timings", False)
    nprint = int(simulation_parameters.get("nprint", 10))
    assert nprint > 0, "nprint must be > 0"
    nsummary = simulation_parameters.get("nsummary", 100 * nprint)
    assert nsummary > nprint, "nsummary must be > nprint"

    include_thermostat_energy = "thermostat_energy" in system["thermostat"]
    thermostat_name = dyn_state["thermostat_name"]
    pimd = dyn_state["pimd"]
    nbeads = system_data.get("nbeads", 1)
    dyn_name = "PIMD" if pimd else "MD"
    print("#" * 84)
    print(
        f"# Running {nsteps:_} steps of {thermostat_name} {dyn_name} simulation on {device}"
    )
    header = "#     Step   Time[ps]        Etot        Epot        Ekin    Temp[K]"
    if pimd:
        header += "  Temp_c[K]"
    if include_thermostat_energy:
        header += "      Etherm"
    if estimate_pressure:
        header += "  Press[kbar]"
    print(header)

    with open(traj_file, "a+") as fout:
        properties_traj = defaultdict(list)
        if print_timings:
            timings = defaultdict(lambda: 0.0)

        t0 = time.time()
        t0dump = t0
        istep = 0
        t0full = time.time()
        force_preprocess = False

        for istep in range(1, nsteps + 1):
            dyn_state, system, conformation, preproc_state = step(
                istep, dyn_state, system, conformation, preproc_state, force_preprocess
            )

            if istep % nprint == 0:
                t1 = time.time()
                if istep > 0:
                    tperstep = (t1 - t0) / nprint
                    nsperday = (24 * 60 * 60 / tperstep) * dt / 1e6
                else:
                    tperstep = 0
                    nsperday = 0
                t0 = t1

                ek = system.get("ek", torch.tensor(0.0))
                epot = system.get("epot", torch.tensor(0.0))
                etot = ek + epot
                temper = 2 * ek / (3.0 * nat) * au.KELVIN if nat > 0 else torch.tensor(0.0)

                th_state = system["thermostat"]
                if include_thermostat_energy:
                    etherm = th_state["thermostat_energy"]
                    etot = etot + etherm

                properties_traj[f"Etot[{atom_energy_unit_str}]"].append(
                    etot.item() * atom_energy_unit
                )
                properties_traj[f"Epot[{atom_energy_unit_str}]"].append(
                    epot.item() * atom_energy_unit
                )
                properties_traj[f"Ekin[{atom_energy_unit_str}]"].append(
                    ek.item() * atom_energy_unit
                )
                properties_traj["Temper[Kelvin]"].append(temper.item())
                if pimd:
                    ek_c = system["ek_c"]
                    temper_c = 2 * ek_c / (3.0 * nat) * au.KELVIN
                    properties_traj["Temper_c[Kelvin]"].append(temper_c.item())

                line = f"{istep:10.6g} {(start_time+istep*dt)/1000: 10.3f}  {etot.item()*atom_energy_unit: #10.4f}  {epot.item()*atom_energy_unit: #10.4f}  {ek.item()*atom_energy_unit: #10.4f} {temper.item(): 10.2f}"
                if pimd:
                    line += f" {temper_c.item(): 10.2f}"
                if include_thermostat_energy:
                    line += f"  {etherm.item()*atom_energy_unit: #10.4f}"
                    properties_traj[f"Etherm[{atom_energy_unit_str}]"].append(
                        etherm.item() * atom_energy_unit
                    )
                if estimate_pressure:
                    pres = system["pressure"]
                    properties_traj["Pressure[kbar]"].append(pres.item() * au.KBAR * au.BOHR**3)
                    line += f" {pres.item()*au.KBAR*au.BOHR**3:10.3f}"

                print(line)

            if istep % ndump == 0:
                line = "# Write XYZ frame"
                if do_wrap_box:
                    if pimd:
                        centroid = wrapbox(system["coordinates"][0], cell, reciprocal_cell)
                        system["coordinates"][0] = centroid
                    else:
                        system["coordinates"] = wrapbox(
                            system["coordinates"], cell, reciprocal_cell
                        )
                    conformation["coordinates"] = update_conformation(system["coordinates"])
                    line += " (atoms have been wrapped into the box)"
                    force_preprocess = True
                print(line)
                properties = {
                    "energy": float(system["epot"]) * energy_unit,
                    "Time": start_time + istep * dt,
                    "energy_unit": energy_unit_str,
                }
                write_frame(
                    fout,
                    system_data["symbols"],
                    conformation["coordinates"].reshape(nbeads, nat, 3)[0].cpu().numpy(),
                    cell=cell.cpu().numpy() if cell is not None else None,
                    properties=properties,
                    forces=None,
                )

            if istep % (nsummary) == 0:
                if check_nan(system):
                    raise ValueError(f"dynamics crashed at step {istep}.")
                tfull = time.time() - t0full
                t0full = time.time()
                tperstep = tfull / (nsummary)
                nsperday = (24 * 60 * 60 / tperstep) * dt / 1e6
                elapsed_time = time.time() - tstart_dyn
                estimated_remaining_time = tperstep * (nsteps - istep)
                estimated_total_time = elapsed_time + estimated_remaining_time

                print("#" * 50)
                print(f"# Step {istep:_} of {nsteps:_}  ({istep/nsteps*100:.5g} %)")
                print(f"# Tot. elapsed time   : {human_time_duration(elapsed_time)}")
                print(
                    f"# Est. total time     : {human_time_duration(estimated_total_time)}"
                )
                print(
                    f"# Est. remaining time : {human_time_duration(estimated_remaining_time)}"
                )
                print(f"# Time for {nsummary:_} steps : {human_time_duration(tfull)}")

                if print_timings:
                    print(f"# Detailed per-step timings :")
                    dsteps = nsummary
                    tother = tfull - sum([t for t in timings.values()])
                    timings["Other"] = tother
                    # sort timings
                    timings = {
                        k: v
                        for k, v in sorted(
                            timings.items(), key=lambda item: item[1], reverse=True
                        )
                    }
                    for k, v in timings.items():
                        print(
                            f"#   {k:15} : {human_time_duration(v/dsteps):>12} ({v/tfull*100:5.3g} %)"
                        )
                    print(f"#   {'Total':15} : {human_time_duration(tfull/dsteps):>12}")
                    ## reset timings
                    timings = defaultdict(lambda: 0.0)
                
                corr_kin = system["thermostat"].get("corr_kin",None)
                if corr_kin is not None:
                    print(f"# QTB kin. correction : {100*(corr_kin-1.):.2f} %")
                print(f"# Averages over last {nsummary:_} steps :")
                for k, v in properties_traj.items():
                    if len(properties_traj[k]) == 0:
                        continue
                    mu = np.mean(properties_traj[k])
                    sig = np.std(properties_traj[k])
                    ksplit = k.split("[")
                    name = ksplit[0].strip()
                    unit = ksplit[1].replace("]", "").strip() if len(ksplit) > 1 else ""
                    print(f"#   {name:10} : {mu: #10.5g}   +/- {sig: #9.3g}  {unit}")

                print(f"# Perf.: {nsperday:.2f} ns/day  ( {1.0 / tperstep:.2f} step/s )")
                print("#" * 50)
                if istep < nsteps:
                    print(header)

                if istep == nsteps:
                    break
                ## reset property trajectories
                properties_traj = defaultdict(list)

        print(f"# Run done in {human_time_duration(time.time()-tstart_dyn)}")

if __name__ == "__main__":
    main()
                
