import sys
import os
import time
from pathlib import Path
import math

import numpy as np
import torch
from typing import Optional, Callable
from collections import defaultdict
from functools import partial
from ..utils.io import last_xyz_frame
from ..utils.periodic_table import PERIODIC_TABLE_REV_IDX, ATOMIC_MASSES
from ..utils.atomic_units import AtomicUnits as au
from ..utils.input_parser import parse_input
from .thermostats import get_thermostat

from copy import deepcopy
from .initial import initialize_system

def initialize_dynamics(
    simulation_parameters, system_data, conformation, model, fprec, generator
):
    step, update_conformation, dyn_state, thermostat_state, vel = initialize_integrator(
        simulation_parameters, system_data, model, fprec, generator
    )
    system = initialize_system(
        conformation,
        vel,
        model,
        system_data,
        fprec,
    )
    system["thermostat"] = thermostat_state
    return step, update_conformation, dyn_state, system

def initialize_integrator(simulation_parameters, system_data, model, fprec, generator):
    dt = simulation_parameters.get("dt") * au.FS
    dt2 = 0.5 * dt
    nbeads = system_data.get("nbeads", None)

    mass = system_data["mass"]
    nat = system_data["nat"]
    dt2m = (dt2 / mass.unsqueeze(1)).to(dtype=fprec)
    if nbeads is not None:
        dt2m = dt2m.unsqueeze(0)

    dyn_state = {
        "istep": 0,
        "dt": dt,
        "pimd": nbeads is not None,
    }

    thermostat, thermostat_post, thermostat_state, vel, dyn_state["thermostat_name"] = (
        get_thermostat(simulation_parameters, dt, system_data, fprec, generator)
    )

    do_thermostat_post = thermostat_post is not None
    if do_thermostat_post:
        thermostat_post, post_state = thermostat_post
        dyn_state["thermostat_post_state"] = post_state

    pbc_data = system_data.get("pbc_data", None)
    if pbc_data is not None:
        pscale = pbc_data["pscale"]
        estimate_pressure = pbc_data["estimate_pressure"]
    else:
        pscale = 1.0
        estimate_pressure = False

    dyn_state["estimate_pressure"] = estimate_pressure

    dyn_state["print_timings"] = simulation_parameters.get("print_timings", False)
    if dyn_state["print_timings"]:
        dyn_state["timings"] = defaultdict(lambda: 0.0)

    nblist_verbose = simulation_parameters.get("nblist_verbose", False)
    nblist_stride = int(simulation_parameters.get("nblist_stride", -1))
    nblist_warmup_time = simulation_parameters.get("nblist_warmup_time", -1.0) * au.FS
    nblist_warmup = int(nblist_warmup_time / dt) if nblist_warmup_time > 0 else 0
    nblist_skin = simulation_parameters.get("nblist_skin", -1.0)
    if nblist_skin > 0:
        if nblist_stride <= 0:
            t_ref = 40.0  # FS
            nblist_skin_ref = 2.0  # A
            nblist_stride = int(math.floor(nblist_skin / nblist_skin_ref * t_ref / dt))
        print(
            f"# nblist_skin: {nblist_skin:.2f} A, nblist_stride: {nblist_stride} steps, nblist_warmup: {nblist_warmup} steps"
        )

    if nblist_skin <= 0:
        nblist_stride = 1

    dyn_state["nblist_countdown"] = 0
    dyn_state["print_skin_activation"] = nblist_warmup > 0

    if nbeads is not None:
        cay_correction = simulation_parameters.get("cay_correction", True)
        omk = system_data["omk"]
        eigmat = system_data["eigmat"]
        cayfact = 1.0 / torch.sqrt(4.0 + (dt * omk[1:, None, None]) ** 2)
        if cay_correction:
            axx = 2 * cayfact
            axv = dt * cayfact
            avx = -dt * cayfact * omk[1:, None, None] ** 2
        else:
            axx = torch.cos(omk[1:, None, None] * dt2)
            axv = torch.sin(omk[1:, None, None] * dt2) / omk[1:, None, None]
            avx = -omk[1:, None, None] * torch.sin(omk[1:, None, None] * dt2)

        def update_conformation(eigx):
            return torch.einsum("in,n...->i...", eigmat, eigx).reshape(
                nbeads * nat, 3
            ) * (nbeads**0.5)

        def coords_to_eig(x):
            return torch.einsum("in,i...->n...", eigmat, x.reshape(nbeads, nat, 3)) * (
                1.0 / nbeads**0.5
            )
            
        def halfstep_free_polymer(eigx0, eigv0):
            eigx_c = eigx0[0] + dt2 * eigv0[0]
            eigv_c = eigv0[0]
            eigx = eigx0[1:] * axx + eigv0[1:] * axv
            eigv = eigx0[1:] * avx + eigv0[1:] * axx

            return torch.cat((eigx_c.unsqueeze(0), eigx), dim=0), torch.cat((eigv_c.unsqueeze(0), eigv), dim=0)
        
        
        def stepA(system):
            eigx = system["coordinates"]
            eigv = system["vel"] + dt2m * system["forces"]
            eigx, eigv = halfstep_free_polymer(eigx, eigv)
            eigv, thermostat_state = thermostat(eigv, system["thermostat"])
            eigx, eigv = halfstep_free_polymer(eigx, eigv)

            return {
                **system,
                "coordinates": eigx,
                "vel": eigv,
                "thermostat": thermostat_state,
            }
        
        def update_forces(system, conformation):
            if estimate_pressure:
                epot, f, vir_t, _ = model._energy_and_forces_and_virial(
                    model.variables, conformation
                )
                return {
                    **system,
                    "forces": coords_to_eig(f),
                    "epot": torch.mean(epot),
                    "virial": torch.trace(torch.mean(vir_t, dim=0)),
                }
            else:
                epot, f, _ = model._energy_and_forces(model.variables, conformation)
                return {**system, "forces": coords_to_eig(f), "epot": torch.mean(epot)}
        
        def stepB(system):
            eigv = system["vel"] + dt2m * system["forces"]

            ek_c = 0.5 * torch.sum(mass.unsqueeze(1) * eigv[0] ** 2)
            ek = ek_c - 0.5 * torch.sum(system["coordinates"][1:] * system["forces"][1:])
            system = {**system, "vel": eigv, "ek": ek, "ek_c": ek_c}

            if estimate_pressure:
                vir = system["virial"]
                Pkin = (2 * pscale) * ek
                Pvir = (-pscale) * vir
                system["pressure"] = Pkin + Pvir

            return system

    else:
        def update_conformation(coordinates):
            return coordinates
        
        def stepA(system):
            v = system["vel"]
            f = system["forces"]
            x = system["coordinates"]

            v = v + f * dt2m
            x = x + dt2 * v
            v, state_th = thermostat(v, system["thermostat"])
            x = x + dt2 * v

            new_system = dict(system)
            new_system["coordinates"] = x
            new_system["vel"] = v
            new_system["thermostat"] = state_th
            return new_system
        

        def update_forces(system, conformation):
            if estimate_pressure:
                epot, f, vir_t, _ = model._energy_and_forces_and_virial(
                    model.variables, conformation
                )
                return {
                    **system,
                    "forces": f,
                    "epot": epot[0],
                    "virial": torch.trace(vir_t[0]),
                }
            else:
                epot, f, _ = model._energy_and_forces(model.variables, conformation)
                return {**system, "forces": f, "epot": epot[0]}

        def stepB(system):
            v = system["vel"]
            f = system["forces"]
            state_th = system["thermostat"]

            v = v + f * dt2m
            ek = 0.5 * torch.sum(mass.unsqueeze(1) * v**2) / state_th.get("corr_kin", 1.0)
            system = {
                **system,
                "vel": v,
                "ek": ek,
            }

            if estimate_pressure:
                vir = system["virial"]
                Pkin = (2 * pscale) * ek
                Pvir = (-pscale) * vir
                system["pressure"] = Pkin + Pvir

            return system

    def step(
        istep, dyn_state, system, conformation, preproc_state, force_preprocess=False
    ):
        tstep0 = time.time()
        print_timings = "timings" in dyn_state

        dyn_state = {
            **dyn_state,
            "istep": dyn_state["istep"] + 1,
        }
        if print_timings:
            prev_timings = dyn_state["timings"]
            timings = defaultdict(lambda: 0.0)
            timings.update(prev_timings)

        system = stepA(system)

        if print_timings:
            torch.cuda.synchronize()
            timings["Integrator"] += time.time() - tstep0
            tstep0 = time.time()

        nblist_countdown = dyn_state["nblist_countdown"]
        if nblist_countdown <= 0 or force_preprocess or (istep < nblist_warmup):
            dyn_state["nblist_countdown"] = nblist_stride - 1
            conformation = model.preprocessing.process(
                preproc_state,
                {
                    **conformation,
                    "coordinates": update_conformation(system["coordinates"]),
                },
            )
            preproc_state, state_up, conformation, overflow = (
                model.preprocessing.check_reallocate(preproc_state, conformation)
            )
            if nblist_verbose and overflow:
                print("step", istep, ", nblist overflow => reallocating nblist")
                print("size updates:", state_up)

            if print_timings:
                torch.cuda.synchronize()
                timings["Preprocessing"] += time.time() - tstep0
                tstep0 = time.time()

        else:
            if dyn_state["print_skin_activation"]:
                if nblist_verbose:
                    print(
                        "step",
                        istep,
                        ", end of nblist warmup phase => activating skin updates",
                    )
                dyn_state["print_skin_activation"] = False

            dyn_state["nblist_countdown"] = nblist_countdown - 1
            conformation = model.preprocessing.update_skin(
                {
                    **conformation,
                    "coordinates": update_conformation(system["coordinates"]),
                }
            )

            if print_timings:
                torch.cuda.synchronize()
                timings["Skin update"] += time.time() - tstep0
                tstep0 = time.time()

        system = update_forces(system, conformation)
        if print_timings:
            torch.cuda.synchronize()
            timings["Forces"] += time.time() - tstep0
            tstep0 = time.time()

        system = stepB(system)

        if do_thermostat_post:
            system["thermostat"], dyn_state["thermostat_post_state"] = thermostat_post(
                system["thermostat"], dyn_state["thermostat_post_state"]
            )

        if print_timings:
            torch.cuda.synchronize()
            timings["Integrator"] += time.time() - tstep0
            tstep0 = time.time()

            dyn_state["timings"] = timings

        return dyn_state, system, conformation, preproc_state

    return step, update_conformation, dyn_state, thermostat_state, vel
