import numpy as np
import torch
import math
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..utils.atomic_units import AtomicUnits as au
from ..utils import Counter
from ..utils.deconvolution import (
    deconvolute_spectrum,
    kernel_lorentz_pot,
    kernel_lorentz,
)

def linear_onecycle_schedule(peak_value, div_factor, final_div_factor, total_steps, pct_start, pct_final):
    """
    Implements a linear one-cycle learning rate schedule.

    Args:
        peak_value (float): Maximum value of the schedule.
        div_factor (float): Initial value divisor.
        final_div_factor (float): Final value divisor.
        total_steps (int): Total number of steps in the schedule.
        pct_start (float): Percentage of total steps for the warmup phase.
        pct_final (float): Percentage of total steps before the cooldown phase.

    Returns:
        function: A function that takes a step number and returns the scheduled value.
    """
    def schedule(step):
        if step < pct_start * total_steps:
            return peak_value / div_factor + (peak_value - peak_value / div_factor) * (step / (pct_start * total_steps))
        elif step < pct_final * total_steps:
            return peak_value
        else:
            return peak_value / final_div_factor + (peak_value - peak_value / final_div_factor) * (1 - (step - pct_final * total_steps) / ((1 - pct_final) * total_steps))
    return schedule

def cosine_onecycle_schedule(peak_value, div_factor, final_div_factor, total_steps, pct_start):
    """
    Implements a cosine one-cycle learning rate schedule.

    Args:
        peak_value (float): Maximum value of the schedule.
        div_factor (float): Initial value divisor.
        final_div_factor (float): Final value divisor.
        total_steps (int): Total number of steps in the schedule.
        pct_start (float): Percentage of total steps for the warmup phase.

    Returns:
        function: A function that takes a step number and returns the scheduled value.
    """
    initial_value = peak_value / div_factor
    final_value = peak_value / final_div_factor
    
    def schedule(step):
        if step < pct_start * total_steps:
            cos_val = math.cos(math.pi * step / (pct_start * total_steps))
            return initial_value + 0.5 * (peak_value - initial_value) * (1 - cos_val)
        else:
            cos_val = math.cos(math.pi * (step - pct_start * total_steps) / ((1 - pct_start) * total_steps))
            return final_value + 0.5 * (peak_value - final_value) * (1 + cos_val)
    
    return schedule

def get_thermostat(simulation_parameters, dt, system_data, fprec, generator):
    """
    Creates and returns a thermostat function based on the specified parameters.

    Args:
        simulation_parameters (dict): Dictionary containing simulation parameters.
        dt (float): Time step for the simulation.
        system_data (dict): Dictionary containing system data (mass, species, etc.).
        fprec (torch.dtype): Floating-point precision for calculations.
        generator (torch.Generator): Random number generator for reproducibility.

    Returns:
        tuple: (thermostat_function, postprocess_function, initial_state, initial_velocities, thermostat_name)
    """
    state = {}
    postprocess = None

    thermostat_name = str(simulation_parameters.get("thermostat", "LGV")).upper()
    compute_thermostat_energy = simulation_parameters.get(
        "include_thermostat_energy", False
    )

    kT = system_data.get("kT", None)
    nbeads = system_data.get("nbeads", None)
    mass = system_data["mass"]
    gamma = simulation_parameters.get("gamma", 1.0 / au.THZ) / au.FS
    species = system_data["species"]

    if nbeads is not None:
        trpmd_lambda = simulation_parameters.get("trpmd_lambda", 1.0)
        gamma = torch.max(trpmd_lambda * system_data["omk"], torch.tensor(gamma))

    if thermostat_name in ["LGV", "LANGEVIN", "FFLGV"]:
        assert generator is not None, "generator must be provided for QTB thermostat"
        assert kT is not None, "kT must be specified for QTB thermostat"
        assert gamma is not None, "gamma must be specified for QTB thermostat"
        if nbeads is None:
            a1 = math.exp(-gamma * dt)
            a2 = torch.sqrt(((1 - a1 * a1) * kT / mass.unsqueeze(1))).to(dtype=fprec)
            vel = torch.randn(mass.shape[0], 3, dtype=fprec, generator=generator) * torch.sqrt(kT / mass.unsqueeze(1))
        else:
            if isinstance(gamma, float):
                gamma = torch.full((nbeads,), gamma)
            assert isinstance(gamma, torch.Tensor), "gamma must be a float or a torch.Tensor"
            assert gamma.shape[0] == nbeads, "gamma must have the same length as nbeads"
            a1 = torch.exp(-gamma * dt).unsqueeze(1).unsqueeze(2)
            a2 = torch.sqrt(((1 - a1 * a1) * kT / mass.unsqueeze(0).unsqueeze(2))).to(dtype=fprec)
            vel = torch.randn(nbeads, mass.shape[0], 3, dtype=fprec, generator=generator) * torch.sqrt(kT / mass.unsqueeze(1))

        state["generator"] = generator
        if compute_thermostat_energy:
            state["thermostat_energy"] = torch.tensor(0.0, dtype=fprec)
        if thermostat_name == "FFLGV":
            
            def thermostat(vel, state):
                noise = torch.randn(vel.shape, dtype=vel.dtype, device=vel.device, generator=state["generator"])
                norm_vel = torch.norm(vel, dim=-1, keepdim=True)
                dirvel = vel / norm_vel
                if compute_thermostat_energy:
                    v2 = torch.sum(vel**2, dim=-1)
                vel = a1 * vel + a2 * noise
                new_norm_vel = torch.norm(vel, dim=-1, keepdim=True)
                vel = dirvel * new_norm_vel
                new_state = {**state, "generator": state["generator"]}
                if compute_thermostat_energy:
                    v2new = torch.sum(vel**2, dim=-1)
                    new_state["thermostat_energy"] = (
                        state["thermostat_energy"] + 0.5 * torch.sum(mass * (v2 - v2new))
                    )
                return vel, new_state
        else:
            def thermostat(vel, state):
                noise = torch.randn(vel.shape, dtype=vel.dtype, device=vel.device, generator=state["generator"])
                if compute_thermostat_energy:
                    v2 = torch.sum(vel**2, dim=-1)
                vel = a1 * vel + a2 * noise
                new_state = {**state, "generator": state["generator"]}
                if compute_thermostat_energy:
                    v2new = torch.sum(vel**2, dim=-1)
                    new_state["thermostat_energy"] = (
                        state["thermostat_energy"] + 0.5 * torch.sum(mass * (v2 - v2new))
                    )
                return vel, new_state

    elif thermostat_name in ["BUSSI"]:
        assert generator is not None, "generator must be provided for QTB thermostat"
        assert kT is not None, "kT must be specified for QTB thermostat"
        assert gamma is not None, "gamma must be specified for QTB thermostat"
        assert nbeads is None, "Bussi thermostat is not compatible with PIMD"

        a1 = math.exp(-gamma * dt)
        a2 = (1 - a1) * kT
        vel = torch.randn(mass.shape[0], 3, dtype=fprec, generator=generator) * torch.sqrt(kT / mass.unsqueeze(1))

        state["generator"] = generator
        if compute_thermostat_energy:
            state["thermostat_energy"] = torch.tensor(0.0, dtype=fprec)

        def thermostat(vel, state):
            noise = torch.randn(vel.shape, dtype=vel.dtype, device=vel.device, generator=state["generator"])
            new_state = {**state, "generator": state["generator"]}
            R2 = torch.sum(noise**2)
            R1 = noise[0, 0]
            c = a2 / torch.sum(mass.unsqueeze(1) * vel**2)
            d = torch.sqrt(a1 * c)
            scale = torch.sqrt(a1 + c * R2 + 2 * d * R1)
            if compute_thermostat_energy:
                dek = 0.5 * torch.sum(mass.unsqueeze(1) * vel**2) * (scale**2 - 1)
                new_state["thermostat_energy"] = state["thermostat_energy"] + dek
            return scale * vel, new_state

    elif thermostat_name in ["GD", "DESCENT", "GRADIENT_DESCENT", "MIN", "MINIMIZE"]:
        assert nbeads is None, "Gradient descent is not compatible with PIMD"
        a1 = math.exp(-gamma * dt)

        if nbeads is None:
            vel = torch.zeros((mass.shape[0], 3), dtype=fprec)
        else:
            vel = torch.zeros((nbeads, mass.shape[0], 3), dtype=fprec)

        def thermostat(vel, state):
            return a1 * vel, state

    elif thermostat_name in ["NVE", "NONE"]:
        if nbeads is None:
            vel = torch.randn(mass.shape[0], 3, dtype=fprec, generator=generator) * torch.sqrt(kT / mass.unsqueeze(1))
            kTsys = torch.sum(mass.unsqueeze(1) * vel**2) / (mass.shape[0] * 3)
            vel = vel * torch.sqrt(kT / kTsys)
        else:
            vel = torch.randn(nbeads, mass.shape[0], 3, dtype=fprec, generator=generator) * torch.sqrt(kT / mass.unsqueeze(0).unsqueeze(2))
            kTsys = torch.sum(mass.unsqueeze(0).unsqueeze(2) * vel**2, dim=(1, 2)) / (mass.shape[0] * 3)
            vel = vel * torch.sqrt(kT / kTsys.unsqueeze(1).unsqueeze(2))
        thermostat = lambda x, s: (x, s)

    elif thermostat_name in ["NOSE", "NOSEHOOVER", "NOSE_HOOVER"]:
        assert gamma is not None, "gamma must be specified for QTB thermostat"
        ndof = mass.shape[0] * 3
        nkT = ndof * kT
        nose_mass = nkT / gamma**2
        assert nbeads is None, "Nose-Hoover is not compatible with PIMD"
        state["nose_s"] = torch.tensor(0.0, dtype=fprec)
        state["nose_v"] = torch.tensor(0.0, dtype=fprec)
        if compute_thermostat_energy:
            state["thermostat_energy"] = torch.tensor(0.0, dtype=fprec)
        print(
            "# WARNING: Nose-Hoover thermostat is not well tested yet. Energy conservation is not guaranteed."
        )
        vel = torch.randn(mass.shape[0], 3, dtype=fprec, generator=generator) * torch.sqrt(kT / mass.unsqueeze(1))

        def thermostat(vel, state):
            nose_s = state["nose_s"]
            nose_v = state["nose_v"]
            kTsys = torch.sum(mass.unsqueeze(1) * vel**2)
            nose_v = nose_v + (0.5 * dt / nose_mass) * (kTsys - nkT)
            nose_s = nose_s + dt * nose_v
            vel = torch.exp(-nose_v * dt) * vel
            kTsys = torch.sum(mass.unsqueeze(1) * vel**2)
            nose_v = nose_v + (0.5 * dt / nose_mass) * (kTsys - nkT)
            new_state = {**state, "nose_s": nose_s, "nose_v": nose_v}

            if compute_thermostat_energy:
                new_state["thermostat_energy"] = (
                    nkT * nose_s + (0.5 * nose_mass) * nose_v**2
                )
            return vel, new_state

    elif thermostat_name in ["QTB", "ADQTB"]:
        assert nbeads is None, "QTB is not compatible with PIMD"
        qtb_parameters = simulation_parameters.get("qtb", None)
        assert qtb_parameters is not None, "qtb_parameters must be provided for QTB thermostat"
        assert generator is not None, "generator must be provided for QTB thermostat"
        assert kT is not None, "kT must be specified for QTB thermostat"
        assert gamma is not None, "gamma must be specified for QTB thermostat"
        assert species is not None, "species must be provided for QTB thermostat"
        vel = torch.randn(mass.shape[0], 3, dtype=fprec, generator=generator) * torch.sqrt(kT / mass.unsqueeze(1))

        thermostat, postprocess, qtb_state = initialize_qtb(
            qtb_parameters,
            fprec=fprec,
            dt=dt,
            mass=mass,
            gamma=gamma,
            kT=kT,
            species=species,
            generator=generator,
            adaptive=thermostat_name.startswith("AD"),
            compute_thermostat_energy=compute_thermostat_energy,
        )
        state = {**state, **qtb_state}

    elif thermostat_name in ["ANNEAL", "ANNEALING"]:
        assert generator is not None, "generator must be provided for QTB thermostat"
        assert kT is not None, "kT must be specified for QTB thermostat"
        assert gamma is not None, "gamma must be specified for QTB thermostat"
        assert nbeads is None, "ANNEAL is not compatible with PIMD"
        a1 = math.exp(-gamma * dt)
        a2 = torch.sqrt(((1 - a1 * a1) * kT / mass.unsqueeze(1))).to(dtype=fprec)

        anneal_parameters = simulation_parameters.get("annealing", {})
        init_factor = anneal_parameters.get("init_factor", 1.0 / 25.0)
        assert init_factor > 0.0, "init_factor must be positive"
        final_factor = anneal_parameters.get("final_factor", 1.0 / 10000.0)
        assert final_factor > 0.0, "final_factor must be positive"
        nsteps = simulation_parameters.get("nsteps")
        anneal_steps = anneal_parameters.get("anneal_steps", 1.0)
        assert (
            anneal_steps <= 1.0 and anneal_steps > 0.0
        ), "anneal_steps must be between 0 and nsteps"
        pct_start = anneal_parameters.get("warmup_steps", 0.3)
        assert (
            pct_start < 1.0 and pct_start > 0.0
        ), "warmup_steps must be between 0 and nsteps"

        anneal_type = anneal_parameters.get("type", "cosine_onecycle").lower()
        if anneal_type == "linear":
            schedule = linear_onecycle_schedule(
                peak_value=1.0,
                div_factor=1.0 / init_factor,
                final_div_factor=1.0 / final_factor,
                total_steps=int(anneal_steps * nsteps),
                pct_start=pct_start,
                pct_final=1.0,
            )
        elif anneal_type in ["cosine", "cosine_onecycle"]:
            schedule = cosine_onecycle_schedule(
                peak_value=1.0,
                div_factor=1.0 / init_factor,
                final_div_factor=1.0 / final_factor,
                total_steps=int(anneal_steps * nsteps),
                pct_start=pct_start,
            )
        else:
            raise ValueError(f"Unknown anneal_type {anneal_type}")
        
        state["generator"] = generator
        state["istep_anneal"] = 0

        Tscale = schedule(0)
        print(f"# ANNEAL: initial temperature = {Tscale*kT*au.KELVIN:.3e} K")
        vel = torch.randn(mass.shape[0], 3, dtype=fprec, generator=generator) * torch.sqrt(kT * Tscale / mass.unsqueeze(1))

        def thermostat(vel, state):
            noise = torch.randn(vel.shape, dtype=vel.dtype, device=vel.device, generator=state["generator"])
            Tscale = torch.tensor(schedule(state["istep_anneal"]), dtype=torch.float32)
            vel = a1 * vel + a2 * Tscale * noise
            return vel, {
                **state,
                "generator": state["generator"],
                "istep_anneal": state["istep_anneal"] + 1,
            }

    else:
        raise ValueError(f"Unknown thermostat {thermostat_name}")

    return thermostat, postprocess, state, vel, thermostat_name


def initialize_qtb(
    qtb_parameters,
    fprec,
    dt,
    mass,
    gamma,
    kT,
    species,
    generator,
    adaptive,
    compute_thermostat_energy=False,
):
    """
    Initializes the Quantum Thermal Bath (QTB) thermostat.

    Args:
        qtb_parameters (dict): Dictionary containing QTB-specific parameters.
        fprec (torch.dtype): Floating-point precision for calculations.
        dt (float): Time step for the simulation.
        mass (torch.Tensor): Masses of the particles.
        gamma (float): Friction coefficient.
        kT (float): Thermal energy (Boltzmann constant * Temperature).
        species (torch.Tensor): Species identifiers for each particle.
        generator (torch.Generator): Random number generator for reproducibility.
        adaptive (bool): Whether to use adaptive QTB.
        compute_thermostat_energy (bool): Whether to compute and track thermostat energy.

    Returns:
        tuple: (thermostat_function, postprocess_function, initial_state)
    """
    state = {}
    post_state = {}
    verbose = qtb_parameters.get("verbose", False)
    if compute_thermostat_energy:
        state["thermostat_energy"] = torch.tensor(0.0, dtype=fprec)

    mass = mass.to(dtype=fprec)

    nat = species.shape[0]
    species_set = set(species.tolist())
    nspecies = len(species_set)
    idx = {sp: i for i, sp in enumerate(species_set)}
    type_idx = torch.tensor([idx[sp.item()] for sp in species], dtype=torch.int64)

    n_of_type = torch.zeros(nspecies, dtype=torch.int64)
    for i in range(nspecies):
        n_of_type[i] = (type_idx == i).nonzero().shape[0]
    n_of_type = n_of_type.to(dtype=fprec)
    mass_idx = torch.zeros(nspecies, dtype=fprec)
    for i in range(nspecies):
        mass_idx[i] = mass[type_idx == i].sum() / n_of_type[i]

    corr_kin = qtb_parameters.get("corr_kin", -1)
    do_corr_kin = corr_kin <= 0
    if do_corr_kin:
        corr_kin = 1.0
    state["corr_kin"] = corr_kin
    post_state["corr_kin_prev"] = corr_kin
    post_state["do_corr_kin"] = do_corr_kin
    post_state["isame_kin"] = 0

    # spectra parameters
    omegasmear = np.pi / dt / 100.0
    Tseg = qtb_parameters.get("tseg", 1.0 / au.PS) * au.FS
    nseg = int(Tseg / dt)
    Tseg = nseg * dt
    dom = 2 * np.pi / (3 * Tseg)
    omegacut = qtb_parameters.get("omegacut", 15000.0 / au.CM1) / au.FS
    nom = int(omegacut / dom)
    omega = dom * torch.arange((3 * nseg) // 2 + 1)
    cutoff = (1.0 / (1.0 + torch.exp((omega - omegacut) / omegasmear))).to(dtype=fprec)
    assert (
        omegacut < omega[-1]
    ), f"omegacut must be smaller than {omega[-1]*au.CM1} CM-1"

    # initialize gammar
    assert (
        gamma < 0.5 * omegacut
    ), "gamma must be much smaller than omegacut (at most 0.5*omegacut)"
    gammar_min = qtb_parameters.get("gammar_min", 0.1)
    post_state["gammar"] = torch.ones((nspecies, nom), dtype=fprec)

    # Ornstein-Uhlenbeck correction for colored noise
    a1 = np.exp(-gamma * dt)
    OUcorr = ((1.0 - 2.0 * a1 * torch.cos(omega * dt) + a1**2) / (dt**2 * (gamma**2 + omega**2))).to(dtype=fprec)

    # hbar schedule
    classical_kernel = qtb_parameters.get("classical_kernel", False)
    hbar = qtb_parameters.get("hbar", 1.0) * au.FS
    u = 0.5 * hbar * torch.abs(omega) / kT
    theta = kT * torch.ones_like(omega)
    if hbar > 0:
        theta[1:] *= u[1:] / torch.tanh(u[1:])
    theta = theta.to(dtype=fprec)

    post_state["generator"] = generator
    post_state["white_noise"] = torch.randn((3 * nseg, nat, 3), dtype=fprec, device=mass.device, generator=generator)

    startsave = qtb_parameters.get("startsave", 1)
    counter = Counter(nseg, startsave=startsave)
    state["istep"] = 0
    post_state["nadapt"] = 0
    post_state["nsample"] = 0

    write_spectra = qtb_parameters.get("write_spectra", True)
    do_compute_spectra = write_spectra or adaptive

    if do_compute_spectra:
        state["vel"] = torch.zeros((nseg, nat, 3), dtype=fprec)

        post_state["dFDT"] = torch.zeros((nspecies, nom), dtype=fprec)
        post_state["mCvv"] = torch.zeros((nspecies, nom), dtype=fprec)
        post_state["Cvf"] = torch.zeros((nspecies, nom), dtype=fprec)
        post_state["Cff"] = torch.zeros((nspecies, nom), dtype=fprec)
        post_state["dFDT_avg"] = torch.zeros((nspecies, nom), dtype=fprec)
        post_state["mCvv_avg"] = torch.zeros((nspecies, nom), dtype=fprec)
        post_state["Cvfg_avg"] = torch.zeros((nspecies, nom), dtype=fprec)
        post_state["Cff_avg"] = torch.zeros((nspecies, nom), dtype=fprec)

    if not adaptive:
        update_gammar = lambda x: x
    else:
        # adaptation parameters
        skipseg = qtb_parameters.get("skipseg", 1)

        adaptation_method = (
            str(qtb_parameters.get("adaptation_method", "ADABELIEF")).upper().strip()
        )
        authorized_methods = ["SIMPLE", "RATIO", "ADABELIEF"]
        assert (
            adaptation_method in authorized_methods
        ), f"adaptation_method must be one of {authorized_methods}"
        if adaptation_method == "SIMPLE":
            agamma = qtb_parameters.get("agamma", 1.0e-3) / au.FS
            assert agamma > 0, "agamma must be positive"
            a1_ad = agamma * Tseg
            print(f"# ADQTB SIMPLE: agamma = {agamma*au.FS:.3f}")

            def update_gammar(post_state):
                g = post_state["dFDT"]
                gammar = post_state["gammar"] - a1_ad * g
                gammar = torch.clamp(gammar, min=gammar_min)
                return {**post_state, "gammar": gammar}

        elif adaptation_method == "RATIO":
            tau_ad = qtb_parameters.get("tau_ad", 5.0 / au.PS) * au.FS
            tau_s = qtb_parameters.get("tau_s", 10 * tau_ad) * au.FS
            assert tau_ad > 0, "tau_ad must be positive"
            print(
                f"# ADQTB RATIO: tau_ad = {tau_ad*1e-3:.2f} ps, tau_s = {tau_s*1e-3:.2f} ps"
            )
            b1 = np.exp(-Tseg / tau_ad)
            b2 = np.exp(-Tseg / tau_s)
            post_state["mCvv_m"] = torch.zeros((nspecies, nom), dtype=fprec)
            post_state["Cvf_m"] = torch.zeros((nspecies, nom), dtype=fprec)
            post_state["n_adabelief"] = 0

            def update_gammar(post_state):
                n_adabelief = post_state["n_adabelief"] + 1
                mCvv_m = post_state["mCvv_m"] * b1 + post_state["mCvv"] * (1.0 - b1)
                Cvf_m = post_state["Cvf_m"] * b2 + post_state["Cvf"] * (1.0 - b2)
                mCvv = mCvv_m / (1.0 - b1**n_adabelief)
                Cvf = Cvf_m / (1.0 - b2**n_adabelief)
                gammar = Cvf / (mCvv + 1.0e-8)
                gammar = torch.clamp(gammar, min=gammar_min)
                return {
                    **post_state,
                    "gammar": gammar,
                    "mCvv_m": mCvv_m,
                    "Cvf_m": Cvf_m,
                    "n_adabelief": n_adabelief,
                }
            
        elif adaptation_method == "ADABELIEF":
            agamma = qtb_parameters.get("agamma", 0.1)
            tau_ad = qtb_parameters.get("tau_ad", 1.0 / au.PS) * au.FS
            tau_s = qtb_parameters.get("tau_s", 100 * tau_ad) * au.FS
            assert tau_ad > 0, "tau_ad must be positive"
            assert tau_s > 0, "tau_s must be positive"
            assert agamma > 0, "agamma must be positive"
            print(
                f"# ADQTB ADABELIEF: agamma = {agamma:.3f}, tau_ad = {tau_ad*1.e-3:.2f} ps, tau_s = {tau_s*1.e-3:.2f} ps"
            )

            a1_ad = agamma * gamma
            b1 = np.exp(-Tseg / tau_ad)
            b2 = np.exp(-Tseg / tau_s)
            post_state["dFDT_m"] = torch.zeros((nspecies, nom), dtype=fprec)
            post_state["dFDT_s"] = torch.zeros((nspecies, nom), dtype=fprec)
            post_state["n_adabelief"] = 0

            def update_gammar(post_state):
                n_adabelief = post_state["n_adabelief"] + 1
                dFDT = post_state["dFDT"]
                dFDT_m = post_state["dFDT_m"] * b1 + dFDT * (1.0 - b1)
                dFDT_s = (
                    post_state["dFDT_s"] * b2
                    + (dFDT - dFDT_m) ** 2 * (1.0 - b2)
                    + 1.0e-8
                )
                # bias correction
                mt = dFDT_m / (1.0 - b1**n_adabelief)
                st = dFDT_s / (1.0 - b2**n_adabelief)
                gammar = post_state["gammar"] - a1_ad * mt / (st.sqrt() + 1.0e-8)
                gammar = torch.clamp(gammar, min=gammar_min)
                return {
                    **post_state,
                    "gammar": gammar,
                    "dFDT_m": dFDT_m,
                    "n_adabelief": n_adabelief,
                    "dFDT_s": dFDT_s,
                }

    def compute_corr_pot(niter=20, verbose=False):
        """
        Computes the potential energy correction factor for QTB.

        Args:
            niter (int): Number of iterations for deconvolution.
            verbose (bool): Whether to print verbose output.

        Returns:
            torch.Tensor: Potential energy correction factor.
        """
        if classical_kernel or hbar == 0:
            return torch.ones(nom)

        s_0 = (theta / kT * cutoff)[:nom].numpy()
        s_out, s_rec, _ = deconvolute_spectrum(
            s_0,
            omega[:nom].numpy(),
            gamma,
            niter,
            kernel=kernel_lorentz_pot,
            trans=True,
            symmetrize=True,
            verbose=verbose,
        )
        corr_pot = 1.0 + (s_out - s_0) / s_0
        columns = np.column_stack(
            (omega[:nom].numpy() * au.CM1, corr_pot - 1.0, s_0, s_out, s_rec)
        )
        np.savetxt(
            "corr_pot.dat", columns, header="omega(cm-1) corr_pot s_0 s_out s_rec"
        )
        return torch.as_tensor(corr_pot, dtype=fprec)

    def compute_corr_kin(post_state, niter=7, verbose=False):
        """
        Computes the kinetic energy correction factor for QTB.

        Args:
            post_state (dict): Current state of the QTB.
            niter (int): Number of iterations for deconvolution.
            verbose (bool): Whether to print verbose output.

        Returns:
            tuple: (kinetic_energy_correction, updated_post_state)
        """
        if not post_state["do_corr_kin"]:
            return post_state["corr_kin_prev"], post_state
        if classical_kernel or hbar == 0:
            return 1.0, post_state

        K_D = post_state.get("K_D", None)
        mCvv = (post_state["mCvv_avg"][:, :nom] * n_of_type.unsqueeze(1)).sum(dim=0) / nat
        s_0 = (mCvv * kT / theta[:nom] / post_state["corr_pot"]).numpy()
        s_out, s_rec, K_D = deconvolute_spectrum(
            s_0,
            omega[:nom].numpy(),
            gamma,
            niter,
            kernel=kernel_lorentz,
            trans=False,
            symmetrize=True,
            verbose=verbose,
            K_D=K_D,
        )
        s_out = torch.tensor(s_out) * theta[:nom] / kT
        s_rec = torch.tensor(s_rec) * theta[:nom] / kT * post_state["corr_pot"]
        mCvvsum = mCvv.sum()
        rec_ratio = mCvvsum / s_rec.sum()
        if rec_ratio < 0.95 or rec_ratio > 1.05:
            print(
                "# WARNING: reconvolution error is too high, corr_kin was not updated"
            )
            return post_state["corr_kin_prev"], post_state

        corr_kin = mCvvsum / s_out.sum()
        if torch.abs(corr_kin - post_state["corr_kin_prev"]) < 1.0e-4:
            isame_kin = post_state["isame_kin"] + 1
        else:
            isame_kin = 0

        do_corr_kin = post_state["do_corr_kin"]
        if isame_kin > 10:
            print(
                "# INFO: corr_kin is converged (it did not change for 10 consecutive segments)"
            )
            do_corr_kin = False

        return corr_kin, {
            **post_state,
            "corr_kin_prev": corr_kin,
            "isame_kin": isame_kin,
            "do_corr_kin": do_corr_kin,
            "K_D": K_D,
        }

    def ff_kernel(post_state):
        """
        Computes the force-force correlation kernel for QTB.

        Args:
            post_state (dict): Current state of the QTB.

        Returns:
            torch.Tensor: Force-force correlation kernel.
        """
        if classical_kernel:
            kernel = cutoff * (2 * gamma * kT / dt)
        else:
            kernel = theta * cutoff * OUcorr * (2 * gamma / dt)
        gamma_ratio = torch.cat(
            (
                post_state["gammar"].t() * post_state["corr_pot"].unsqueeze(1),
                torch.ones(
                    (kernel.shape[0] - nom, nspecies), dtype=post_state["gammar"].dtype
                ),
            ),
            dim=0,
        )
        return kernel.unsqueeze(1) * gamma_ratio * mass_idx

    def refresh_force(post_state):
        """
        Refreshes the random forces for QTB.

        Args:
            post_state (dict): Current state of the QTB.

        Returns:
            tuple: (new_forces, updated_post_state)
        """
        white_noise = torch.cat(
            (
                post_state["white_noise"][nseg:],
                torch.randn(
                    nseg, nat, 3, dtype=post_state["white_noise"].dtype, generator=post_state["generator"]
                ),
            ),
            dim=0,
        )
        amplitude = ff_kernel(post_state).sqrt()
        s = torch.fft.rfft(white_noise, 3 * nseg, dim=0) * amplitude[:, type_idx, None]
        force = torch.fft.irfft(s, 3 * nseg, dim=0)[nseg : 2 * nseg]
        return force, {**post_state, "generator": post_state["generator"], "white_noise": white_noise}

    def compute_spectra(force, vel, post_state):
        """
        Computes various spectra for QTB analysis.

        Args:
            force (torch.Tensor): Current forces.
            vel (torch.Tensor): Current velocities.
            post_state (dict): Current state of the QTB.

        Returns:
            dict: Updated post_state with computed spectra.
        """
        sf = torch.fft.rfft(force / gamma, 3 * nseg, dim=0, norm="ortho")
        sv = torch.fft.rfft(vel, 3 * nseg, dim=0, norm="ortho")
        Cvv = torch.sum(torch.abs(sv[:nom]) ** 2, dim=-1).t()
        Cff = torch.sum(torch.abs(sf[:nom]) ** 2, dim=-1).t()
        Cvf = torch.sum(torch.real(sv[:nom] * torch.conj(sf[:nom])), dim=-1).t()

        mCvv = (
            (dt / 3.0)
            * torch.zeros_like(post_state["mCvv"]).index_add_(0, type_idx, Cvv)
            * mass_idx.unsqueeze(1)
            / n_of_type.unsqueeze(1)
        )
        Cvf = (
            (dt / 3.0)
            * torch.zeros_like(post_state["Cvf"]).index_add_(0, type_idx, Cvf)
            / n_of_type.unsqueeze(1)
        )
        Cff = (
            (dt / 3.0)
            * torch.zeros_like(post_state["Cff"]).index_add_(0, type_idx, Cff)
            / n_of_type.unsqueeze(1)
        )
        dFDT = mCvv * post_state["gammar"] - Cvf

        nsinv = 1.0 / post_state["nsample"]
        b1 = 1.0 - nsinv
        dFDT_avg = post_state["dFDT_avg"] * b1 + dFDT * nsinv
        mCvv_avg = post_state["mCvv_avg"] * b1 + mCvv * nsinv
        Cvfg_avg = post_state["Cvfg_avg"] * b1 + Cvf / post_state["gammar"] * nsinv
        Cff_avg = post_state["Cff_avg"] * b1 + Cff * nsinv

        return {
            **post_state,
            "mCvv": mCvv,
            "Cvf": Cvf,
            "Cff": Cff,
            "dFDT": dFDT,
            "dFDT_avg": dFDT_avg,
            "mCvv_avg": mCvv_avg,
            "Cvfg_avg": Cvfg_avg,
            "Cff_avg": Cff_avg,
        }

    def write_spectra_to_file(post_state):
        """
        Writes QTB spectra to output files.

        Args:
            post_state (dict): Current state of the QTB.
        """
        mCvv_avg = post_state["mCvv_avg"].cpu().numpy()
        Cvfg_avg = post_state["Cvfg_avg"].cpu().numpy()
        Cff_avg = post_state["Cff_avg"].cpu().numpy() * 3.0 / dt / (gamma**2)
        dFDT_avg = post_state["dFDT_avg"].cpu().numpy()
        gammar = post_state["gammar"].cpu().numpy()
        Cff_theo = ff_kernel(post_state)[:nom].t().cpu().numpy()
        for i, sp in enumerate(species_set):
            ff_scale = au.KELVIN / ((2 * gamma / dt) * mass_idx[i])
            columns = np.column_stack(
                (
                    omega[:nom].cpu().numpy() * (au.FS * au.CM1),
                    mCvv_avg[i],
                    Cvfg_avg[i],
                    dFDT_avg[i],
                    gammar[i] * gamma * (au.FS * au.THZ),
                    Cff_avg[i] * ff_scale,
                    Cff_theo[i] * ff_scale,
                )
            )
            np.savetxt(
                f"QTB_spectra_{sp}.out",
                columns,
                fmt="%12.6f",
                header="#omega mCvv Cvf dFDT gammar Cff",
            )
        if verbose:
            print("# QTB spectra written.")

    if compute_thermostat_energy:
        state["qtb_energy_flux"] = torch.tensor(0.0, dtype=fprec)

    def thermostat(vel, state):
        """
        Applies the QTB thermostat to the velocities.

        Args:
            vel (torch.Tensor): Current velocities.
            state (dict): Current state of the simulation.

        Returns:
            tuple: (new_velocities, updated_state)
        """
        istep = state["istep"]
        dvel = dt * state["force"][istep] / mass.unsqueeze(1)
        new_vel = vel * a1 + dvel
        new_state = {**state, "istep": istep + 1}
        if do_compute_spectra:
            vel2 = state["vel"].index_copy(0, torch.tensor(istep), vel * a1**0.5 + 0.5 * dvel)
            new_state["vel"] = vel2
        if compute_thermostat_energy:
            dek = 0.5 * (mass.unsqueeze(1) * (vel**2 - new_vel**2)).sum()
            ekcorr = (
                0.5
                * (mass.unsqueeze(1) * new_vel**2).sum()
                * (1.0 - 1.0 / state.get("corr_kin", 1.0))
            )
            new_state["qtb_energy_flux"] = state["qtb_energy_flux"] + dek
            new_state["thermostat_energy"] = new_state["qtb_energy_flux"] + ekcorr
        return new_vel, new_state

    def postprocess_work(state, post_state):
        """
        Performs post-processing work for QTB.

        Args:
            state (dict): Current state of the simulation.
            post_state (dict): Current state of the QTB.

        Returns:
            tuple: (updated_state, updated_post_state)
        """
        if do_compute_spectra:
            post_state = compute_spectra(state["force"], state["vel"], post_state)
        if adaptive:
            post_state = update_gammar(post_state) if post_state["nadapt"] > skipseg else post_state
        new_force, post_state = refresh_force(post_state)
        return {**state, "force": new_force}, post_state

    def postprocess(state, post_state):
        """
        Main post-processing function for QTB.

        Args:
            state (dict): Current state of the simulation.
            post_state (dict): Current state of the QTB.

        Returns:
            tuple: (updated_state, updated_post_state)
        """
        counter.increment()
        if not counter.is_reset_step:
            return state, post_state
        post_state["nadapt"] += 1
        post_state["nsample"] = max(post_state["nadapt"] - startsave + 1, 1)
        if verbose:
            print("# Refreshing QTB forces.")
        state, post_state = postprocess_work(state, post_state)
        state["corr_kin"], post_state = compute_corr_kin(post_state)
        state["istep"] = 0
        if write_spectra:
            write_spectra_to_file(post_state)
        return state, post_state

    post_state["corr_pot"] = compute_corr_pot()

    state["force"], post_state = refresh_force(post_state)
    return thermostat, (postprocess, post_state), state
