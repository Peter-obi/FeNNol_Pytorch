import pytest
import torch
import numpy as np
from unittest.mock import Mock

from fennol.md.thermostats import get_thermostat
from fennol.utils.atomic_units import AtomicUnits as au

@pytest.fixture
def mock_simulation_parameters():
    return {
        "dt": 0.1,
        "thermostat": "Langevin",
        "temperature": 300,
        "friction": 0.1,
        "include_thermostat_energy": True,
        "nsteps": 1000
    }

@pytest.fixture
def mock_system_data():
    return {
        "mass": torch.tensor([1.0, 16.0, 1.0]),
        "nat": 3,
        "kT": 300 * 8.617333262145e-5,
        "species": torch.tensor([1, 8, 1])
    }

@pytest.fixture
def fprec():
    return torch.float32

@pytest.fixture
def generator():
    return torch.Generator().manual_seed(42)

def test_langevin_thermostat(mock_simulation_parameters, mock_system_data, fprec, generator):
    mock_simulation_parameters["thermostat"] = "Langevin"
    thermostat, postprocess, state, vel, thermostat_name = get_thermostat(
        mock_simulation_parameters, mock_simulation_parameters["dt"], mock_system_data, fprec, generator
    )
    
    assert thermostat_name.upper() == "LANGEVIN"
    assert callable(thermostat)
    assert vel.shape == (mock_system_data["nat"], 3)
    assert "generator" in state
    assert "thermostat_energy" in state

    # Test thermostat function
    new_vel, new_state = thermostat(vel, state)
    assert new_vel.shape == vel.shape
    assert "thermostat_energy" in new_state

@pytest.mark.parametrize("thermostat_name", [
    "LGV", "LANGEVIN", "FFLGV", "BUSSI", "GD", "DESCENT", "GRADIENT_DESCENT",
    "MIN", "MINIMIZE", "NVE", "NONE", "NOSE", "NOSEHOOVER", "NOSE_HOOVER",
    "ANNEAL", "ANNEALING"
])
def test_thermostat_initialization(mock_simulation_parameters, mock_system_data, fprec, generator, thermostat_name):
    mock_simulation_parameters["thermostat"] = thermostat_name
    thermostat, postprocess, state, vel, returned_thermostat_name = get_thermostat(
        mock_simulation_parameters, mock_simulation_parameters["dt"], mock_system_data, fprec, generator
    )
    
    assert returned_thermostat_name.upper() == thermostat_name.upper()
    assert callable(thermostat)
    assert vel.shape == (mock_system_data["nat"], 3)

def test_pimd_thermostat(mock_simulation_parameters, mock_system_data, fprec, generator):
    mock_simulation_parameters["thermostat"] = "Langevin"
    mock_system_data["nbeads"] = 4
    mock_system_data["omk"] = torch.rand(4)
    
    thermostat, postprocess, state, vel, thermostat_name = get_thermostat(
        mock_simulation_parameters, mock_simulation_parameters["dt"], mock_system_data, fprec, generator
    )
    
    assert thermostat_name.upper() == "LANGEVIN"
    assert vel.shape == (4, mock_system_data["nat"], 3)

def test_qtb_thermostat(mock_simulation_parameters, mock_system_data, fprec, generator):
    mock_simulation_parameters["thermostat"] = "QTB"
    mock_simulation_parameters["qtb"] = {
        "tseg": 1.0 / au.PS,
        "omegacut": 15000.0 / au.CM1,
        "gammar_min": 0.1,
        "write_spectra": False
    }
    
    thermostat, postprocess, state, vel, thermostat_name = get_thermostat(
        mock_simulation_parameters, mock_simulation_parameters["dt"], mock_system_data, fprec, generator
    )
    
    assert thermostat_name.upper() == "QTB"
    assert "force" in state
    assert callable(postprocess[0])
    assert isinstance(postprocess[1], dict)

def test_annealing_thermostat(mock_simulation_parameters, mock_system_data, fprec, generator):
    mock_simulation_parameters["thermostat"] = "ANNEAL"
    mock_simulation_parameters["annealing"] = {
        "init_factor": 0.04,
        "final_factor": 0.0001,
        "anneal_steps": 0.5,
        "warmup_steps": 0.2,
        "type": "linear"
    }
    
    thermostat, postprocess, state, vel, thermostat_name = get_thermostat(
        mock_simulation_parameters, mock_simulation_parameters["dt"], mock_system_data, fprec, generator
    )
    
    assert thermostat_name.upper() == "ANNEAL"
    assert "istep_anneal" in state

    # Test temperature change
    initial_vel = vel.clone()
    for _ in range(10):
        vel, state = thermostat(vel, state)
    
    assert not torch.allclose(vel, initial_vel), "Velocity should change during annealing"

def test_nose_hoover_thermostat(mock_simulation_parameters, mock_system_data, fprec, generator):
    mock_simulation_parameters["thermostat"] = "NOSE_HOOVER"
    
    thermostat, postprocess, state, vel, thermostat_name = get_thermostat(
        mock_simulation_parameters, mock_simulation_parameters["dt"], mock_system_data, fprec, generator
    )
    
    assert thermostat_name.upper() == "NOSE_HOOVER"
    assert "nose_s" in state
    assert "nose_v" in state

def test_thermostat_energy_conservation(mock_simulation_parameters, mock_system_data, fprec, generator):
    mock_simulation_parameters["thermostat"] = "NVE"
    
    thermostat, postprocess, state, vel, thermostat_name = get_thermostat(
        mock_simulation_parameters, mock_simulation_parameters["dt"], mock_system_data, fprec, generator
    )
    
    initial_energy = 0.5 * torch.sum(mock_system_data["mass"].unsqueeze(1) * vel**2)
    for _ in range(100):
        vel, state = thermostat(vel, state)
    final_energy = 0.5 * torch.sum(mock_system_data["mass"].unsqueeze(1) * vel**2)
    
    assert torch.isclose(initial_energy, final_energy, rtol=1e-5), "Energy should be conserved in NVE ensemble"

if __name__ == "__main__":
    pytest.main([__file__])