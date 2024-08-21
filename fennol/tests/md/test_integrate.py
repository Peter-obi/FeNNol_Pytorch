import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from collections import defaultdict

from fennol.md.integrate import initialize_dynamics, initialize_integrator
from fennol.md.initial import initialize_system


# Mock dependencies
def setup_mock_model(mock_model):
    mock_preproc = Mock()
    mock_preproc.process.return_value = Mock()
    mock_preproc.check_reallocate.return_value = ({}, {}, Mock(), False)
    mock_model.preprocessing = mock_preproc

# Update the mock_model fixture
@pytest.fixture
def mock_model():
    model = Mock()
    model._energy_and_forces.return_value = (torch.tensor([0.0]), torch.zeros((3, 3)), None)
    model._energy_and_forces_and_virial.return_value = (torch.tensor([0.0]), torch.zeros((3, 3)), torch.zeros((3, 3, 3)), None)
    setup_mock_model(model)
    return model

@pytest.fixture
def mock_system_data():
    return {
        "mass": torch.tensor([1.0, 16.0, 1.0]),
        "nat": 3,
        "kT": 300 * 8.617333262145e-5,
        "species": torch.tensor([1, 8, 1])
    }

@pytest.fixture
def mock_conformation():
    return {
        "coordinates": torch.rand(3, 3),
        "cells": torch.tensor([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
    }

@pytest.fixture
def mock_simulation_parameters():
    return {
        "dt": 0.1,
        "thermostat": "Langevin",
        "temperature": 300,
        "friction": 0.1,
        "nblist_skin": 2.0,
        "nblist_stride": 10,
        "nblist_warmup_time": 1.0,
        "estimate_pressure": False
    }

def test_initialize_dynamics(mock_model, mock_system_data, mock_conformation, mock_simulation_parameters):
    fprec = torch.float32
    generator = torch.Generator()
    
    step, update_conformation, dyn_state, system = initialize_dynamics(
        mock_simulation_parameters, mock_system_data, mock_conformation, mock_model, fprec, generator
    )
    
    assert callable(step)
    assert callable(update_conformation)
    assert isinstance(dyn_state, dict)
    assert isinstance(system, dict)
    
    assert "dt" in dyn_state
    assert "pimd" in dyn_state
    assert "thermostat_name" in dyn_state
    
    assert "coordinates" in system
    assert "vel" in system
    assert "forces" in system
    assert "epot" in system
    assert "ek" in system
    assert "thermostat" in system

def test_initialize_integrator(mock_model, mock_system_data, mock_simulation_parameters):
    fprec = torch.float32
    generator = torch.Generator()
    
    step, update_conformation, dyn_state, thermostat_state, vel = initialize_integrator(
        mock_simulation_parameters, mock_system_data, mock_model, fprec, generator
    )
    
    assert callable(step)
    assert callable(update_conformation)
    assert isinstance(dyn_state, dict)
    assert isinstance(thermostat_state, dict)
    assert isinstance(vel, torch.Tensor)
    
    assert "dt" in dyn_state
    assert "pimd" in dyn_state
    assert "thermostat_name" in dyn_state
    assert "nblist_countdown" in dyn_state

def test_initialize_system(mock_model, mock_system_data, mock_conformation):
    fprec = torch.float32
    vel = torch.rand(3, 3)
    
    system = initialize_system(mock_conformation, vel, mock_model, mock_system_data, fprec)
    
    assert isinstance(system, dict)
    assert "coordinates" in system
    assert "vel" in system
    assert "forces" in system
    assert "epot" in system
    assert "ek" in system



@pytest.mark.parametrize("nbeads", [None, 4])
def test_pimd_initialization(mock_model, mock_system_data, mock_conformation, mock_simulation_parameters, nbeads):
    fprec = torch.float32
    generator = torch.Generator()
    
    if nbeads is not None:
        mock_system_data["nbeads"] = nbeads
        mock_system_data["omk"] = torch.rand(nbeads)
        mock_system_data["eigmat"] = torch.rand(nbeads, nbeads)
        mock_conformation["coordinates"] = torch.rand(nbeads * mock_system_data["nat"], 3)
        
        mock_model._energy_and_forces.return_value = (
            torch.tensor([0.0] * nbeads),
            torch.zeros((nbeads * mock_system_data["nat"], 3)),
            None
        )
        mock_model._energy_and_forces_and_virial.return_value = (
            torch.tensor([0.0] * nbeads),
            torch.zeros((nbeads * mock_system_data["nat"], 3)),
            torch.zeros((nbeads, 3, 3)),
            None
        )
    
    step, update_conformation, dyn_state, system = initialize_dynamics(
        mock_simulation_parameters, mock_system_data, mock_conformation, mock_model, fprec, generator
    )
    
    assert dyn_state["pimd"] == (nbeads is not None)
    
    expected_shape = (nbeads, mock_system_data["nat"], 3) if nbeads else (mock_system_data["nat"], 3)
    assert system["coordinates"].shape == expected_shape
    assert system["forces"].shape == expected_shape
    assert system["vel"].shape == expected_shape
    
    assert system["ek"].numel() == 1, "Expected a single value for total kinetic energy"
    assert system["epot"].numel() == 1, "Expected a single value for total potential energy"
    
    expected_system_keys = ["coordinates", "vel", "forces", "epot", "ek", "thermostat"]
    assert all(key in system for key in expected_system_keys), "Missing expected keys in system"
    
    expected_dyn_state_keys = ["dt", "estimate_pressure", "thermostat_name", "pimd"]
    assert all(key in dyn_state for key in expected_dyn_state_keys), "Missing expected keys in dyn_state"
    
    assert callable(step), "Expected 'step' to be a callable function"
    assert callable(update_conformation), "Expected 'update_conformation' to be a callable function"


@pytest.mark.parametrize("thermostat_name", ["Langevin", "NoseHoover"])
def test_thermostat_initialization(mock_model, mock_system_data, mock_conformation, mock_simulation_parameters, thermostat_name):
    fprec = torch.float32
    generator = torch.Generator()
    
    mock_simulation_parameters["thermostat"] = thermostat_name
    
    step, update_conformation, dyn_state, system = initialize_dynamics(
        mock_simulation_parameters, mock_system_data, mock_conformation, mock_model, fprec, generator
    )
    
    assert dyn_state["thermostat_name"].upper() == thermostat_name.upper()
    assert "thermostat" in system

def test_pressure_estimation(mock_model, mock_system_data, mock_conformation, mock_simulation_parameters):
    fprec = torch.float32
    generator = torch.Generator()
    
    mock_simulation_parameters["estimate_pressure"] = True
    mock_system_data["pbc_data"] = {"pscale": 1.0, "estimate_pressure": True}
    
    step, update_conformation, dyn_state, system = initialize_dynamics(
        mock_simulation_parameters, mock_system_data, mock_conformation, mock_model, fprec, generator
    )
    
    assert dyn_state["estimate_pressure"] == True
    
    # Run one step to check if pressure is calculated
    new_dyn_state, new_system, new_conformation, _ = step(1, dyn_state, system, mock_conformation, {})
    assert "pressure" in new_system

def test_nblist_parameters(mock_model, mock_system_data, mock_conformation, mock_simulation_parameters):
    fprec = torch.float32
    generator = torch.Generator()
    
    step, update_conformation, dyn_state, system = initialize_dynamics(
        mock_simulation_parameters, mock_system_data, mock_conformation, mock_model, fprec, generator
    )
    
    assert "nblist_countdown" in dyn_state
    assert dyn_state["nblist_countdown"] == 0  # Should start at 0
    
    # Run multiple steps to check nblist update behavior
    for i in range(1, 20):
        dyn_state, system, conformation, _ = step(i, dyn_state, system, mock_conformation, {})
        assert 0 <= dyn_state["nblist_countdown"] < mock_simulation_parameters["nblist_stride"]



if __name__ == "__main__":
    pytest.main([__file__])