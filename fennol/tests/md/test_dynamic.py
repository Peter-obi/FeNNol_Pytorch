import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock, call
from pathlib import Path
from collections import defaultdict
from itertools import count

from fennol.md.dynamic import (
    minmaxone,
    wrapbox,
    dynamic,
    human_time_duration,
    main
)
from fennol.utils.atomic_units import AtomicUnits as au

def check_nan(system):
    return torch.isnan(system["vel"]).any() or torch.isnan(system["coordinates"]).any()

# Fixtures

@pytest.fixture
def mock_simulation_parameters():
    return {
        "device": "cpu",
        "double_precision": False,
        "random_seed": 42,
        "nsteps": 100,
        "tdump": 1.0,
        "traj_format": "xyz",
        "wrap_box": True,
        "per_atom_energy": True,
        "energy_unit": "kcal/mol",
        "print_timings": False,
        "nprint": 10,
        "nsummary": 50,
    }

@pytest.fixture
def mock_model():
    model = MagicMock()
    model._energy_and_forces.return_value = (torch.tensor([1.0]), torch.tensor([[0.1, 0.1, 0.1]]), None)
    model._energy_and_forces_and_virial.return_value = (torch.tensor([1.0]), torch.tensor([[0.1, 0.1, 0.1]]), torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]), None)
    model._total_energy.return_value = (torch.tensor([1.0]), None)
    return model

@pytest.fixture
def mock_system_data():
    return {
        "name": "test_system",
        "nat": 3,
        "symbols": ["H", "O", "H"],
        "pbc": {
            "cell": torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]),
            "reciprocal_cell": torch.inverse(torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])),
            "volume": 1000.0,
            "pscale": 1.0,
        },
        "mass": torch.tensor([1.008, 15.999, 1.008]),
    }

@pytest.fixture
def mock_conformation():
    return {
        "coordinates": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        "species": torch.tensor([1, 8, 1]),
        "cells": torch.tensor([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]]),
        "reciprocal_cells": torch.inverse(torch.tensor([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])),
    }

@pytest.fixture
def mock_system():
    return {
        "epot": torch.tensor(1.0),
        "ek": torch.tensor(0.5),
        "forces": torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]]),
        "coordinates": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        "vel": torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]]),
        "thermostat": {"thermostat_energy": torch.tensor(0.1)},
        "pressure": torch.tensor(1.0),
    }

# Utility function tests

def test_minmaxone():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    with patch('builtins.print') as mock_print:
        minmaxone(x, "test")
    mock_print.assert_called_once_with("test", 1.0, 5.0, pytest.approx(3.3166, rel=1e-4))

def test_minmaxone_empty_tensor():
    x = torch.tensor([])
    with patch('builtins.print') as mock_print:
        minmaxone(x, "empty")
    mock_print.assert_called_once_with("empty is empty")

def test_wrapbox():
    x = torch.tensor([[11.0, 0.0, 0.0], [0.0, 11.0, 0.0], [0.0, 0.0, 11.0]])
    cell = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    reciprocal_cell = torch.inverse(cell)
    result = wrapbox(x, cell, reciprocal_cell)
    expected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(result, expected)

def test_wrapbox_edge_case():
    x = torch.tensor([[10.0, 10.0, 10.0], [-0.1, -0.1, -0.1]])
    cell = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    reciprocal_cell = torch.inverse(cell)
    result = wrapbox(x, cell, reciprocal_cell)
    expected = torch.tensor([[0.0, 0.0, 0.0], [9.9, 9.9, 9.9]])
    assert torch.allclose(result, expected, atol=1e-6)

def test_check_nan():
    system_with_nan = {
        "vel": torch.tensor([[float('nan'), 0.0, 0.0], [0.0, 0.0, 0.0]]),
        "coordinates": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    }
    system_without_nan = {
        "vel": torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        "coordinates": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    }
    assert check_nan(system_with_nan) == True
    assert check_nan(system_without_nan) == False

def test_check_nan_edge_cases():
    system_with_inf = {
        "vel": torch.tensor([[float('inf'), 0.0, 0.0], [0.0, 0.0, 0.0]]),
        "coordinates": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    }
    system_empty = {
        "vel": torch.tensor([]),
        "coordinates": torch.tensor([]),
    }
    assert check_nan(system_with_inf) == False
    assert check_nan(system_empty) == False

def test_human_time_duration():
    assert human_time_duration(3661) == "1 h 1 min 1 s"
    assert human_time_duration(86400) == "1 day"
    assert human_time_duration(90) == "1 min 30 s"
    assert human_time_duration(0) == "0 ms"
    assert human_time_duration(604800) == "1 week"
    assert human_time_duration(0.5) == "500 ms"
    assert human_time_duration(5.5) == "5.5 s"
    assert human_time_duration(3600 * 24 * 7 + 3600 * 24 + 3600 + 60 + 1) == "1 week 1 day 1 h 1 min 1 s"

# Main function tests

@patch('fennol.md.dynamic.dynamic')
@patch('fennol.md.dynamic.parse_input')
@patch('argparse.ArgumentParser.parse_args')
def test_main(mock_parse_args, mock_parse_input, mock_dynamic):
    mock_args = MagicMock()
    mock_args.param_file = Path("test_params.txt")
    mock_parse_args.return_value = mock_args
    mock_parse_input.return_value = {"device": "cpu", "double_precision": False}
    
    main()
    
    mock_parse_args.assert_called_once()
    mock_parse_input.assert_called_once_with(Path("test_params.txt"))
    mock_dynamic.assert_called_once()

# Dynamic function tests

@patch('fennol.md.dynamic.load_model')
@patch('fennol.md.dynamic.load_system_data')
@patch('fennol.md.dynamic.initialize_preprocessing')
@patch('fennol.md.dynamic.initialize_dynamics')
@patch('builtins.open')
@patch('fennol.md.dynamic.write_xyz_frame')
@patch('time.time')
def test_dynamic(mock_time, mock_write_xyz, mock_open, mock_init_dynamics, mock_init_preproc, 
                 mock_load_system, mock_load_model, mock_simulation_parameters, 
                 mock_model, mock_system_data, mock_conformation, mock_system):
    mock_load_model.return_value = mock_model
    mock_load_system.return_value = (mock_system_data, mock_conformation)
    mock_init_preproc.return_value = ({}, mock_conformation)
    mock_init_dynamics.return_value = (
        lambda i, d, s, c, p, f: (d, s, c, p),
        lambda x: x,
        {"dt": 0.1, "estimate_pressure": False, "thermostat_name": "Langevin", "pimd": False},
        mock_system
    )
    mock_time.side_effect = count(0, 1)

    dynamic(mock_simulation_parameters, "cpu", torch.float32)

    assert mock_load_model.called
    assert mock_load_system.called
    assert mock_init_preproc.called
    assert mock_init_dynamics.called
    assert mock_open.called
    assert mock_write_xyz.called
    assert mock_time.call_count > 1

@patch('fennol.md.dynamic.load_model')
@patch('fennol.md.dynamic.load_system_data')
@patch('fennol.md.dynamic.initialize_preprocessing')
@patch('fennol.md.dynamic.initialize_dynamics')
@patch('builtins.open')
@patch('fennol.md.dynamic.write_arc_frame')
@patch('time.time')
def test_dynamic_arc_format(mock_time, mock_write_arc, mock_open, mock_init_dynamics, mock_init_preproc, 
                            mock_load_system, mock_load_model, mock_simulation_parameters, 
                            mock_model, mock_system_data, mock_conformation, mock_system):
    mock_simulation_parameters['traj_format'] = 'arc'
    mock_load_model.return_value = mock_model
    mock_load_system.return_value = (mock_system_data, mock_conformation)
    mock_init_preproc.return_value = ({}, mock_conformation)
    mock_init_dynamics.return_value = (
        lambda i, d, s, c, p, f: (d, s, c, p),
        lambda x: x,
        {"dt": 0.1, "estimate_pressure": False, "thermostat_name": "Langevin", "pimd": False},
        mock_system
    )
    mock_time.side_effect = count(0, 1)

    dynamic(mock_simulation_parameters, "cpu", torch.float32)

    assert mock_write_arc.called

@patch('fennol.md.dynamic.load_model')
@patch('fennol.md.dynamic.load_system_data')
@patch('fennol.md.dynamic.initialize_preprocessing')
@patch('fennol.md.dynamic.initialize_dynamics')
@patch('builtins.open')
@patch('fennol.md.dynamic.write_extxyz_frame')
@patch('time.time')
def test_dynamic_extxyz_format(mock_time, mock_write_extxyz, mock_open, mock_init_dynamics, mock_init_preproc, 
                               mock_load_system, mock_load_model, mock_simulation_parameters, 
                               mock_model, mock_system_data, mock_conformation, mock_system):
    mock_simulation_parameters['traj_format'] = 'extxyz'
    mock_load_model.return_value = mock_model
    mock_load_system.return_value = (mock_system_data, mock_conformation)
    mock_init_preproc.return_value = ({}, mock_conformation)
    mock_init_dynamics.return_value = (
        lambda i, d, s, c, p, f: (d, s, c, p),
        lambda x: x,
        {"dt": 0.1, "estimate_pressure": False, "thermostat_name": "Langevin", "pimd": False},
        mock_system
    )
    mock_time.side_effect = count(0, 1)

    dynamic(mock_simulation_parameters, "cpu", torch.float32)

    assert mock_write_extxyz.called

@patch('fennol.md.dynamic.load_model')
@patch('fennol.md.dynamic.load_system_data')
@patch('fennol.md.dynamic.initialize_preprocessing')
@patch('fennol.md.dynamic.initialize_dynamics')
@patch('builtins.open')
@patch('fennol.md.dynamic.write_xyz_frame')
@patch('time.time')
def test_dynamic_with_pressure_estimation(mock_time, mock_write_xyz, mock_open, mock_init_dynamics, mock_init_preproc, 
                                          mock_load_system, mock_load_model, mock_simulation_parameters, 
                                          mock_model, mock_system_data, mock_conformation, mock_system):
    mock_simulation_parameters['estimate_pressure'] = True
    mock_load_model.return_value = mock_model
    mock_load_system.return_value = (mock_system_data, mock_conformation)
    mock_init_preproc.return_value = ({}, mock_conformation)
    mock_init_dynamics.return_value = (
        lambda i, d, s, c, p, f: (d, s, c, p),
        lambda x: x,
        {"dt": 0.1, "estimate_pressure": True, "thermostat_name": "Langevin", "pimd": False},
        mock_system
    )
    mock_time.side_effect = count(0, 1)

    dynamic(mock_simulation_parameters, "cpu", torch.float64)

    assert mock_model._energy_and_forces_and_virial.called
    assert mock_model._total_energy.call_count == 2  # Called for both ep and em


@patch('fennol.md.dynamic.load_model')
@patch('fennol.md.dynamic.load_system_data')
@patch('fennol.md.dynamic.initialize_preprocessing')
@patch('fennol.md.dynamic.initialize_dynamics')
@patch('builtins.open')
@patch('fennol.md.dynamic.write_xyz_frame')
@patch('time.time')
def test_dynamic_with_pimd(mock_time, mock_write_xyz, mock_open, mock_init_dynamics, mock_init_preproc,
                           mock_load_system, mock_load_model, mock_simulation_parameters,
                           mock_model, mock_system_data, mock_conformation, mock_system):
    mock_system_data['nbeads'] = 4
    mock_system_data['nat'] = 3
    mock_system_data['mass'] = torch.tensor([1.0, 16.0, 1.0])
    mock_system_data['species'] = torch.tensor([1, 8, 1])
    mock_system_data['kT'] = 300 * 8.617333262145e-5
    mock_system['ek_c'] = torch.tensor(0.3)
    mock_conformation['coordinates'] = torch.rand(4, 3, 3)  # Shape: [nbeads, nat, 3]
    mock_system['coordinates'] = mock_conformation['coordinates'].clone()  # Keep the full shape
    mock_load_model.return_value = mock_model
    mock_load_system.return_value = (mock_system_data, mock_conformation)
    mock_init_preproc.return_value = ({}, mock_conformation)
    mock_init_dynamics.return_value = (
        lambda i, d, s, c, p, f: (d, s, c, p),
        lambda x: x,
        {"dt": 0.1, "estimate_pressure": False, "thermostat_name": "Langevin", "pimd": True},
        {**mock_system, 'coordinates': mock_conformation['coordinates'].clone()}
    )
    mock_time.side_effect = count(0, 1)
    mock_simulation_parameters['nsteps'] = 1
    mock_simulation_parameters['dt'] = 0.1
    mock_model._energy_and_forces.return_value = (torch.tensor([0.0]), torch.tensor([[0.0, 0.0, 0.0]]), None)

    dynamic(mock_simulation_parameters, "cpu", torch.float32)


@patch('fennol.md.dynamic.load_model')
@patch('fennol.md.dynamic.load_system_data')
@patch('fennol.md.dynamic.initialize_preprocessing')
@patch('fennol.md.dynamic.initialize_dynamics')
@patch('builtins.open')
@patch('fennol.md.dynamic.write_xyz_frame')
@patch('time.time')
def test_dynamic_with_print_timings(mock_time, mock_write_xyz, mock_open, mock_init_dynamics, mock_init_preproc, 
                                    mock_load_system, mock_load_model, mock_simulation_parameters, 
                                    mock_model, mock_system_data, mock_conformation, mock_system):
    mock_simulation_parameters['print_timings'] = True
    mock_load_model.return_value = mock_model
    mock_load_system.return_value = (mock_system_data, mock_conformation)
    mock_init_preproc.return_value = ({}, mock_conformation)
    mock_init_dynamics.return_value = (
        lambda i, d, s, c, p, f: (d, s, c, p),
        lambda x: x,
        {"dt": 0.1, "estimate_pressure": False, "thermostat_name": "Langevin", "pimd": False},
        mock_system
    )
    mock_time.side_effect = count(0, 1)

    with patch('builtins.print') as mock_print:
        dynamic(mock_simulation_parameters, "cpu", torch.float32)

    assert any("Detailed per-step timings" in str(call) for call in mock_print.call_args_list)

@patch('fennol.md.dynamic.load_model')
@patch('fennol.md.dynamic.load_system_data')
@patch('fennol.md.dynamic.initialize_preprocessing')
@patch('fennol.md.dynamic.initialize_dynamics')
@patch('builtins.open')
@patch('fennol.md.dynamic.write_xyz_frame')
@patch('time.time')
def test_dynamic_with_nan_error(mock_time, mock_write_xyz, mock_open, mock_init_dynamics, mock_init_preproc, 
                                mock_load_system, mock_load_model, mock_simulation_parameters, 
                                mock_model, mock_system_data, mock_conformation, mock_system):
    mock_load_model.return_value = mock_model
    mock_load_system.return_value = (mock_system_data, mock_conformation)
    mock_init_preproc.return_value = ({}, mock_conformation)
    
    def step_func(i, d, s, c, p, f):
        s['vel'] = torch.tensor([[float('nan'), 0.0, 0.0]])
        return d, s, c, p
    
    mock_init_dynamics.return_value = (
        step_func,
        lambda x: x,
        {"dt": 0.1, "estimate_pressure": False, "thermostat_name": "Langevin", "pimd": False},
        mock_system
    )
    mock_time.side_effect = count(0, 1)

    with pytest.raises(ValueError, match="dynamics crashed at step"):
        dynamic(mock_simulation_parameters, "cpu", torch.float32)

@patch('fennol.md.dynamic.load_model')
@patch('fennol.md.dynamic.load_system_data')
@patch('fennol.md.dynamic.initialize_preprocessing')
@patch('fennol.md.dynamic.initialize_dynamics')
@patch('builtins.open')
@patch('fennol.md.dynamic.write_xyz_frame')
@patch('time.time')
def test_dynamic_with_wrap_box(mock_time, mock_write_xyz, mock_open, mock_init_dynamics, mock_init_preproc,
                               mock_load_system, mock_load_model, mock_simulation_parameters,
                               mock_model, mock_system_data, mock_conformation, mock_system):
    mock_simulation_parameters['wrap_box'] = True
    mock_load_model.return_value = mock_model
    mock_load_system.return_value = (mock_system_data, mock_conformation)
    mock_init_preproc.return_value = ({}, mock_conformation)
    mock_init_dynamics.return_value = (
        lambda i, d, s, c, p, f: (d, s, c, p),
        lambda x: x,
        {"dt": 0.1, "estimate_pressure": False, "thermostat_name": "Langevin", "pimd": False},
        mock_system
    )
    mock_time.side_effect = count(0, 1)

    dynamic(mock_simulation_parameters, "cpu", torch.float32)

    assert mock_write_xyz.called
    print("Mock write_xyz_frame calls:")
    for call in mock_write_xyz.call_args_list:
        print(call)

@patch('fennol.md.dynamic.load_model')
@patch('fennol.md.dynamic.load_system_data')
@patch('fennol.md.dynamic.initialize_preprocessing')
@patch('fennol.md.dynamic.initialize_dynamics')
@patch('builtins.open')
@patch('fennol.md.dynamic.write_xyz_frame')
@patch('time.time')
def test_dynamic_with_qtb(mock_time, mock_write_xyz, mock_open, mock_init_dynamics, mock_init_preproc, 
                          mock_load_system, mock_load_model, mock_simulation_parameters, 
                          mock_model, mock_system_data, mock_conformation, mock_system):
    mock_system['thermostat']['corr_kin'] = torch.tensor(1.1)
    mock_load_model.return_value = mock_model
    mock_load_system.return_value = (mock_system_data, mock_conformation)
    mock_init_preproc.return_value = ({}, mock_conformation)
    mock_init_dynamics.return_value = (
        lambda i, d, s, c, p, f: (d, s, c, p),
        lambda x: x,
        {"dt": 0.1, "estimate_pressure": False, "thermostat_name": "QTB", "pimd": False},
        mock_system
    )
    mock_time.side_effect = count(0, 1)

    with patch('builtins.print') as mock_print:
        dynamic(mock_simulation_parameters, "cpu", torch.float32)

    assert any("QTB kin. correction" in str(call) for call in mock_print.call_args_list)


@patch('fennol.md.dynamic.load_model')
@patch('fennol.md.dynamic.load_system_data')
@patch('fennol.md.dynamic.initialize_preprocessing')
def test_dynamic_invalid_traj_format(mock_init_preproc, mock_load_system, mock_load_model, mock_simulation_parameters):
    mock_model = MagicMock()
    mock_model._energy_and_forces.return_value = (torch.tensor([0.0]), torch.tensor([[0.0, 0.0, 0.0]]), None)
    mock_load_model.return_value = mock_model
    mock_load_system.return_value = ({
        'name': 'test_system',
        'nat': 1,
        'symbols': ['H'],
        'pbc': None,
        'mass': torch.tensor([1.0]),
        'species': torch.tensor([1]),
        'kT': 300 * 8.617333262145e-5
    }, {
        'coordinates': torch.tensor([[0.0, 0.0, 0.0]]),
        'species': torch.tensor([1]),
        'cells': None,
        'reciprocal_cells': None
    })
    mock_init_preproc.return_value = ({}, {
        'coordinates': torch.tensor([[0.0, 0.0, 0.0]]),
        'species': torch.tensor([1]),
        'cells': None,
        'reciprocal_cells': None
    })
    mock_simulation_parameters['traj_format'] = 'invalid_format'
    mock_simulation_parameters['dt'] = 0.1
    with pytest.raises(ValueError, match="Unknown trajectory format"):
        dynamic(mock_simulation_parameters, "cpu", torch.float32)


@patch('fennol.md.dynamic.load_model')
@patch('fennol.md.dynamic.load_system_data')
@patch('fennol.md.dynamic.initialize_preprocessing')
def test_dynamic_zero_steps(mock_init_preproc, mock_load_system, mock_load_model, mock_simulation_parameters):
    mock_model = MagicMock()
    mock_model._energy_and_forces.return_value = (torch.tensor([0.0]), torch.tensor([[0.0, 0.0, 0.0]]), None)
    mock_load_model.return_value = mock_model
    mock_load_system.return_value = ({
        'name': 'test_system',
        'nat': 1,
        'symbols': ['H'],
        'pbc': None,
        'mass': torch.tensor([1.0]),
        'species': torch.tensor([1]),
        'kT': 300 * 8.617333262145e-5
    }, {
        'coordinates': torch.tensor([[0.0, 0.0, 0.0]]),
        'species': torch.tensor([1]),
        'cells': None,
        'reciprocal_cells': None
    })
    mock_init_preproc.return_value = ({}, {
        'coordinates': torch.tensor([[0.0, 0.0, 0.0]]),
        'species': torch.tensor([1]),
        'cells': None,
        'reciprocal_cells': None
    })
    mock_simulation_parameters['nsteps'] = 0
    mock_simulation_parameters['dt'] = 0.1
    dynamic(mock_simulation_parameters, "cpu", torch.float32)



@patch('fennol.md.dynamic.load_model')
@patch('fennol.md.dynamic.load_system_data')
@patch('fennol.md.dynamic.initialize_preprocessing')
@patch('fennol.md.dynamic.initialize_dynamics')
@patch('builtins.open')
@patch('fennol.md.dynamic.write_xyz_frame')
@patch('time.time')
def test_dynamic_single_atom(mock_time, mock_write_xyz, mock_open, mock_init_dynamics, mock_init_preproc, 
                             mock_load_system, mock_load_model, mock_simulation_parameters, 
                             mock_model, mock_system_data, mock_conformation, mock_system):
    # Set up your mocks as before
    mock_system_data['nat'] = 1
    mock_system_data['symbols'] = ['H']
    mock_conformation['coordinates'] = torch.tensor([[0.0, 0.0, 0.0]])
    mock_system['coordinates'] = torch.tensor([[0.0, 0.0, 0.0]])
    mock_system['vel'] = torch.tensor([[0.1, 0.1, 0.1]])
    
    # Mock the return values
    mock_load_model.return_value = mock_model
    mock_load_system.return_value = (mock_system_data, mock_conformation)
    mock_init_preproc.return_value = ({}, mock_conformation)
    mock_init_dynamics.return_value = (
        lambda i, d, s, c, p, f: (d, s, c, p),
        lambda x: x,
        {"dt": 0.1, "estimate_pressure": False, "thermostat_name": "Langevin", "pimd": False},
        mock_system
    )
    mock_time.side_effect = count(0, 1)

    # Call the function under test
    dynamic(mock_simulation_parameters, "cpu", torch.float32)

    # Assert that the necessary functions were called
    assert mock_load_model.called
    assert mock_load_system.called
    assert mock_init_preproc.called
    assert mock_init_dynamics.called
    assert mock_write_xyz.called

if __name__ == "__main__":
    pytest.main()