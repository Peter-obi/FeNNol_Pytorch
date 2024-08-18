import pytest
import numpy as np
import ase
import ase.units
from unittest.mock import Mock, patch, MagicMock
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch

from fennol.ase import FENNIXCalculator

@pytest.fixture
def mock_fennix():
    mock = Mock()
    mock._energy_and_forces_and_virial.return_value = ([1.0], [[0.1, 0.1, 0.1]], [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], {})
    mock._energy_and_forces.return_value = ([1.0], [[0.1, 0.1, 0.1]], {})
    mock._total_energy.return_value = ([1.0], {})
    mock.preprocess.return_value = {"preprocessed": "data"}
    return mock

@pytest.fixture
def mock_atoms():
    return ase.Atoms('H2O', positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]], cell=[5.0, 5.0, 5.0], pbc=True)

@patch('fennol.ase.FENNIX')
def test_fennix_calculator_initialization(mock_fennix_class):
    mock_model = Mock()
    mock_fennix_class.load.return_value = mock_model
    calc = FENNIXCalculator("path/to/model")
    assert isinstance(calc, FENNIXCalculator)
    assert calc.implemented_properties == ["energy", "forces", "stress"]
    mock_fennix_class.load.assert_called_once_with("path/to/model")

def test_fennix_calculator_with_fennix_instance(mock_fennix):
    calc = FENNIXCalculator(mock_fennix)
    assert calc.model == mock_fennix

@patch('fennol.ase.FENNIX')
def test_fennix_calculator_with_energy_terms(mock_fennix_class):
    mock_model = Mock()
    mock_fennix_class.load.return_value = mock_model
    energy_terms = ["term1", "term2"]
    calc = FENNIXCalculator("path/to/model", energy_terms=energy_terms)
    mock_model.set_energy_terms.assert_called_once_with(energy_terms)

@patch('fennol.ase.FENNIX')
@patch('fennol.ase.torch')
def test_fennix_calculator_with_use_float64(mock_torch, mock_fennix_class):
    FENNIXCalculator("path/to/model", use_float64=True)
    mock_torch.set_default_dtype.assert_called_once_with(mock_torch.float64)

@patch('fennol.ase.FENNIX')
@patch('fennol.ase.convert_to_torch')
def test_fennix_calculator_preprocess(mock_convert_to_torch, mock_fennix_class, mock_atoms):
    mock_model = Mock()
    mock_fennix_class.load.return_value = mock_model
    calc = FENNIXCalculator("path/to/model")
    calc.preprocess(mock_atoms)
    mock_convert_to_torch.assert_called_once()
    inputs = mock_convert_to_torch.call_args[0][0]
    assert "species" in inputs
    assert "coordinates" in inputs
    assert "natoms" in inputs
    assert "batch_index" in inputs

@patch('fennol.ase.FENNIX')
def test_fennix_calculator_preprocess_with_pbc(mock_fennix_class, mock_atoms):
    mock_model = Mock()
    mock_fennix_class.load.return_value = mock_model
    mock_model.preprocess.return_value = {"cells": torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])}
    calc = FENNIXCalculator("path/to/model")
    calc.preprocess(mock_atoms)
    assert calc._fennol_inputs is not None
    assert "cells" in calc._fennol_inputs

@patch('fennol.ase.FENNIX')
def test_fennix_calculator_preprocess_with_partial_pbc(mock_fennix_class, mock_atoms):
    mock_model = Mock()
    mock_fennix_class.load.return_value = mock_model
    mock_atoms.set_pbc([True, False, False])
    calc = FENNIXCalculator("path/to/model")
    with pytest.raises(NotImplementedError):
        calc.preprocess(mock_atoms)

@patch('fennol.ase.FENNIX')
def test_fennix_calculator_calculate_energy(mock_fennix_class, mock_atoms):
    mock_model = Mock()
    mock_fennix_class.load.return_value = mock_model
    mock_model._total_energy.return_value = ([1.0], {})
    calc = FENNIXCalculator("path/to/model")
    calc.calculate(mock_atoms, properties=["energy"])
    assert "energy" in calc.results
    assert isinstance(calc.results["energy"], float)
    assert calc.results["energy"] == pytest.approx(1.0 * ase.units.Hartree)

@patch('fennol.ase.FENNIX')
def test_fennix_calculator_calculate_forces(mock_fennix_class, mock_atoms):
    mock_model = Mock()
    mock_fennix_class.load.return_value = mock_model
    mock_model._energy_and_forces.return_value = ([1.0], [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]], {})
    calc = FENNIXCalculator("path/to/model")
    calc.calculate(mock_atoms, properties=["energy", "forces"])
    assert "energy" in calc.results
    assert "forces" in calc.results
    assert isinstance(calc.results["forces"], np.ndarray)
    assert calc.results["forces"].shape == (3, 3)  # 3 atoms, 3 dimensions

@patch('fennol.ase.FENNIX')
def test_fennix_calculator_calculate_stress(mock_fennix_class, mock_atoms):
    mock_model = Mock()
    mock_fennix_class.load.return_value = mock_model
    mock_model._energy_and_forces_and_virial.return_value = ([1.0], [[0.1, 0.1, 0.1]], [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], {})
    calc = FENNIXCalculator("path/to/model")
    calc.calculate(mock_atoms, properties=["energy", "forces", "stress"])
    assert "energy" in calc.results
    assert "forces" in calc.results
    assert "stress" in calc.results
    assert isinstance(calc.results["stress"], np.ndarray)
    assert calc.results["stress"].shape == (6,)  # Voigt notation

@patch('fennol.ase.FENNIX')
def test_fennix_calculator_system_changes(mock_fennix_class, mock_atoms):
    mock_model = Mock()
    mock_fennix_class.load.return_value = mock_model
    mock_model.preprocess.return_value = {"cells": torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])}
    mock_model._total_energy.return_value = ([1.0], {})  # Mock the return value
    calc = FENNIXCalculator("path/to/model")
    calc.preprocess(mock_atoms)
    
    # Test changes in cell
    mock_atoms.set_cell([6, 6, 6])
    calc.calculate(mock_atoms, properties=["energy"], system_changes=["cell"])
    assert "cells" in calc._fennol_inputs
    assert "reciprocal_cells" in calc._fennol_inputs


@patch('fennol.ase.FENNIX')
@patch('fennol.ase.convert_to_torch')
def test_fennix_calculator_gpu_preprocessing(mock_convert_to_torch, mock_fennix_class, mock_atoms):
    mock_model = Mock()
    mock_fennix_class.load.return_value = mock_model
    mock_model.preprocessing = MagicMock()
    mock_model.preprocessing.process.return_value = {"processed": "data"}
    mock_model.preprocessing.check_reallocate.return_value = (None, None, {"processed": "data"}, False)

    mock_convert_to_torch.return_value = {"some": "data"}

    calc = FENNIXCalculator("path/to/model", gpu_preprocessing=True)
    calc._fennol_inputs = {"some": "data"}  # Simulate previous preprocessing
    calc.preprocess(mock_atoms)

    mock_model.preprocessing.process.assert_called_once()
    mock_model.preprocessing.check_reallocate.assert_called_once()

@patch('fennol.ase.FENNIX')
@patch('fennol.ase.convert_to_torch')
def test_fennix_calculator_verbose_mode(mock_convert_to_torch, mock_fennix_class, mock_atoms, capsys):
    mock_model = Mock()
    mock_fennix_class.load.return_value = mock_model
    mock_model.preprocessing = MagicMock()
    mock_model.preprocessing.process.return_value = {"processed": "data"}
    mock_model.preprocessing.check_reallocate.return_value = (None, {"state": "update"}, {"processed": "data"}, True)

    mock_convert_to_torch.return_value = {"some": "data"}

    calc = FENNIXCalculator("path/to/model", gpu_preprocessing=True, verbose=True)
    calc._fennol_inputs = {"some": "data"}  # Simulate previous preprocessing
    calc.preprocess(mock_atoms)

    captured = capsys.readouterr()
    assert "FENNIX nblist overflow => reallocating nblist" in captured.out
    assert "size updates: {'state': 'update'}" in captured.out