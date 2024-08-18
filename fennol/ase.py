import ase
import ase.calculators.calculator
import ase.units
import numpy as np
import torch
from . import FENNIX
from .models.preprocessing import convert_to_torch
from typing import Sequence, Union, Optional
from ase.stress import full_3x3_to_voigt_6_stress

class FENNIXCalculator(ase.calculators.calculator.Calculator):
    """
    ASE calculator implementation for FENNIX machine learning potential.

    This calculator interfaces the FENNIX model with ASE, allowing for
    energy, force, and stress calculations within the ASE framework.

    Attributes:
        implemented_properties (list): List of properties that this calculator can compute.

    Args:
        model (Union[str, FENNIX]): Either a path to a FENNIX model file or a FENNIX instance.
        gpu_preprocessing (bool): Whether to use GPU for preprocessing. Default is False.
        atoms (Optional[ase.Atoms]): Initial atomic configuration. Default is None.
        verbose (bool): Whether to print verbose output. Default is False.
        energy_terms (Optional[Sequence[str]]): Specific energy terms to compute. Default is None.
        use_float64 (bool): Whether to use double precision. Default is False.
        **kwargs: Additional keyword arguments to pass to FENNIX.load() if model is a string.
    """


    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        model: Union[str, FENNIX],
        gpu_preprocessing: bool = False,
        atoms: Optional[ase.Atoms] = None,
        verbose: bool = False,
        energy_terms: Optional[Sequence[str]] = None,
        use_float64: bool = False,
        **kwargs
    ):
        super().__init__()
        if use_float64:
            torch.set_default_dtype(torch.float64)
        
        if isinstance(model, str):
            self.model = FENNIX.load(model, **kwargs)
        elif hasattr(model, '_energy_and_forces_and_virial') and hasattr(model, '_energy_and_forces') and hasattr(model, '_total_energy'):
            # This checks for FENNIX-like interface without using isinstance
            self.model = model
        else:
            raise ValueError("model must be either a string path or a FENNIX instance")
        
        if energy_terms is not None:
            self.model.set_energy_terms(energy_terms)
        self.dtype = "float64" if use_float64 else "float32"
        self.gpu_preprocessing = gpu_preprocessing
        self.verbose = verbose
        self._fennol_inputs = None
        if atoms is not None:
            self.preprocess(atoms)

    def calculate(
        self,
        atoms=None,
        properties=["energy"],
        system_changes=ase.calculators.calculator.all_changes,
    ):
        """
        Perform the calculation of properties for the given atomic configuration.

        This method is called by ASE to compute the requested properties. It handles
        energy, forces, and stress calculations based on the input properties.

        Args:
            atoms (ase.Atoms): Atomic configuration. If None, uses the stored configuration.
            properties (list): List of properties to calculate. Default is ["energy"].
            system_changes (list): List of changes since last calculation. Default is all changes.

        Note:
            The results are stored in self.results dictionary, following ASE conventions.
        """
        super().calculate(atoms, properties, system_changes)
        inputs = self.preprocess(self.atoms, system_changes=system_changes)

        if "stress" in properties:
            e, f, virial, output = self.model._energy_and_forces_and_virial(
                self.model.variables, inputs
            )
            volume = self.atoms.get_volume()
            stress = -np.asarray(virial[0]) * ase.units.Hartree / volume
            self.results["stress"] = full_3x3_to_voigt_6_stress(stress)
            self.results["forces"] = np.asarray(f) * ase.units.Hartree
        elif "forces" in properties:
            e, f, output = self.model._energy_and_forces(self.model.variables, inputs)
            self.results["forces"] = np.asarray(f) * ase.units.Hartree
        else:
            e, output = self.model._total_energy(self.model.variables, inputs)

        self.results["energy"] = float(e[0]) * ase.units.Hartree

    def preprocess(self, atoms, system_changes=ase.calculators.calculator.all_changes):
        """
        Preprocess the atomic configuration for FENNIX calculations.

        This method converts ASE Atoms object into the input format required by FENNIX.
        It handles periodic boundary conditions and updates only the necessary parts
        of the input based on the system changes.

        Args:
            atoms (ase.Atoms): Atomic configuration to preprocess.
            system_changes (list): List of changes since last calculation. Default is all changes.

        Returns:
            dict: Preprocessed inputs for FENNIX calculations.

        Note:
            This method updates the internal _fennol_inputs attribute.
        """
        force_cpu_preprocessing = False
        if self._fennol_inputs is None:
            force_cpu_preprocessing = True
            cell = np.asarray(atoms.get_cell(complete=True).array, dtype=self.dtype)
            pbc = np.asarray(atoms.get_pbc(), dtype=bool)
            if np.all(pbc):
                use_pbc = True
            elif np.any(pbc):
                raise NotImplementedError("PBC should be activated in all directions.")
            else:
                use_pbc = False

            species = np.asarray(atoms.get_atomic_numbers(), dtype=np.int32)
            coordinates = np.asarray(atoms.get_positions(), dtype=self.dtype)
            natoms = np.array([len(species)], dtype=np.int32)
            batch_index = np.array([0] * len(species), dtype=np.int32)

            inputs = {
                "species": species,
                "coordinates": coordinates,
                "natoms": natoms,
                "batch_index": batch_index,
            }
            if use_pbc:
                reciprocal_cell = np.linalg.inv(cell)
                inputs["cells"] = cell.reshape(1, 3, 3)
                inputs["reciprocal_cells"] = reciprocal_cell.reshape(1, 3, 3)
            self._fennol_inputs = convert_to_torch(inputs)
        else:
            if "cell" in system_changes:
                pbc = np.asarray(atoms.get_pbc(), dtype=bool)
                if np.all(pbc):
                    use_pbc = True
                elif np.any(pbc):
                    raise NotImplementedError(
                        "PBC should be activated in all directions."
                    )
                else:
                    use_pbc = False
                if use_pbc:
                    cell = np.asarray(
                        atoms.get_cell(complete=True).array, dtype=self.dtype
                    )
                    reciprocal_cell = np.linalg.inv(cell)
                    self._fennol_inputs["cells"] = torch.tensor(cell.reshape(1, 3, 3))
                    self._fennol_inputs["reciprocal_cells"] = torch.tensor(
                        reciprocal_cell.reshape(1, 3, 3)
                    )
                elif "cells" in self._fennol_inputs:
                    del self._fennol_inputs["cells"]
                    del self._fennol_inputs["reciprocal_cells"]
            if "numbers" in system_changes:
                self._fennol_inputs["species"] = torch.tensor(
                    atoms.get_atomic_numbers(), dtype=torch.int32
                )
                self._fennol_inputs["natoms"] = torch.tensor(
                    [len(self._fennol_inputs["species"])], dtype=torch.int32
                )
                self._fennol_inputs["batch_index"] = torch.tensor(
                    [0] * len(self._fennol_inputs["species"]), dtype=torch.int32
                )
                force_cpu_preprocessing = True
            if "positions" in system_changes:
                self._fennol_inputs["coordinates"] = torch.tensor(
                    atoms.get_positions(), dtype=torch.as_tensor(1.0).dtype
                )
        
        if self.gpu_preprocessing:
            self._fennol_inputs = self.model.preprocessing.process(
                self.model.preproc_state, self._fennol_inputs
            )
            self.model.preproc_state, state_up, self._fennol_inputs, overflow = (
                self.model.preprocessing.check_reallocate(
                    self.model.preproc_state, self._fennol_inputs
                )
            )
            if overflow and self.verbose:
                print("FENNIX nblist overflow => reallocating nblist")
                print(f"size updates: {state_up}")
        else:
            self._fennol_inputs = self.model.preprocess(**self._fennol_inputs)

        return self._fennol_inputs
