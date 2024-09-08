import pathlib
from enum import Enum
from typing import List, Union, Literal

import pydantic as pd
import yaml

class Geometry(pd.BaseModel):
    dimension: int = pd.Field(gt=0, le=3)
    lb: Union[List[float], float]
    ub: Union[List[float], float]
    n: Union[List[int], int]

    # runs through the inputs 1 by 1
    @pd.field_validator('lb', 'ub', 'n')
    def bounds(cls, field_value, info: pd.ValidationInfo):
        # print(f"{cls}: field_value = {field_value}")
        # print(f"values = {info}")
        if info.data['dimension'] == 1:
            if isinstance(field_value, float) or \
               isinstance(field_value, int):
                field_value = [field_value]
        if len(field_value) != info.data['dimension']:
            raise ValueError(
                (f"Bounds must be a list with a number"
                 f" of elements equal to the dimension"))

        return field_value


class Angles(pd.BaseModel):
    n_theta: int = pd.Field(gt=0)
    n_psi: int = pd.Field(gt=0)


class Stencil(Enum):
    upwind = 'upwind'
    rusanov = 'rusanov'
    jiang = 'jiang'

class SolverMethod(Enum):

    forward_euler = 'forward euler'
    backward_euler = 'backward euler'

class Solver(pd.BaseModel):

    cfl: float = pd.Field(gt=0.0)
    stencil: Stencil
    method: SolverMethod
    num_steps: int = pd.Field(gt=0)
    round_tol: float = pd.Field(gt=0.0)
    round_freq: int

    class Config:
        use_enum_values = True

class SourceChoices(Enum):

    no_source = 'none'
    iso_scatter = 'isotropic scattering'

class Equations(pd.BaseModel):

    source: SourceChoices

    class Config:
        use_enum_values = True

class Saving(pd.BaseModel):

    directory: str
    plot_freq: int

class Config(pd.BaseModel):
    """CLI config class."""
    problem: Literal['hohlraum']
    geometry: Geometry
    angles: Angles
    solver: Solver
    equations: Equations
    saving: Saving
    # add validation


def load_yml_config(path:pathlib.Path, logger):
    """Return Config."""
    try:
        return yaml.safe_load(path.read_text())
    except FileNotFoundError as error:
        message = "Error.yml config file not found."
        logger.exception(message)
        raise FileNotFoundError(error, message) from error
