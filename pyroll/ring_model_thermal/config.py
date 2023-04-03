from pyroll.core import config


@config("PYROLL_RING_MODEL_THERMAL")
class Config:
    RADIATION_COEFFICIENT = 5.670374419e-8
    """The Stefan-Boltzmann radiation constant. Only to change if not working with SI-units."""

    ROLL_PASS_ATMOSPHERE_TRANSFER = True
    """Whether to include the heat transfer to atmosphere at free surfaces in roll passes into the calculation."""
