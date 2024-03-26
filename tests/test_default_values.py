import numpy as np
import pyroll.ring_model_thermal

from pyroll.core import CoolingPipe, Transport, Roll, RollPass, CircularOvalGroove


def test_roll_pass_default_htc():
    u = RollPass(
        label="Test Oval",
        roll=Roll(
            groove=CircularOvalGroove(
                depth=8e-3,
                r1=6e-3,
                r2=40e-3
            ),
        )
    )
    assert np.isclose(u.heat_transfer_coefficient, 150)
    assert np.isclose(u.roll.heat_transfer_coefficient, 6000)


def test_transport_default_htc():
    u = Transport()
    assert np.isclose(u.heat_transfer_coefficient, 15)


def test_cooling_pipe_default_htc():
    u = CoolingPipe()
    assert np.isclose(u.heat_transfer_coefficient, 4000)
