from pyroll.core import CoolingPipe, Unit, Hook, root_hooks


@CoolingPipe.extension_class
class CoolingPipeExt(CoolingPipe):
    heat_transfer_coefficient = Hook[float]()
    """Heat transfer coefficient by convection to atmosphere."""


@CoolingPipeExt.heat_transfer_coefficient
def heat_transfer_coefficient(self: CoolingPipe):
    """Default value from measurements by H. Wehage; (Beitrag zur rechnergestützten Erarbeitung von Projekten und
    Technologien für kontinuierliche Feinstahl- und Drahtstraßen, PhD, TU Freiberg, 1990)"""
    return 4000
