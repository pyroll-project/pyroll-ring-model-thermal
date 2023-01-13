from pyroll.core import RollPass

# def get_increments(unit: Unit, transport: TransportExt) -> Tuple[np.ndarray, np.ndarray]:
#     p = unit.in_profile
#     capacity = p.density * p.thermal_capacity * unit.volume
#
#     dr = p.equivalent_radius / RADIAL_DISCRETIZATION_COUNT
#     radii = p.temperature_profile[0]
#     temperatures = p.temperature_profile[1]
#     increments = np.zeros_like(temperatures[1])
#
#     source = 0
#
#     increments[0] = unit.duration / capacity * (
#             np.pi * p.thermal_conductivity * unit.length * (temperatures[1] - temperatures[0])
#             + source
#     )
#
#     increments[-1] = unit.duration / capacity * (
#             2 * np.pi * unit.length
#             * (
#                     (
#                             -transport.heat_transfer_factor
#                             * (transport.environment_temperature - p.surface_temperature)
#                             - RADIATION_COEFFICIENT * transport.relative_radiation_coefficient
#                             * (transport.environment_temperature ** 4 - p.surface_temperature ** 4)
#                     )
#                     * (radii[-1] + dr / 2)
#                     - p.thermal_conductivity * (temperatures[-1] - temperatures[-2]) / dr * (radii[-1] - dr / 2)
#             )
#             + source
#     )
#
#     for i in range(1, len(increments) - 1):
#         increments[i] = unit.duration / capacity * (
#                 2 * np.pi * unit.length * p.thermal_conductivity / dr
#                 * (
#                         (temperatures[i + 1] - temperatures[i]) * (radii[i] + dr / 2)
#                         - (temperatures[i] - temperatures[i - 1]) * (radii[i] - dr / 2)
#                 )
#                 + source
#         )
#
#     return increments