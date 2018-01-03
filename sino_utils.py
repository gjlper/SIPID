"""
Functions related to sinogram preprocessing
"""
#pylint: disable=C0103, C0301
import odl
import numpy as np

mu_water = 0.02
photons_per_pixel = 10000.0
epsilon = 1.0/photons_per_pixel

size = 512
space = odl.uniform_discr([-128, -128], [128, 128], [size, size], dtype='float32', weighting='const')

def get_op(angle, det=800, src_radius=500, det_radius=500):
    """
    Get the Fan beam projection operator
    Args:
        angle: the angle of rotations
    Returns:
        operator: projection operator that projects a phantom to sinogram.
        pseudoinverse: Filtered back projection of the given geometry.
    """
    angle_partition = odl.uniform_partition(0, 2*np.pi, angle)
    detector_partition = odl.uniform_partition(-360, 360, det)
    geometry = odl.tomo.FanFlatGeometry(angle_partition, detector_partition, src_radius=src_radius, det_radius=det_radius)

    operator = odl.tomo.RayTransform(space, geometry)
    pseudoinverse = odl.tomo.fbp_op(operator)
    #nonlinear_operator = odl.ufunc_ops.exp(operator.range)*(-mu_water * operator)
    return operator, pseudoinverse
