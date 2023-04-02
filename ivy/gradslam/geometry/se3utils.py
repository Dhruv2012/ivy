"""
Operations over the Lie Group SE(3), for rigid-body transformations in 3D
"""

# global
from typing import Union

# local
import ivy
from ivy.func_wrapper import (
    handle_array_function,
    inputs_to_ivy_arrays,
    to_native_arrays_and_back,
    handle_out_argument,
    handle_nestable,
    handle_array_like_without_promotion,
    to_ivy_arrays_and_back,
)
from ivy.utils.exceptions import handle_exceptions


# Threshold to determine if a quantity can be considered 'small'
_eps = 1e-6


@to_ivy_arrays_and_back
@handle_nestable
@handle_exceptions
@handle_array_function
@handle_array_like_without_promotion
def so3_hat(
    omega: Union[ivy.Array, ivy.NativeArray]
) -> ivy.Array:
    """Implements the hat operator for SO(3), given an input axis-angle
    vector omega.
    """

    assert ivy.is_array(omega)
    omega_hat = ivy.astype(ivy.zeros((3, 3)), ivy.dtype(omega))
    omega_hat = ivy.to_device(omega_hat, omega.device)
    omega_hat[0, 1] = -omega[2]
    omega_hat[1, 0] = omega[2]
    omega_hat[0, 2] = omega[1]
    omega_hat[2, 0] = -omega[1]
    omega_hat[1, 2] = -omega[0]
    omega_hat[2, 1] = omega[0]

    return omega_hat


@to_ivy_arrays_and_back
@handle_nestable
@handle_exceptions
@handle_array_function
@handle_array_like_without_promotion
def se3_hat(
    xi: Union[ivy.Array, ivy.NativeArray]
) -> ivy.Array:
    """Implements the SE(3) hat operator, given a vector of twist
    (exponential) coordinates.
    """

    assert ivy.is_array(xi)
    v = xi[:3]
    omega = xi[3:]
    omega_hat = so3_hat(omega)

    xi_hat = ivy.astype(ivy.zeros((4, 4)), ivy.dtype(xi))
    xi_hat = ivy.to_device(xi_hat, xi.device)
    xi_hat[:3, :3] = omega_hat
    xi_hat[:3, 3] = v

    return xi_hat


@to_ivy_arrays_and_back
@handle_nestable
@handle_exceptions
@handle_array_function
@handle_array_like_without_promotion
def so3_exp(
    omega: Union[ivy.Array, ivy.NativeArray]
) -> ivy.Array:
    """Computes the exponential map for the coordinate-vector omega.
    Returns a 3 x 3 SO(3) matrix.
    """

    assert ivy.is_array(omega)

    omega_hat = so3_hat(omega)
    theta = ivy.norm(omega)

    if theta < _eps:
        R = ivy.eye(3, 3, dtype=ivy.dtype(omega), device=omega.device) + omega_hat
    else:
        s = ivy.sin(theta)
        c = ivy.cos(theta)
        omega_hat_sq = ivy.matmul(omega_hat, omega_hat)
        # Coefficients of the Rodrigues formula
        A = s / theta
        B = (1 - c) / ivy.pow(theta, 2)
        C = (theta - s) / ivy.pow(theta, 3)

        R = (
            ivy.eye(3, 3, dtype=ivy.dtype(omega), device=omega.device)
            + A * omega_hat
            + B * omega_hat_sq
        )
    
    return R


@to_ivy_arrays_and_back
@handle_nestable
@handle_exceptions
@handle_array_function
@handle_array_like_without_promotion
def se3_exp(
    xi: Union[ivy.Array, ivy.NativeArray]
) -> ivy.Array:
    """Computes the exponential map for the coordinate-vector xi.
    Returns a 4 x 4 SE(3) matrix.
    """

    assert ivy.is_array(xi)

    v = xi[:3]
    omega = xi[3:]
    omega_hat = so3_hat(omega)

    theta = ivy.norm(omega)

    if theta < _eps:
        R = ivy.eye(3, 3, dtype=ivy.dtype(omega), device=omega.device) + omega_hat
        V = ivy.eye(3, 3, dtype=ivy.dtype(omega), device=omega.device) + omega_hat
    else:
        s = ivy.sin(theta)
        c = ivy.cos(theta)
        omega_hat_sq = ivy.matmul(omega_hat, omega_hat)
        # Coefficients of the Rodrigues formula
        A = s / theta
        B = (1 - c) / ivy.pow(theta, 2)
        C = (theta - s) / ivy.pow(theta, 3)
        R = (
            ivy.eye(3, 3, dtype=ivy.dtype(omega), device=omega.device)
            + A * omega_hat
            + B * omega_hat_sq
        )
        V = (
            ivy.eye(3, 3, dtype=ivy.dtype(omega), device=omega.device)
            + B * omega_hat
            + C * omega_hat_sq
        )
    
    t = ivy.matmul(V, ivy.reshape(v, (3, 1)))
    last_row = ivy.array([0, 0, 0, 1], dtype=ivy.dtype(xi), device=xi.device)

    return ivy.concat([ivy.concat([R, t], axis=1), ivy.expand_dims(last_row, axis=0)], axis=0)