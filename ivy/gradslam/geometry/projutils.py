"""
Projective geometry utility functions.
"""

from typing import Union, Optional

import ivy
from ivy.func_wrapper import (
    handle_array_function,
    handle_nestable,
    handle_array_like_without_promotion,
    to_ivy_arrays_and_back,
)
from ivy.utils.exceptions import handle_exceptions


def _handle_padding_shape(padding, n, mode):
    padding = tuple(
        [
            (padding[i * 2], padding[i * 2 + 1])
            for i in range(int(len(padding) / 2) - 1, -1, -1)
        ]
    )
    while len(padding) < n:
        if mode == "circular":
            padding = padding + ((0, 0),)
        else:
            padding = ((0, 0),) + padding
    if mode == "circular":
        padding = tuple(list(padding)[::-1])
    return padding


@to_ivy_arrays_and_back
@handle_nestable
@handle_exceptions
@handle_array_function
@handle_array_like_without_promotion
def homogenize_points(
    pts: Union[ivy.Array, ivy.NativeArray]
)   -> ivy.Array:
    """
   Convert a set of points to homogeneous coordinates.

    Parameters
    ----------
    pts
        Array of shape (..., 3) representing a set of 3D points to be homogenized.
    
    Returns
    -------
    ret
        Array of shape (..., 4) representing Homogeneous coordinates of pts.
    
    Shape:
        - pts: :math:`(N, *, K)` where :math:`N` indicates the number of points in a cloud if
          the shape is :math:`(N, K)` and indicates batchsize if the number of dimensions is greater than 2.
          Also, :math:`*` means any number of additional dimensions, and `K` is the dimensionality of each point.
        - Output: :math:`(N, *, K + 1)` where all but the last dimension are the same shape as `pts`.
    
    Examples::
        >>> pts = ivy.random_uniform(shape(10, 3))
        >>> pts_homo = homogenize_points(pts)
        >>> pts_homo.shape
            (10, 4)   
    """

    assert ivy.is_array(pts), TypeError(
        "Expected input pts to be of type ivy.Array or ivy.NativeArray. Got {} instead".format(type(pts))
    )

    assert pts.ndim >= 2, ValueError(
        "Input pts must have at least 2 dimensions. Got {} instad.".format(
            pts.ndim
        )
    )

    pad = _handle_padding_shape((0,1), pts.ndim, "constant")
    return ivy.pad(pts, pad, mode="constant", constant_values=1.0)


@to_ivy_arrays_and_back
@handle_nestable
@handle_exceptions
@handle_array_function
@handle_array_like_without_promotion
def unhomogenize_points(
    pts: Union[ivy.Array, ivy.NativeArray],
    eps: float = 1e-6
)   -> ivy.Array:
    """
    Convert a set of points from homogeneous coordinates to Euclidean
    coordinates. This is usually done by taking each point :math:`(X, Y, Z, W)` and dividing it by
    the last coordinate :math:`(w)`.

    Parameters
    ----------
    pts
        Array representing a set of 3D points to be unhomogenized.

    Returns
    -------
    ret
        Array: Unhomogenized points.
    """

    assert ivy.is_array(pts), TypeError(
        "Expected input pts to be of type ivy.Array or ivy.NativeArray. Got {} instead".format(type(pts))
    )

    assert pts.ndim >= 2, ValueError(
        "Input pts must have at least 2 dimensions. Got {} instad.".format(pts.ndim)
    )

    # Get points with the last coordinate (scale) as 0 (points at infinity).
    w = pts[..., -1:]
    # Determine the scale factor each point needs to be multiplied by
    # For points at infinity, use a scale factor of 1 (used by OpenCV
    # and by kornia)
    # https://github.com/opencv/opencv/pull/14411/files
    scale = ivy.where(ivy.abs(w) > eps, 1.0 / w, ivy.ones_like(w))

    return pts[..., :-1] * scale


@to_ivy_arrays_and_back
@handle_nestable
@handle_exceptions
@handle_array_function
@handle_array_like_without_promotion
def project_points(
    cam_coords: Union[ivy.Array, ivy.NativeArray],
    proj_mat: Union[ivy.Array, ivy.NativeArray],
    eps: Optional[float] = 1e-6
) -> ivy.Array:
    r"""Projects points from the camera coordinate frame to the image (pixel) frame.
    
    Parameters
    ----------
    cam_coords: 
        pixel coordinates (defined in the frame of the first camera).
    proj_mat: 
        projection matrix between the reference and the non-reference camera frame.

    Returns
    ----------
    ret:
        Image (pixel) coordinates corresponding to the input 3D points.

    Shapes:
    - cam_coords: :math:`(N, *, 3)` or :math:`(*, 4)` where :math:`*` indicates an arbitrary number of dimensions.
        Here :math:`N` indicates the number of points in a cloud if the shape is :math:`(N, 3)` and indicates
        batchsize if the number of dimensions is greater than 2.
    - proj_mat: :math:`(*, 4, 4)` where :math:`*` indicates an arbitrary number of dimensions.
        dimension contains a :math:`(4, 4)` camera projection matrix.
    - Output: :math:`(N, *, 2)`, where :math:`*` indicates the same dimensions as in `cam_coords`.
        Here :math:`N` indicates the number of points in a cloud if the shape is :math:`(N, 3)` and indicates
        batchsize if the number of dimensions is greater than 2.
    """    
    # Based on
    # https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py#L43
    # and Kornia.
    assert ivy.is_array(cam_coords), TypeError(
        "Expected input cam_coords to be of type ivy.Array or ivy.NativeArray. Got {} instead".format(
            type(cam_coords)
        )
    )

    assert ivy.is_array(proj_mat), TypeError(
        "Expected input proj_mat to be of type ivy.Array or ivy.NativeArray. Got {} instead".format(
            type(proj_mat)
        )
    )

    assert cam_coords.ndim >= 2, ValueError(
        "Input cam_coords must have at least 2 dimensions. Got {} instad.".format(
            cam_coords.ndim
        )
    )

    assert cam_coords.shape[-1] in (3, 4), ValueError(
        "Input cam_coords must have last dimension of shape (*, 3) or (*, 4). Got {} instad.".format(
            cam_coords.shape
        )
    )

    assert proj_mat.ndim >= 2, ValueError(
        "Input proj_mat must have at least 2 dimensions. Got {} instad.".format(proj_mat.ndim)
    )

    assert proj_mat.shape[-1] == 4 and proj_mat[-2] == 4, ValueError(
        "Input proj_mat must have shape (*, 4, 4). Got {} instad.".format(
            proj_mat.shape
        )
    )

    assert proj_mat.ndim == 2 and proj_mat.ndim == cam_coords.ndim, ValueError(
        "Input proj_mat must either have 2 dimensions, or have equal number of dimensions to cam_coords. "
            "Got {} instead.".format(
            proj_mat.ndim
        )
    )

    assert proj_mat.ndim ==2 and proj_mat.shape[0] == cam_coords.shape[0], ValueError(
        "Batch sizes of proj_mat and cam_coords do not match. Shapes: {0} and {1} respectively.".format(
                proj_mat.shape, cam_coords.shape
            )
    )

    # Determine whether or not to homogenize `cam_coords`.
    to_homogenize: bool = cam_coords.shape[:-1] == 3

    pts_homo = None
    if to_homogenize:
        pts_homo = homogenize_points(cam_coords)
    else:
        pts_homo = cam_coords

    # Determine whether `proj_mat` needs to be expanded to match dims of `cam_coords`.
    to_expand_proj_mat: bool = (proj_mat.ndim == 2) and (pts_homo.ndim > 2)
    if to_expand_proj_mat:
        while proj_mat.ndim < pts_homo.ndim:
            proj_mat = ivy.expand_dims(proj_mat, axis=0)

    # Whether to perform simple matrix multiplaction instead of batch matrix multiplication.
    need_bmm: bool = pts_homo.ndim > 2

    if not need_bmm:
        pts = ivy.matmul(ivy.expand_dims(proj_mat, axis=0), ivy.expand_dims(pts_homo, axis=-1))
    else:
        pts = ivy.matmul(ivy.expand_dims(proj_mat, axis=-3), ivy.expand_dims(pts_homo, axis=-1))

    # Remove the extra dimension resulting from ivy.matmul()
    pts =  ivy.squeeze(pts, axis=-1)
    # Unhomogenize and stack.
    x = pts[..., 0]
    y = pts[..., 1]
    z = pts[..., 2]
    u = x / ivy.where(z != 0, z, ivy.ones_like(z))
    v = y / ivy.where(z != 0, z, ivy.ones_like(z))

    return ivy.stack((u,v), axis=-1)


@to_ivy_arrays_and_back
@handle_nestable
@handle_exceptions
@handle_array_function
@handle_array_like_without_promotion
def unproject_points(
    pixel_coords: Union[ivy.Array, ivy.NativeArray],
    intrinsics_inv: Union[ivy.Array, ivy.NativeArray],
    depths: Union[ivy.Array, ivy.NativeArray],
) -> ivy.Array:
    r"""Unprojects points from the image (pixel) frame to the camera coordinate frame.

    Parameters
    ----------
    pixel_coords: 
        pixel coordinates.
    intrinsics_inv: 
        inverse of the camera intrinsics matrix.
    depths: 
        per-pixel depth estimates.

    
    Returns:
        ivy.Array: camera coordinates

     Shapes:
        - pixel_coords: :math:`(N, *, 2)` or :math:`(*, 3)`, where * indicates an arbitrary number of dimensions.
          Here :math:`N` indicates the number of points in a cloud if the shape is :math:`(N, 3)` and indicates
          batchsize if the number of dimensions is greater than 2.
        - intrinsics_inv: :math:`(*, 3, 3)`, where * indicates an arbitrary number of dimensions.
        - depths: :math:`(N, *)` where * indicates the same number of dimensions as in `pixel_coords`.
          Here :math:`N` indicates the number of points in a cloud if the shape is :math:`(N, 3)` and indicates
          batchsize if the number of dimensions is greater than 2.
        - output: :math:`(N, *, 3)` where * indicates the same number of dimensions as in `pixel_coords`.
          Here :math:`N` indicates the number of points in a cloud if the shape is :math:`(N, 3)` and indicates
          batchsize if the number of dimensions is greater than 2.
    """

    assert ivy.is_array(pixel_coords), TypeError(
        "Expected input pixel_coords to be of type ivy.Array or ivy.NativeArray. Got {} instead".format(
            type(pixel_coords)
        )    
    )

    assert ivy.is_array(intrinsics_inv), TypeError(
        "Expected input intrinsics_inv to be of type ivy.Array or ivy.NativeArray. Got {} instead".format(
            type(intrinsics_inv)
        )
    )

    assert ivy.is_array(depths), TypeError(
        "Expected input depths to be of type ivy.Array or ivy.NativeArray. Got {} instead".format(
            type(depths)
        )
    )

    assert pixel_coords.ndim >= 2, ValueError(
        "Input pixel_coords must have at least 2 dimensions. Got {} instad.".format(
            pixel_coords.ndim
        )
    )

    assert pixel_coords.shape[-1] in (2, 3), ValueError(
        "Input pixel_coords must have last dimension of shape (*, 2) or (*, 3). Got {} instad.".format(
            pixel_coords.shape
        )
    )

    assert intrinsics_inv.ndim >= 2, ValueError(
        "Input intrinsics_inv must have at least 2 dimensions. Got {} instad.".format(
            intrinsics_inv.ndim
        )
    )

    assert intrinsics_inv.shape[-1] == 3 and intrinsics_inv[-2] == 3, ValueError(
        "Input intrinsics_inv must have shape (*, 3, 3). Got {} instad.".format(
            intrinsics_inv.shape
        )
    )

    assert intrinsics_inv.ndim == 2 or intrinsics_inv.shape[0] == pixel_coords.shape[0], ValueError(
         "Input intrinsics_inv must either have 2 dimensions, or have equal number of dimensions to pixel_coords. "
            "Got {0} instead.".format(intrinsics_inv.ndim)
    )

    assert intrinsics_inv.ndim == 2 or intrinsics_inv.shape[0] == depths.shape[0], ValueError(
        "Batch sizes of intrinsics_inv and pixel_coords do not match. Shapes: {0} and {1} respectively.".format(
                intrinsics_inv.shape, pixel_coords.shape
        )
    )

    assert pixel_coords.shape[:-1] == depths.shape, ValueError(
           "Input pixel_coords and depths must have the same shape for all dimensions except the last. "
            " Got {0} and {1} respectively.".format(
            pixel_coords.shape, depths.shape
        )
    )

    # Determine whether or not to homogenize `pixel_coords`.
    to_homogenize: bool = pixel_coords.shape[-1] == 2

    pts_homo = None
    if to_homogenize:
        pts_homo = homogenize_points(pixel_coords)
    else:
        pts_homo = pixel_coords

    # Determine whether `intrinsics_inv` needs to be expanded to match dims of `pixel_coords`.
    to_expand_intrinsics_inv: bool = (intrinsics_inv.ndim == 2) and (
        pts_homo.ndim > 2
    )
    if to_expand_intrinsics_inv:
        while intrinsics_inv.ndim < pts_homo.ndim:
            intrinsics_inv = ivy.expand_dims(intrinsics_inv, axis=0)
    
    # Whether to perform simple matrix multiplaction instead of batch matrix multiplication.
    need_bmm: bool = pts_homo.ndim > 2

    if not need_bmm:
        pts = ivy.matmul(ivy.expand_dims(intrinsics_inv, axis=0), ivy.expand_dims(pts_homo, axis=-1))
    else:
        pts = ivy.matmul(ivy.expand_dims(intrinsics_inv, axis=-3), ivy.expand_dims(pts_homo, axis=-1))
    
    # Remove the extra dimension resulting from torch.matmul()
    pts = ivy.squeeze(pts, axis=-1)

    return pts * ivy.expand_dims(pts, axis=-1)


@to_ivy_arrays_and_back
@handle_nestable
@handle_exceptions
@handle_array_function
@handle_array_like_without_promotion
def inverse_intrinsics(
    K: Union[ivy.Array, ivy.NativeArray],
    eps: float = 1e-6
) -> ivy.Array:
    r"""Efficient inversion of intrinsics matrix
    
    Parameters
    ----------
    K: Intrinsics matrix
    eps (float): Epsilon for numerical stability

    Returns
    ----------
    Ivy.Array: Inverted intrinsics matrix
    
    Shape
    ----------
    - K: :math:`(*, 4, 4)` or :math:`(*, 3, 3)`
    - Kinv: Matches shape of `K` (:math:`(*, 4, 4)` or :math:`(*, 3, 3)`)
    """

    assert ivy.is_array(K), TypeError(
        "Expected input K to be of type ivy.Array or ivy.NativeArray. Got {} instead".format(
            type(K)
        )
    )

    assert K.ndim >= 2, ValueError(
        "Input K must have at least 2 dimensions. Got {} instad.".format(K.ndim)
    )

    assert ((K.shape[-1] == 3 and K.shape[-2] == 3) or (K.shape[-1] == 4 and K.shape[-2] == 4)), ValueError(
        "Input K must have shape (*, 4, 4) or (*, 3, 3). Got {} instad.".format(
            K.shape
        )
    )

    Kinv = ivy.zeros_like(K)

    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]

    Kinv[..., 0, 0] = 1.0 / (fx + eps)
    Kinv[..., 1, 1] = 1.0 / (fy + eps)
    Kinv[..., 0, 2] = -1.0 * cx / (fx + eps)
    Kinv[..., 1, 2] = -1.0 * cy / (fy + eps)
    Kinv[..., 2, 2] = 1
    Kinv[..., -1, -1] = 1
    return Kinv

