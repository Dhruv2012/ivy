import ivy

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


@handle_test(
    fn_tree="gradslam.geometry.so3_hat",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
        shared_dtype=True,
        min_value=0.0,
        max_value=1.0,
        min_num_dims=1,
        max_num_dims=2,
        min_dim_size=3,
    ),
    ground_truth_backend="jax"
)
def test_so3_hat(
    *,
    dtype_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        omega=x[0],
    )


@handle_test(
    fn_tree="gradslam.geometry.se3_hat",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
        shared_dtype=True,
        min_value=0.0,
        max_value=1.0,
        min_num_dims=1,
        max_num_dims=2,
        min_dim_size=6,
    ),
    ground_truth_backend="jax"
)
def test_se3_hat(
    *,
    dtype_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        xi=x[0],
    )


@handle_test(
    fn_tree="gradslam.geometry.so3_exp",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
        shared_dtype=True,
        min_value=0.0,
        max_value=1.0,
        min_num_dims=1,
        max_num_dims=2,
        min_dim_size=3,
    ),
    ground_truth_backend="jax"
)
def test_so3_exp(
    *,
    dtype_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        omega=x[0],
    )


@handle_test(
    fn_tree="gradslam.geometry.se3_exp",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
        shared_dtype=True,
        min_value=0.0,
        max_value=1.0,
        min_num_dims=1,
        max_num_dims=2,
        min_dim_size=6,
    ),
    ground_truth_backend="jax"
)
def test_se3_exp(
    *,
    dtype_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        xi=x[0],
    )
