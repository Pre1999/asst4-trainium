import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal
from neuronxcc.nki import benchmark

from conv2d import fused_conv2d_maxpool as conv2d

from conv2d_numpy import conv2d_cpu_torch
import logging
import argparse
import io
import sys

import subprocess

logging.disable(logging.OFF)


def save_trace(profile_name, neff_file_name):
    """Run the profiler and save the NEFF and NTFF files with the specified name."""
    subprocess.run(
        [
            "neuron-profile",
            "capture",
            "-n",
            neff_file_name,
            "-s",
            profile_name + ".ntff",
        ],
        check=True,
    )

    subprocess.run(["mv", neff_file_name, profile_name + ".neff"], check=True)

    print(
        f"\n\nNEFF / NTFF files generated with names: {profile_name + '.neff'}, {profile_name + '.ntff'}"
    )


def test_correctness_conv2d_kernel(
    kernel,
    use_cpu_impl=False,
    use_larger_images=False,
    use_bias=False,
    use_maxpool=False,
):
    ref_impl = conv2d_cpu_torch

    input_channels_list = [3]
    output_channels_list = [1]
    kernel_size_list = [2]
    batch_size_list = [1]
    image_dims_list = [(3, 3)]
    pool_size = 2 if use_maxpool else 1

    if use_larger_images:
        input_channels_list = [512]
        output_channels_list = [512]
        image_dims_list = [(256, 256)]

    for input_channels in input_channels_list:
        for output_channels in output_channels_list:
            for kernel_size in kernel_size_list:
                for batch_size in batch_size_list:
                    for image_dims in image_dims_list:
                        # X = np.random.rand(
                        #     batch_size, input_channels, image_dims[0], image_dims[1]
                        # ).astype(np.float32)
                        # X = np.ones(
                        #     (batch_size, input_channels, image_dims[0], image_dims[1])
                        # ).astype(np.float32)

                        # Define the base 3x3 matrix with values from 1 to 9
                        base_matrix = np.array([[1, 2, 3],
                                                [4, 5, 6],
                                                [7, 8, 9]], dtype=np.float32)
                        X = np.tile(base_matrix, (2, 3, 1, 1))
                        print("Input Matrix : ")
                        print(X)
                        print("\nShape : ", X.shape)

                        # W = np.random.rand(
                        #     output_channels, input_channels, kernel_size, kernel_size
                        # ).astype(np.float32)

                        # Define Filter Matrix
                        base_matrix = np.array([[1, 2],
                                                [3, 4]], dtype=np.float32)
                        W = np.tile(base_matrix, (1, 3, 1, 1))

                        # Define a single channel for Output Channel 1
                        # channel1 = np.array([[1., 2.],
                        #                     [3., 4.]])

                        # # Define a single channel for Output Channel 2
                        # channel2 = np.array([[5., 6.],
                        #                     [7., 8.]])

                        # # Stack three copies of channel1 for Input Channels of Output Channel 1
                        # output_channel1 = np.stack([channel1] * 3)

                        # # Stack three copies of channel2 for Input Channels of Output Channel 2
                        # output_channel2 = np.stack([channel2] * 3)

                        # Combine the two output channels
                        # W = np.stack([output_channel1, output_channel2])
                        # print()
                        print("Filter Matrix : ")
                        print(W)
                        print("\nShape : ", W.shape)
                        print("\n ----- Done Initializing Matrices ----- \n")

                        bias = (
                            np.zeros(output_channels).astype(np.float32)
                            if not use_bias
                            else np.random.rand(output_channels).astype(np.float32)
                        )

                        args = [X, W, bias]
                        kwargs = {"pool_size": pool_size}

                        out = kernel(*args, **kwargs)
                        out_ref = ref_impl(*args, **kwargs)
                        
                        print("Shapes of output : ", out.shape, " | Ref Shape : ", out_ref.shape)

                        print("Out Ref : ", out_ref)

                        if not np.allclose(out, out_ref):
                            print(
                                f"Output mismatch for input_channels: {input_channels}, \
                        output_channels: {output_channels}, kernel_size: {kernel_size}, batch_size: {batch_size},\
                         image_dims: {image_dims}, use_bias: {use_bias}, use_maxpool: {use_maxpool}"
                            )

                            return False

    return True


def test_performance_conv2d_kernel(
    kernel,
    dtype=np.float32,
    batch_size=10,
    in_channels=256,
    out_channels=256,
    image_height=224,
    image_width=224,
    kernel_height=3,
    kernel_width=3,
    pool_size=1,
):

    performance_requirements_by_dtype = {
        np.float32: 42500,
        np.float16: 12000
    }

    X = np.random.rand(batch_size, in_channels, image_height, image_width).astype(dtype)
    W = np.random.rand(out_channels, in_channels, kernel_height, kernel_width).astype(
        dtype
    )
    bias = np.random.rand(out_channels).astype(dtype)

    args = [X, W, bias]
    kwargs = {"pool_size": pool_size}

    bench_func = nki.benchmark(
        warmup=5, iters=20, save_neff_name=f"file_pool_{pool_size}.neff"
    )(kernel)
    text_trap = io.StringIO()
    sys.stdout = text_trap
    bench_func(*args, **kwargs)
    sys.stdout = sys.__stdout__
    p99_us_student = bench_func.benchmark_result.nc_latency.get_latency_percentile(99)
    print(f"\n\nExecution Time for student implementation: {p99_us_student} μs")

    if p99_us_student > performance_requirements_by_dtype[dtype]:
        print(f"Performance requirement not met: need to be under {performance_requirements_by_dtype[dtype]} μs")
        return False

    return True


# write a function g which when passed a function f, returns a new function that when called with some *args and **kwargs, calls
# nki.simulate_kernel(f, *args, **kwargs) and returns the result
def simulate_kernel_wrapper(kernel):
    def temp_func(*args, **kwargs):
        return nki.simulate_kernel(kernel, *args, **kwargs)

    return temp_func


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_maxpool", action="store_true", help="Use smaller images for testing"
    )
    parser.add_argument(
        "--profile", type=str, default=None, help="File to save the neff file"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Use nki.simulate_kernel to run student implementation",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for random number generation"
    )

    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.simulate:
        conv2d = simulate_kernel_wrapper(conv2d)
    # running correctness tests
    print(
        "Running correctness test for conv2d kernel with smaller images...",
        end="",
        flush=True,
    )
    print()
    test_result = test_correctness_conv2d_kernel(conv2d, use_larger_images=False)
    if test_result:
        print("Passed 😎")
    else:
        print("Failed 😢")
    sys.exit()
    print(
        "Running correctness test for conv2d kernel with larger images...",
        end="",
        flush=True,
    )
    test_result = test_correctness_conv2d_kernel(conv2d, use_larger_images=True)
    if test_result:
        print("Passed 😇")
    else:
        print("Failed 😢")

    print(
        "Running correctness test for conv2d kernel with larger images + bias...",
        end="",
        flush=True,
    )
    test_result = test_correctness_conv2d_kernel(
        conv2d, use_bias=True, use_larger_images=True
    )
    if test_result:
        print("Passed 😍")
    else:
        print("Failed 😢")

    if args.test_maxpool:
        print(
            "Running correctness test for conv2d kernel with larger images + bias + maxpool...",
            end="",
            flush=True,
        )
        test_result = test_correctness_conv2d_kernel(
            conv2d, use_bias=True, use_maxpool=True, use_larger_images=True
        )
        if test_result:
            print("Passed 😍")
        else:
            print("Failed 😢")

    if args.simulate:
        exit()

    print("Comparing performance with reference kernel (no maxpool, float32)...")
    test_result = test_performance_conv2d_kernel(conv2d, pool_size=1, dtype = np.float32)
    if test_result:
        print("Performance test passed 😍")
    else:
        print("Performance test failed 😢")

    if args.profile is not None:
        save_trace(args.profile, "file_pool_1.neff")
    
    print("Comparing performance with reference kernel (no maxpool, float16)...")
    test_result = test_performance_conv2d_kernel(conv2d, pool_size=1, dtype = np.float16)
    if test_result:
        print("Performance test passed 😍")
    else:
        print("Performance test failed 😢")

    if args.profile is not None:
        save_trace(args.profile, "file_pool_1.neff")

    if args.test_maxpool:
        print("Comparing performance with reference kernel (with maxpool, float32)...")
        test_result = test_performance_conv2d_kernel(conv2d, pool_size=2, dtype = np.float32)
        if test_result:
            print("Performance test passed 😍")
        else:
            print("Performance test failed 😢")

        if args.profile is not None:
            save_trace(args.profile + "_pool", "file_pool_2.neff")

        print("Comparing performance with reference kernel (with maxpool, float16)...")
        test_result = test_performance_conv2d_kernel(conv2d, pool_size=2, dtype = np.float16)
        if test_result:
            print("Performance test passed 😍")
        else:
            print("Performance test failed 😢")

        if args.profile is not None:
            save_trace(args.profile + "_pool", "file_pool_2.neff")
