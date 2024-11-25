import numpy as np
import math
import sys
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""

@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    # print(X.shape)
    # print(X)

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size
    
    # Can assume multiple of 128 to avoid using mask
    # assert in_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )
    X_out.reshape((batch_size, out_pool_height * out_pool_width, out_channels))

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax

    # Reshape input and weight to align for matrix multiplication
    # input =  X.reshape((batch_size, input_height * input_width, in_channels))
    weight = W.reshape((filter_height, filter_width, in_channels, out_channels))

    # Process the images in batches
    for b in nl.affine_range(batch_size):
        # TODO: Perform the convolution of X[b] with the weights W and bias b, followed by a maxpool
        # and store the result in X_out[b]

        # Iterate over the filter height
        for i in range(filter_height):
            # Iterate over the filter width
            for j in range(filter_width):

                # Shift the Input tensor by (i, j) to align with the filter's current position
                # input_shifted = shift(input, i, j, filter_height, filter_width)

                input = X[b:b+1, :, i : input_height - filter_height + i + 1, j : input_width - filter_width + j + 1]

                print("Here : ", input.shape)
                my_print(input)
                # input_shifted = nl.reshape(input, (out_width * out_height, in_channels))
                # input_shifted = X_reshaped[b:b+1, :, i : input_height - filter_height + i + 1, j : input_width - filter_width + j + 1].reshape((out_width * out_height, in_channels))
                # input_shifted = X.reshape((1, 1, out_width * out_height, in_channels))

                # Perform matrix multiplication between the input and the weights from the filter slice
                # X_out[b] += nl.matmul(input_shifted, weight[i, j, :, :])

    # Store the result tile into HBM
    # nl.store(X_out, value=X)
    X_out.reshape((batch_size, out_channels, out_pool_height, out_pool_width))
    for b in nl.affine_range(batch_size):
        for c in nl.affine_range(out_channels):
            for h in nl.affine_range(out_pool_height):
                for w in nl.affine_range(out_pool_width):
                    X_out[b, c, h, w] = 1

    return X_out

def my_print(input):
    nl.device_print("Printing Here:", x=input)
    # b_, c_, h_, w_ = input.shape
    # for b in range(b_):
    #     for c in range(c_):
    #         for h in range(h_):
    #             # for w in range(w_):
    #             row = nl.slice(input, (b, c, h, 0), (1,1,1,w_))
    #             nl.device_print(x=row)
    #         nl.device_print("---")
    # nl.device_print("\n --- Next Batch ---\n")
            