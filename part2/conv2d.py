import numpy as np
import math
import sys
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


@nki.jit
def shift(input, input_shifted, i, j, X_shape, W_shape):
    batch_size, in_channels, input_height, input_width = X_shape
    out_channels, in_channels_, filter_height, filter_width = W_shape
    
    shift_idx = 0

    for x in nl.affine_range(i, input_height - filter_height + i + 1):
        for y in nl.affine_range(j, input_width - filter_width + j + 1):
            flattened_idx = x * input_width + y
            shift_idx = (x-i) * filter_width + (y-j)
            # my_print_2D(input[:, flattened_idx], "Spliced Input : ")
            input_shifted[:, shift_idx] = input[:, flattened_idx]
            # my_print_2D(input_shifted[:, shift_idx], "Shifted Input : ")

    return input_shifted


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
def manual_transpose_3d(tensor, transposed_tensor):
    
    batches,height,width = tensor.shape
    
    # Assuming tensor is a 3-dimensional list of lists
    # transposed_tensor = np.zeros([batches,width,height])
    # transposed_tensor = nl.ndarray((batches, width, height), dtype=tensor.dtype, buffer=nl.hbm)
    temp = nl.ndarray((1, 1), dtype=tensor.dtype, buffer=nl.sbuf)

    for i in range(batches):
        for j in range(height):
            for k in range(width):
                # transposed_tensor[i][k][j] = tensor[i][j][k]
                temp[...] = nl.load(tensor[i][j][k])
                nl.store(transposed_tensor[i][k][j], value=temp)


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

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax

    # Reshape input and weight to align for matrix multiplication
    input =  X.reshape((batch_size, in_channels, input_height * input_width))
    # transposed_input = nl.ndarray((batch_size, input_height * input_width, in_channels), dtype=input.dtype, buffer=nl.hbm)
    # manual_transpose_3d(input, transposed_input)
    my_print(input, "INPUT : ")
    print("Shape of the INPUT : ", input.shape)

    weight = W.reshape((out_channels, in_channels, filter_height * filter_width))
    # transposed_weight = nl.ndarray((out_channels, filter_height * filter_width, in_channels), dtype=input.dtype, buffer=nl.hbm)
    # manual_transpose_3d(weight, transposed_weight)
    my_print(weight, "Weight : ")
    print("Shape of the Wts : ", weight.shape)

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_height, out_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    X_out_mul = nl.ndarray(
        shape=(batch_size, out_channels, out_height, out_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    PARTITION_DIM = min(in_channels, 128)
    FREE_DIM = min(input_height * input_width, 512)
    FREE_DIM_shifted = min(out_height * out_width, 512)

    TILE_M = min(out_channels, nl.tile_size.gemm_stationary_fmax)  # 128
    TILE_K = nl.tile_size.pmax  # 128
    TILE_N = min(out_height * out_width, nl.tile_size.gemm_moving_fmax)  # 512

    # Process the images in batches
    for b in nl.affine_range(batch_size):
        # TODO: Perform the convolution of X[b] with the weights W and bias b, followed by a maxpool
        # and store the result in X_out[b]

        res_psum = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

        # Iterate over the filter height
        for i in range(filter_height):
            # Iterate over the filter width
            for j in range(filter_width):
                input_tile = nl.ndarray((PARTITION_DIM, FREE_DIM), dtype=input.dtype, buffer=nl.sbuf)
                input_shifted_tile = nl.ndarray((PARTITION_DIM, FREE_DIM_shifted), dtype=input.dtype, buffer=nl.sbuf)
                weight_tile = nl.ndarray((out_channels, in_channels, filter_height * filter_width), dtype=input.dtype, buffer=nl.sbuf)

                input_tile = nl.load(input[b])
                input_shifted_tile = shift(input_tile, input_shifted_tile, i, j, X.shape, W.shape)
                my_print_2D(input_shifted_tile, "Input Shifted Tile : ")

                weight_tile = nl.load(weight)
                my_print_2D(weight_tile[:,:,i * filter_width + j], "Weight Tile : ")

                res_psum += nl.matmul(weight_tile[:,:,i * filter_width + j], input_shifted_tile)
        
        nl.store(X_out_mul[b], value=res_psum.reshape((out_channels, out_height, out_width)))

        my_print_2D(X_out_mul[b], "X_OUT : ")

        print("Shape of Xout : ", X_out_mul.shape)

    # # Store the result tile into HBM
    # # nl.store(X_out, value=X)
    # temp = nl.ndarray((1, 1), dtype=X_out.dtype, buffer=nl.sbuf)
    
    # # # X_out_reshaped = X_out.reshape((batch_size, out_channels, out_pool_height, out_pool_width))
    # for b in nl.affine_range(batch_size):
    #     for c in nl.affine_range(out_channels):
    #         for h in nl.affine_range(out_height):
    #             for w in nl.affine_range(out_width):
    #                 temp[0][0] = 0
    #                 nl.store(X_out[b, c, h, w], value=temp)
    #                 # X_out[b, c, h * out_width + w] = 1

    # # print("---> Shape of X_out : ", X_out_reshaped.shape)

    return X_out_mul

def my_print(input, name):
    nl.device_print(f"{name}", x=input[0, 0, 0])
    nl.device_print("", x=input)
    # b_, c_, h_, w_ = input.shape
    # for b in range(b_):
    #     for c in range(c_):
    #         for h in range(h_):
    #             # for w in range(w_):
    #             row = nl.slice(input, (b, c, h, 0), (1,1,1,w_))
    #             nl.device_print(x=row)
    #         nl.device_print("---")
    # nl.device_print("\n --- Next Batch ---\n")

def my_print_2D(input, name):
    nl.device_print(f"{name}", x=input[0, 0])
    nl.device_print("", x=input)

def my_print_generic(input):
    nl.device_print("", x=input)

############################################## CPU Implementation ##############################################

@nki.jit
def nki_matmul_tiled_(lhsT, rhs, result):
  """NKI kernel to compute a matrix multiplication operation in a tiled manner

  Args:
      lhsT: an input tensor of shape [K,M], where both K and M are multiples for
        128.  It is the left-hand-side argument of the matrix multiplication,
        delivered transposed for optimal performance.
      rhs: an input tensor of shape [K,N], where K is a multiple of 128, and N
        is a multiple of 512.  It is the right-hand-side argument of the matrix
        multiplication.
      result: the resulting output tensor of shape [M,N]
  """

  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"

  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  # Use affine_range to loop over tiles
  for m in nl.affine_range(M // TILE_M):
    for n in nl.affine_range(N // TILE_N):
      # Allocate a tensor in PSUM
      res_psum = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

      for k in nl.affine_range(K // TILE_K):
        print("Here123")
        # Declare the tiles on SBUF
        lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
        rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=nl.float32, buffer=nl.sbuf)

        # Load tiles from lhsT and rhs
        lhsT_tile[...] = nl.load(lhsT[k * TILE_K:(k + 1) * TILE_K,
                                      m * TILE_M:(m + 1) * TILE_M])
        rhs_tile[...] = nl.load(rhs[k * TILE_K:(k + 1) * TILE_K,
                                    n * TILE_N:(n + 1) * TILE_N])

        # Accumulate partial-sums into PSUM
        res_psum += nl.matmul(lhsT_tile[...], rhs_tile[...], transpose_x=True)

      # Copy the result from PSUM back to SBUF, and cast to expected output data-type
      res_sb = nl.copy(res_psum, dtype=nl.float32)
      nl.store(result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N],
               value=res_sb)

def shift_cpu(input, input_shifted, i, j, X_shape, W_shape):
    batch_size, in_channels, input_height, input_width = X_shape
    out_channels, in_channels_, filter_height, filter_width = W_shape
    
    shift_idx = 0

    for x in range(i, input_height - filter_height + i + 1):
        for y in range(j, input_width - filter_width + j + 1):
            flattened_idx = x * input_width + y
            input_shifted[shift_idx] = input[flattened_idx]
            shift_idx += 1
    
    return input_shifted

def fused_conv2d_maxpool_cpu(X, W, bias, pool_size=1):
    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size

    # Reshape input and weight to align for matrix multiplication
    input = X.reshape((batch_size, in_channels, input_height * input_width))
    print(input)
    print(input.shape)
    transposed_X = np.transpose(input, (0, 2, 1))
    
    weight = W.reshape((out_channels, in_channels, filter_height * filter_width))
    transposed_W = np.transpose(weight, (0, 2, 1))
    
    print("Transposed X : ")
    print(transposed_X)
    print("Shape : ", transposed_X.shape)
    print()

    print("Transposed W : ")
    print(transposed_W)
    print("Shape : ", transposed_W.shape)
    print()

    # Initialize Output with zeros
    output = np.zeros([batch_size, out_height * out_width, out_channels], dtype=np.float32)

    input_shifted = np.zeros([out_height * out_width, in_channels], dtype=np.float32)

    print("\n----------------- Begin Iterations -----------------")
    for b in range(batch_size):
        output[b] = np.zeros([out_height * out_width, out_channels])
        result = np.zeros([out_height * out_width, out_channels])

        # Iterate over the filter height
        for i in range(filter_height):
            # Iterate over the filter width
            for j in range(filter_width):
                
                # Shift the Input tensor by (i, j) to align with the filter's current position
                input_shifted = shift_cpu(transposed_X[b], input_shifted, i, j, X.shape, W.shape)

                print(f"\n----- Batch : {b} -----")
                print("Shifted Input for filter indices : (", i, j, ")")
                print(input_shifted.shape)
                input_shifted_T = np.transpose(input_shifted)
                print()

                # Getting the right set of weights across (output_channels, input_channels)
                weight_sliced = transposed_W[:, i * filter_width + j]
                weight_sliced_T = np.transpose(weight_sliced)

                print("Weight slice for filter indices : (", i, j, ")")
                print(weight_sliced_T.shape)
                print()
                print(f"\n-----------------------")

                # Perform matrix multiplication between the input and the weights from the filter slice
                # output[b] += np.matmul(input_shifted, weight_sliced_T)

                nki_matmul_tiled_(input_shifted_T, weight_sliced_T, result)
                # output[b] += result
    print("\n----------------- End Iterations -----------------")

    print("\n------- ")
    print("Output before transposing and reshaping : ")
    print(output)
    print("------- \n")

    output_T = np.transpose(output, (0, 2, 1))
    output = output_T.reshape((batch_size, out_channels, out_height, out_width))

    print("\n------- ")
    print("Output Matrix (CPU Implementation): ", out_height, out_width)
    print(output)
    print("------- \n")

    return output
