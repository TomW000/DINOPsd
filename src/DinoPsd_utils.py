import numpy as np
import torchvision.transforms.v2.functional as T
from torch.nn import functional as F
from torchvision import transforms
import math


def get_img_processing_f(
    resize_size=518,
    interpolation=transforms.InterpolationMode.BICUBIC,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    # input  tensor: [(b),h,w,c]
    # output tensor: [(b),c,h,w]
    def _img_processing_f(x):
        if len(x.shape) == 4:
            if x.shape[-1] == 1:
                x = x.repeat(1, 1, 1, 3)
            x = x.permute(0, 3, 1, 2)
        else:
            if x.shape[-1] == 1:
                x = x.repeat(1, 1, 3)
            x = x.permute(2, 0, 1)
        x = T.resize(
            x, resize_size, interpolation=interpolation, antialias=True
        )
        x = T.normalize(x, mean=mean, std=std)
        return x

    return _img_processing_f


def resizeLongestSide(np_image, new_longest_size):
    h, w, *_ = np_image.shape
    scale = new_longest_size / max(h, w)
    hNew, wNew = h * scale, w * scale
    new_shape = (int(hNew + 0.5), int(wNew + 0.5))
    return np.array(T.resize(T.to_pil_image(np_image), new_shape))


def mirror_border(image, sizeH, sizeW):
    h_res = sizeH - image.shape[0]
    w_res = sizeW - image.shape[1]

    top = bot = h_res // 2
    left = right = w_res // 2
    top += 1 if h_res % 2 != 0 else 0
    left += 1 if w_res % 2 != 0 else 0

    res_image = np.pad(image, ((top, bot), (left, right), (0, 0)), "symmetric")
    return res_image


def crop_data_with_overlap(
    data,
    crop_shape,
    data_mask=None,
    overlap=(0, 0),
    padding=(0, 0),
    verbose=True,
):
    """Crop data into small square pieces with overlap. The difference with :func:`~crop_data` is that this function
    allows you to create patches with overlap.

    The opposite function is :func:`~merge_data_with_overlap`.

    Parameters
    ----------
    data : 4D Numpy array
        Data to crop. E.g. ``(num_of_images, y, x, channels)``.

    crop_shape : 3 int tuple
        Shape of the crops to create. E.g. ``(y, x, channels)``.

    data_mask : 4D Numpy array, optional
        Data mask to crop. E.g. ``(num_of_images, y, x, channels)``.

    overlap : Tuple of 2 floats, optional
        Amount of minimum overlap on x and y dimensions. The values must be on range ``[0, 1)``, that is, ``0%`` or
        ``99%`` of overlap. E. g. ``(y, x)``.

    padding : tuple of ints, optional
        Size of padding to be added on each axis ``(y, x)``. E.g. ``(24, 24)``.

    verbose : bool, optional
         To print information about the crop to be made.

    Returns
    -------
    cropped_data : 4D Numpy array
        Cropped image data. E.g. ``(num_of_images, y, x, channels)``.

    cropped_data_mask : 4D Numpy array, optional
        Cropped image data masks. E.g. ``(num_of_images, y, x, channels)``.

    Examples
    --------
    ::

        # EXAMPLE 1
        # Divide in crops of (256, 256) a given data with the minimum overlap
        X_train = np.ones((165, 768, 1024, 1))
        Y_train = np.ones((165, 768, 1024, 1))

        X_train, Y_train = crop_data_with_overlap(X_train, (256, 256, 1), Y_train, (0, 0))

        # Notice that as the shape of the data has exact division with the wnanted crops shape so no overlap will be
        # made. The function will print the following information:
        #     Minimum overlap selected: (0, 0)
        #     Real overlapping (%): (0.0, 0.0)
        #     Real overlapping (pixels): (0.0, 0.0)
        #     (3, 4) patches per (x,y) axis
        #     **** New data shape is: (1980, 256, 256, 1)


        # EXAMPLE 2
        # Same as example 1 but with 25% of overlap between crops
        X_train, Y_train = crop_data_with_overlap(X_train, (256, 256, 1), Y_train, (0.25, 0.25))

        # The function will print the following information:
        #     Minimum overlap selected: (0.25, 0.25)
        #     Real overlapping (%): (0.33203125, 0.3984375)
        #     Real overlapping (pixels): (85.0, 102.0)
        #     (4, 6) patches per (x,y) axis
        #     **** New data shape is: (3960, 256, 256, 1)


        # EXAMPLE 3
        # Same as example 1 but with 50% of overlap between crops
        X_train, Y_train = crop_data_with_overlap(X_train, (256, 256, 1), Y_train, (0.5, 0.5))

        # The function will print the shape of the created array. In this example:
        #     Minimum overlap selected: (0.5, 0.5)
        #     Real overlapping (%): (0.59765625, 0.5703125)
        #     Real overlapping (pixels): (153.0, 146.0)
        #     (6, 8) patches per (x,y) axis
        #     **** New data shape is: (7920, 256, 256, 1)


        # EXAMPLE 4
        # Same as example 2 but with 50% of overlap only in x axis
        X_train, Y_train = crop_data_with_overlap(X_train, (256, 256, 1), Y_train, (0.5, 0))

        # The function will print the shape of the created array. In this example:
        #     Minimum overlap selected: (0.5, 0)
        #     Real overlapping (%): (0.59765625, 0.0)
        #     Real overlapping (pixels): (153.0, 0.0)
        #     (6, 4) patches per (x,y) axis
        #     **** New data shape is: (3960, 256, 256, 1)
    """

    if data_mask is not None:
        if data.shape[:-1] != data_mask.shape[:-1]:
            raise ValueError(
                f"data and data_mask shapes mismatch: {data.shape[:-1]} vs {data_mask.shape[:-1]}"
            )

    for i, p in enumerate(padding):
        if p >= crop_shape[i] // 2:
            raise ValueError(
                f"'Padding' can not be greater than the half of 'crop_shape'. Max value for this {data.shape} input shape is {[(crop_shape[0] // 2) - 1, (crop_shape[1] // 2) - 1]}"
            )
    if len(crop_shape) != 3:
        raise ValueError(
            f"crop_shape expected to be of length 3, given {crop_shape}"
        )
    if crop_shape[0] > data.shape[1]:
        raise ValueError(
            f"'crop_shape[0]' {crop_shape[0]} greater than {data.shape[1]} (you can reduce 'DATA.PATCH_SIZE' or use 'DATA.REFLECT_TO_COMPLETE_SHAPE')"
        )
    if crop_shape[1] > data.shape[2]:
        raise ValueError(
            f"'crop_shape[1]' {crop_shape[1]} greater than {data.shape[2]} (you can reduce 'DATA.PATCH_SIZE' or use 'DATA.REFLECT_TO_COMPLETE_SHAPE')"
        )
    if (overlap[0] >= 1 or overlap[0] < 0) or (
        overlap[1] >= 1 or overlap[1] < 0
    ):
        raise ValueError(
            "'overlap' values must be floats between range [0, 1)"
        )

    if verbose:
        print("### OV-CROP ###")
        print(
            f"Cropping {data.shape} images into {crop_shape} with overlapping. . ."
        )
        print(f"Minimum overlap selected: {overlap}")
        print(f"Padding: {padding}")

    if (overlap[0] >= 1 or overlap[0] < 0) and (
        overlap[1] >= 1 or overlap[1] < 0
    ):
        raise ValueError(
            "'overlap' values must be floats between range [0, 1)"
        )

    padded_data = np.pad(
        data,
        ((0, 0), (padding[1], padding[1]), (padding[0], padding[0]), (0, 0)),
        "reflect",
    )
    if data_mask is not None:
        padded_data_mask = np.pad(
            data_mask,
            (
                (0, 0),
                (padding[1], padding[1]),
                (padding[0], padding[0]),
                (0, 0),
            ),
            "reflect",
        )

    # Calculate overlapping variables
    overlap_x = 1 if overlap[0] == 0 else 1 - overlap[0]
    overlap_y = 1 if overlap[1] == 0 else 1 - overlap[1]

    # Y
    step_y = int((crop_shape[0] - padding[0] * 2) * overlap_y)
    crops_per_y = math.ceil(data.shape[1] / step_y)
    last_y = (
        0
        if crops_per_y == 1
        else (((crops_per_y - 1) * step_y) + crop_shape[0])
        - padded_data.shape[1]
    )
    ovy_per_block = last_y // (crops_per_y - 1) if crops_per_y > 1 else 0
    step_y -= ovy_per_block
    last_y -= ovy_per_block * (crops_per_y - 1)

    # X
    step_x = int((crop_shape[1] - padding[1] * 2) * overlap_x)
    crops_per_x = math.ceil(data.shape[2] / step_x)
    last_x = (
        0
        if crops_per_x == 1
        else (((crops_per_x - 1) * step_x) + crop_shape[1])
        - padded_data.shape[2]
    )
    ovx_per_block = last_x // (crops_per_x - 1) if crops_per_x > 1 else 0
    step_x -= ovx_per_block
    last_x -= ovx_per_block * (crops_per_x - 1)

    # Real overlap calculation for printing
    real_ov_y = ovy_per_block / (crop_shape[0] - padding[0] * 2)
    real_ov_x = ovx_per_block / (crop_shape[1] - padding[1] * 2)

    if verbose:
        print(f"Real overlapping (%): {real_ov_x}")
        print(
            f"Real overlapping (pixels): {(crop_shape[1] - padding[1] * 2) * real_ov_x}"
        )
        print(f"{crops_per_x} patches per (x,y) axis")

    total_vol = data.shape[0] * (crops_per_x) * (crops_per_y)
    cropped_data = np.zeros((total_vol,) + crop_shape, dtype=data.dtype)
    if data_mask is not None:
        cropped_data_mask = np.zeros(
            (total_vol,) + crop_shape[:2] + (data_mask.shape[-1],),
            dtype=data_mask.dtype,
        )

    c = 0
    for z in range(data.shape[0]):
        for y in range(crops_per_y):
            for x in range(crops_per_x):
                d_y = (
                    0
                    if (y * step_y + crop_shape[0]) < padded_data.shape[1]
                    else last_y
                )
                d_x = (
                    0
                    if (x * step_x + crop_shape[1]) < padded_data.shape[2]
                    else last_x
                )

                cropped_data[c] = padded_data[
                    z,
                    y * step_y - d_y : y * step_y + crop_shape[0] - d_y,
                    x * step_x - d_x : x * step_x + crop_shape[1] - d_x,
                ]

                if data_mask is not None:
                    cropped_data_mask[c] = padded_data_mask[
                        z,
                        y * step_y - d_y : y * step_y + crop_shape[0] - d_y,
                        x * step_x - d_x : x * step_x + crop_shape[1] - d_x,
                    ]
                c += 1

    if verbose:
        print(f"**** New data shape is: {cropped_data.shape}")
        print("### END OV-CROP ###")

    if data_mask is not None:
        return cropped_data, cropped_data_mask
    else:
        return cropped_data
    
    
def gaussian_kernel(size=3, sigma=1):

    upper = size - 1
    lower = -int(size / 2)

    y, x = np.mgrid[lower:upper, lower:upper]

    kernel = (1 / (2 * np.pi * sigma**2)) * np.exp(
        -(x**2 + y**2) / (2 * sigma**2)
    )
    kernel = kernel / kernel.sum()

    return kernel
    
    
def torch_convolve(input, weights, mode="reflect", cval=0.0, origin=0):
    """
    Multidimensional convolution using PyTorch.

    Parameters
    ----------
    input : torch.Tensor
        The input tensor to be convolved.
    weights : torch.Tensor
        Convolution kernel, with the same number of dimensions as the input.
    mode : str, optional
        Padding mode. Options are 'reflect', 'constant', 'replicate', or 'circular'.
        Default is 'reflect'.
    cval : float, optional
        Value to fill past edges of input if `mode` is 'constant'. Default is 0.0.
    origin : int, optional
        Controls the origin of the input signal. Positive values shift the filter
        to the right, and negative values shift the filter to the left. Default is 0.

    Returns
    -------
    result : torch.Tensor
        The result of convolution of `input` with `weights`.
    """
    # Ensure input is 4D (batch, channels, height, width)
    if input.dim() == 2:  # Single channel 2D image
        input = input.unsqueeze(0).unsqueeze(0)
    elif input.dim() == 3:  # Add batch dimension if missing
        input = input.unsqueeze(0)

    # Add channel dimension for weights if necessary
    if weights.dim() == 2:
        weights = weights.unsqueeze(0).unsqueeze(0)

    # Apply padding based on mode
    padding = (
        weights.shape[-1] // 2 - origin
    )  # Adjust padding for origin shift
    input_padded = F.pad(
        input, (padding, padding, padding, padding), mode=mode, value=cval
    )

    # Perform convolution
    result = F.conv2d(input_padded, weights)

    return result.squeeze()  # Remove extra dimensions for output
