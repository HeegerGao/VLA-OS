"""Utils for evaluating robot policies in various environments."""

import os
import random
import time

import numpy as np
import torch
import open3d as o3d # type: ignore # Open3D might not have type stubs

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_image_resize_size(cfg):
    """
    Gets image resize size for a model class.
    If `resize_size` is an int, then the resized image will be a square.
    Else, the image will be a rectangle.
    """
    if cfg.model_family == "openvla":
        resize_size = 224
    elif cfg.model_family == "vla":
        resize_size = 224
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return resize_size

def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action

def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action

def _create_uniform_pixel_coords_image(resolution: np.ndarray, batch_size=None):
    """Creates uniform pixel coordinates.
    
    Args:
        resolution: Image resolution [height, width]
        batch_size: Optional batch size
        
    Returns:
        Pixel coordinates with shape [batch_size, height, width, 3] if batch_size is provided,
        otherwise [height, width, 3]
    """
    pixel_x_coords = np.reshape(
        np.tile(np.arange(resolution[1]), [resolution[0]]),
        (resolution[0], resolution[1], 1)).astype(np.float32)
    pixel_y_coords = np.reshape(
        np.tile(np.arange(resolution[0]), [resolution[1]]),
        (resolution[1], resolution[0], 1)).astype(np.float32)
    pixel_y_coords = np.transpose(pixel_y_coords, (1, 0, 2))
    uniform_pixel_coords = np.concatenate(
        (pixel_x_coords, pixel_y_coords, np.ones_like(pixel_x_coords)), -1)

    if batch_size is not None:
        # Add batch dimension
        uniform_pixel_coords = np.tile(
            np.expand_dims(uniform_pixel_coords, 0), [batch_size, 1, 1, 1])

    return uniform_pixel_coords

def _transform(coords, trans):
    h, w = coords.shape[:2]
    coords = np.reshape(coords, (h * w, -1))
    coords = np.transpose(coords, (1, 0))
    transformed_coords_vector = np.matmul(trans, coords)
    transformed_coords_vector = np.transpose(
        transformed_coords_vector, (1, 0))
    return np.reshape(transformed_coords_vector,
                    (h, w, -1))

def _pixel_to_world_coords(pixel_coords, cam_proj_mat_inv):
    h, w = pixel_coords.shape[:2]
    pixel_coords = np.concatenate(
        [pixel_coords, np.ones((h, w, 1))], -1)
    world_coords = _transform(pixel_coords, cam_proj_mat_inv)
    world_coords_homo = np.concatenate(
        [world_coords, np.ones((h, w, 1))], axis=-1)
    return world_coords_homo

def pointcloud_from_depth_and_camera_params(
        depth: np.ndarray, extrinsics: np.ndarray,
        intrinsics: np.ndarray) -> np.ndarray:
    """Converts depth (in meters) to point cloud in world frame.
    
    Args:
        depth: Depth map with shape [batch_size, height, width, 1] or [height, width, 1]
        extrinsics: Camera extrinsics with shape [batch_size, 4, 4] or [4, 4]
        intrinsics: Camera intrinsics with shape [batch_size, 3, 3] or [3, 3]
        
    Returns:
        Point cloud in world coordinates with shape [batch_size, height, width, 3] or [height, width, 3]
    """
    # Check if inputs have batch dimension
    has_batch = len(depth.shape) > 3

    if has_batch:
        batch_size = depth.shape[0]
        resolution = np.array(depth.shape[1:3])

        # Create uniform pixel coordinates with batch dimension
        upc = _create_uniform_pixel_coords_image(resolution, batch_size)

        # Multiply by depth to get camera space coordinates
        pc = upc * depth

        # Process each item in the batch
        world_coords_list = []
        for b in range(batch_size):
            # Extract extrinsics and intrinsics for this batch item
            extr_b = extrinsics[b]
            intr_b = intrinsics[b]

            # Calculate camera projection matrix and its inverse
            C = np.expand_dims(extr_b[:3, 3], 0).T
            R = extr_b[:3, :3]
            R_inv = R.T  # inverse of rot matrix is transpose
            R_inv_C = np.matmul(R_inv, C)
            extr_transformed = np.concatenate((R_inv, -R_inv_C), -1)
            cam_proj_mat = np.matmul(intr_b, extr_transformed)
            cam_proj_mat_homo = np.concatenate(
                [cam_proj_mat, [np.array([0, 0, 0, 1])]])
            cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[0:3]

            # Transform to world coordinates
            world_coords_homo = _pixel_to_world_coords(
                pc[b], cam_proj_mat_inv)
            world_coords = world_coords_homo[..., :-1]
            world_coords_list.append(world_coords)

        # Stack results along batch dimension
        return np.stack(world_coords_list, axis=0)
    else:
        # Original non-batched implementation
        upc = _create_uniform_pixel_coords_image(depth.shape)
        pc = upc * depth
        C = np.expand_dims(extrinsics[:3, 3], 0).T
        R = extrinsics[:3, :3]
        R_inv = R.T  # inverse of rot matrix is transpose
        R_inv_C = np.matmul(R_inv, C)
        extrinsics_transformed = np.concatenate((R_inv, -R_inv_C), -1)
        cam_proj_mat = np.matmul(intrinsics, extrinsics_transformed)
        cam_proj_mat_homo = np.concatenate(
            [cam_proj_mat, [np.array([0, 0, 0, 1])]])
        cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[0:3]
        world_coords_homo = np.expand_dims(_pixel_to_world_coords(
            pc, cam_proj_mat_inv), 0)
        world_coords = world_coords_homo[..., :-1][0]
        return world_coords


def save_point_cloud_to_pcd(point_cloud_array, output_file="point_cloud.pcd"):
    """
    Save a numpy array of 3D points to a PCD file.
    
    Args:
        point_cloud_array: numpy array of shape (n_cam, H, W, 3) containing XYZ coordinates
        output_file: path to save the PCD file
    """
    # Reshape the array to a list of points
    n_cam, height, width, _ = point_cloud_array.shape
    points = point_cloud_array.reshape(-1, 3)

    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()

    # Set the points
    pcd.points = o3d.utility.Vector3dVector(points)

    # Save to PCD file
    o3d.io.write_point_cloud(output_file, pcd)

    print(f"Point cloud saved to {output_file}")
    print(f"Total points: {len(pcd.points)}")