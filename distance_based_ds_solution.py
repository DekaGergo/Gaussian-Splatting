import sys
import os
import numpy as np
import json
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('C:/Users/Admin/Documents/MSc/HD map/thesis/argoverse-api'))
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader

def rotmat2quat(R: np.ndarray) -> np.ndarray:
    """Convert a rotation-matrix to a quaternion in Argo's scalar-first notation (w, x, y, z)."""
    quat_xyzw = Rotation.from_matrix(R).as_quat()
    quat_wxyz = quat_scipy2argo(quat_xyzw)
    return quat_wxyz

def quat2rotmat(q: np.ndarray) -> np.ndarray:
    """Normalizes a quaternion to unit-length, then converts it into a rotation matrix.

    Note that libraries such as Scipy expect a quaternion in scalar-last [x, y, z, w] format,
    whereas at Argo we work with scalar-first [w, x, y, z] format, so we convert between the
    two formats here. We use the [w, x, y, z] order because this corresponds to the
    multidimensional complex number `w + ix + jy + kz`.

    Args:
        q: Array of shape (4,) representing (w, x, y, z) coordinates

    Returns:
        R: Array of shape (3, 3) representing a rotation matrix.
    """
    norm = np.linalg.norm(q)
    if not np.isclose(norm, 1.0, atol=1e-12):
        if np.isclose(norm, 0.0):
            raise ZeroDivisionError("Normalize quaternioning with norm=0 would lead to division by zero.")
        q /= norm

    quat_xyzw = quat_argo2scipy(q)
    return Rotation.from_quat(quat_xyzw).as_matrix()
def quat_argo2scipy(q: np.ndarray) -> np.ndarray:
    """Re-order Argoverse's scalar-first [w,x,y,z] quaternion order to Scipy's scalar-last [x,y,z,w]"""
    w, x, y, z = q
    q_scipy = np.array([x, y, z, w])
    return q_scipy
def quat_scipy2argo(q: np.ndarray) -> np.ndarray:
    """Re-order Scipy's scalar-last [x,y,z,w] quaternion order to Argoverse's scalar-first [w,x,y,z]."""
    x, y, z, w = q
    q_argo = np.array([w, x, y, z])
    return q_argo
def create_transformation_matrix(quaternion, translation):
    #quaternion_scalar_first = quaternion[3], quaternion[0], quaternion[1], quaternion[2]
    # Convert quaternion to a rotation matrix
    #rotation_matrix = Rotation.from_quat(quaternion).as_matrix()
    rotation_matrix = quat2rotmat(quaternion)
    # Create a 4x4 transformation matrix
    transformation_matrix = np.eye(4)  # Start with an identity matrix
    transformation_matrix[:3, :3] = rotation_matrix  # Set the top-left 3x3 rotation
    transformation_matrix[:3, 3] = translation  # Set the translation part

    return transformation_matrix
def matrix_to_quaternion_and_translation(matrix):
    # Step 1: Extract the rotation part (top-left 3x3)
    rotation_matrix = matrix[:3, :3]
    quaternion = rotmat2quat(rotation_matrix)
    # Step 2: Convert rotation matrix to quaternion
    #rotation = Rotation.from_matrix(rotation_matrix)
    #quaternion = rotation.as_quat(scalar_first=True)  # Get quaternion in scalar-first format [qw, qx, qy, qz]

    # Step 3: Extract the translation part (last column)
    translation = matrix[:3, 3]
    return quaternion, translation
def write_ply_from_points(filename, points):
    """
    Writes 3D points to a .ply file.

    Args:
        filename (str): The name of the output .ply file.
        points (numpy.ndarray): An array of shape (N, 3) containing 3D points (X, Y, Z).
    """

    if not isinstance(points, np.ndarray):
        raise ValueError("points must be a numpy array")
    if points.shape[1] != 3:
        raise ValueError("points array must have shape (N, 3)")

    # Header for PLY file
    header = f"""ply
            format ascii 1.0
            element vertex {len(points)}
            property float x
            property float y
            property float z
            end_header
            """

    with open(filename, 'w') as ply_file:
        # Write header
        ply_file.write(header)

        # Write points
        for point in points:
            ply_file.write(f"{point[0]} {point[1]} {point[2]}\n")

def downsample_pointcloud(points, voxel_size, distance_based):
    """
    Downsamples a point cloud based on adaptive voxel sizes according to absolute distance from the origin.
    Args:
        points (numpy.ndarray): Nx3 array representing the point cloud.
        voxel_size (float): Base voxel size for downsampling.
    Returns:
        numpy.ndarray: Processed point cloud with adaptive downsampling.
    """
    # Ensure input is a numpy array
    if isinstance(points, list):
        points = np.vstack(points)  # Convert list of arrays to a single numpy array
    if distance_based:
        # Compute absolute distance from the origin (vehicle coordinate system)
        distances = np.linalg.norm(points, axis=1)

        # Determine distance-based zones
        d_max = np.max(distances)
        d1 = d_max / 9  # Nearest zone limit
        d2 = d_max / 4  # Middle zone limit

        # Separate points into zones
        nearest_zone = points[distances < d1]  # No downsampling
        middle_zone = points[(distances >= d1) & (distances < d2)]  # Downsample with voxel_size/2
        farthest_zone = points[distances >= d2]  # Downsample with voxel_size

        def downsample(points, voxel_size):
            """Helper function to downsample points using a given voxel size."""
            if len(points) == 0:
                return points
            voxel_indices = np.floor(points / voxel_size).astype(np.int32)
            voxel_dict = {}
            for index, voxel in enumerate(voxel_indices):
                voxel_key = tuple(voxel)
                if voxel_key not in voxel_dict:
                    voxel_dict[voxel_key] = points[index]
            return np.array(list(voxel_dict.values()))

        # Apply downsampling
        downsampled_nearest = downsample(nearest_zone, voxel_size / 1.25)
        downsampled_middle = downsample(middle_zone, voxel_size / 1.15)
        downsampled_farthest = downsample(farthest_zone, voxel_size)

        # Combine all zones
        final_points = np.vstack((downsampled_nearest, downsampled_middle, downsampled_farthest))

        return final_points
    else:
        # Compute the voxel indices for each point
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)

        # Use a dictionary to store one point per voxel
        voxel_dict = {}
        for idx, voxel in enumerate(voxel_indices):
            voxel_key = tuple(voxel)  # Convert to a hashable type
            if voxel_key not in voxel_dict:
                voxel_dict[voxel_key] = points[idx]

        # Collect the downsampled points
        downsampled_points = np.array(list(voxel_dict.values()))

        return downsampled_points
def Ego2Cam_Intrinsic(index):
    # Calibration data from vehicle_calibration_info.json
    if index == 0:
        quat_RFC2Ego = [0.4967372363037331, -0.5000173904299812, 0.5031821481132892, -0.5000424513548993]
        trans_RFC2Ego = [1.624149079605374, -0.0030615876218737445, 1.3612996597210592]
        T_RFC2Ego = create_transformation_matrix(quat_RFC2Ego, trans_RFC2Ego)
        RFCcameraIntrinsic = np.array([[1402.7679150197941, 0, 981.7422076685901],
                                    [0, 1402.7679150197941, 610.7617664117456],
                                    [0, 0, 1]])
        return np.linalg.inv(T_RFC2Ego), RFCcameraIntrinsic
    elif index == 1:
        quat_RFL2Ego = [0.649746336399919, -0.6565568619233455, 0.2713495800232557, -0.2704296411591286] #[0.26880684086386686, -0.2696498529701928, 0.6598824259958552, -0.6477553727756842]
        trans_RFL2Ego = [1.4943957421933252, 0.24337693249524553, 1.3562870689337334]
        T_RFL2Ego = create_transformation_matrix(quat_RFL2Ego, trans_RFL2Ego)
        RFLcameraIntrinsic = np.array([[1405.9260829999466, 0, 965.5927801503077],
                                       [0, 1405.9260829999466, 624.872946435148],
                                       [0, 0, 1]])
        return np.linalg.inv(T_RFL2Ego), RFLcameraIntrinsic
    elif index == 2:
        quat_RFR2Ego = [0.26880684086386686, -0.2696498529701928, 0.6598824259958552, -0.6477553727756842] #[0.649746336399919, -0.6565568619233455, 0.2713495800232557, -0.2704296411591286]
        trans_RFR2Ego = [1.4932357382847226, -0.24823603366992714, 1.3587824941218138]
        T_RFR2Ego = create_transformation_matrix(quat_RFR2Ego, trans_RFR2Ego)
        RFRcameraIntrinsic = np.array([[1400.8767358013986, 0, 979.7809617196746],
                                       [0, 1400.8767358013986, 626.7470046039097],
                                       [0, 0, 1]])
        return np.linalg.inv(T_RFR2Ego), RFRcameraIntrinsic
    elif index == 3:
        quat_RRL2Ego = [0.6122928820212344, -0.6135230179644687, -0.3504444657749796, 0.35478952839900724] #[0.362369855772586, -0.3530109412576115, -0.6110974617860507, 0.6087949204604265]
        trans_RRL2Ego = [1.1331683594454462, 0.22922319415971137, 1.3586830041554878]
        T_RRL2Ego = create_transformation_matrix(quat_RRL2Ego, trans_RRL2Ego)
        RRLcameraIntrinsic = np.array([[1405.2875676247684, 0, 991.0536701398403],
                                       [0, 1405.2875676247684, 614.6572646050513],
                                       [0, 0, 1]])
        return np.linalg.inv(T_RRL2Ego), RRLcameraIntrinsic
    elif index == 4:
        quat_RRR2Ego = [0.362369855772586, -0.3530109412576115, -0.6110974617860507, 0.6087949204604265] # [0.6122928820212344, -0.6135230179644687, -0.3504444657749796, 0.35478952839900724]
        trans_RRR2Ego = [1.1269278369381142, -0.23261977473744602, 1.3538847618134586]
        T_RRR2Ego = create_transformation_matrix(quat_RRR2Ego, trans_RRR2Ego)
        RRRcameraIntrinsic = np.array([[1398.8981635526468, 0, 956.7924764366256],
                                       [0, 1398.8981635526468, 580.9004713513332],
                                       [0, 0, 1]])
        return np.linalg.inv(T_RRR2Ego), RRRcameraIntrinsic
    elif index == 5:
        quat_RSL2Ego = [0.7030736181975932, -0.7057676075277391, -0.06103952791523889, 0.06207855983349632] #[0.06488116790963043, -0.059474102879727396, -0.7084230059868283, 0.7002785943650709]
        trans_RSL2Ego = [1.3054465436889988, 0.2545301570601615, 1.3564117670535663]
        T_RSL2Ego = create_transformation_matrix(quat_RSL2Ego, trans_RSL2Ego)
        RSLcameraIntrinsic = np.array([[1406.668708764027, 0, 969.1274587919514],
                                       [0, 1406.668708764027, 622.6176623508292],
                                       [0, 0, 1]])
        return np.linalg.inv(T_RSL2Ego), RSLcameraIntrinsic
    elif index == 6:
        quat_RSR2Ego = [0.06488116790963043, -0.059474102879727396, -0.7084230059868283, 0.7002785943650709] #[0.7030736181975932, -0.7057676075277391, -0.06103952791523889, 0.06207855983349632]
        trans_RSR2Ego = [1.3047278980866743, -0.2514982343104796, 1.3540564085034235]
        T_RSR2Ego = create_transformation_matrix(quat_RSR2Ego, trans_RSR2Ego)
        RSRcameraIntrinsic = np.array([[1404.8619028377182, 0, 980.2938136368103],
                                       [0, 1404.8619028377182, 619.1984107180808],
                                       [0, 0, 1]])
        return np.linalg.inv(T_RSR2Ego), RSRcameraIntrinsic
def downsample_image_points(points: np.ndarray, base_grid_size: int) -> np.ndarray:
    """
    Downsample 2D points based on a grid size.
    Parameters:
        points (np.ndarray): Array of shape (N, 2) containing 2D points.
        grid_size (int): Size of the grid cell (e.g., 3 means a 3x3 pixel area).
    Returns:
        np.ndarray: Boolean mask indicating which points to keep.
    """
    if len(points) == 0:
        return np.array([], dtype=bool)
        # Extract depth values
    depths = points[:, 2]
    # Define depth zones
    max_depth = np.max(depths)
    d1 = max_depth / 16  # Near
    d2 = max_depth / 9
    d3 = max_depth / 4  # Mid
    #d4 = max_depth / 2.5
    # Create masks for each zone
    near_mask = depths < d1
    near_mid_mask = (depths >= d1) & (depths < d2)
    far_mid_mask = (depths >= d2) & (depths < d3)
    far_mask = depths >= d3

    # Helper function for downsampling in a grid
    def downsample_zone(zone_points, grid_size, indices):
        if len(zone_points) == 0:
            return np.array([], dtype=int)
        pixel_coords = np.floor(zone_points[:, :2] / grid_size).astype(int)  # Use only x, y
        unique_cells = {}
        keep_indices = []
        for i, (gx, gy) in enumerate(pixel_coords):
            cell_key = (gx, gy)
            if cell_key not in unique_cells:
                unique_cells[cell_key] = indices[i]
                keep_indices.append(indices[i])
        return np.array(keep_indices)

    # Apply adaptive downsampling
    keep_near = downsample_zone(points[near_mask], base_grid_size // 2.5, np.where(near_mask)[0])
    keep_near_mid = downsample_zone(points[near_mid_mask], base_grid_size // 2, np.where(near_mid_mask)[0])
    keep_far_mid = downsample_zone(points[far_mid_mask], base_grid_size // 1.5, np.where(far_mid_mask)[0])
    keep_far = downsample_zone(points[far_mask], base_grid_size, np.where(far_mask)[0])

    # Combine and generate final mask
    keep_indices = np.concatenate([keep_near, keep_near_mid, keep_far_mid, keep_far])
    mask = np.zeros(len(points), dtype=bool)
    mask[keep_indices] = True

    return mask

def timestamp(input_string):
    split_string = input_string.replace('_', ' ')
    parts = split_string.split()
    last_part = parts[-1].rsplit('.', 1)[0]  # Remove the .jpg extension
    return last_part

def find_pose(image_filename, json_folder):
    """Finds the corresponding JSON file for the given image and returns quaternion & translation."""
    img_timestamp = timestamp(image_filename)  # Extract timestamp from image filename

    # Look for a JSON file containing the same timestamp
    for filename in os.listdir(json_folder):
        if img_timestamp in filename:
            json_path = os.path.join(json_folder, filename)
            with open(json_path, 'r') as json_file:
                pose_data = json.load(json_file)  # Load JSON data

            # Extract quaternion (rotation) and translation
            quaternion = pose_data.get("rotation", None)
            translation = pose_data.get("translation", None)

            return quaternion, translation  # Return structured pose data

    return None, None  # No matching JSON found

rootDir = 'C:/Users/Admin/Documents/MSc/HD map/argoverse/argoverse-tracking/sample/'
datasetPath = 'C:/Users/Admin/Documents/MSc/HD map/argoverse/argoverse-tracking/sample/c6911883-1843-3727-8eaa-41dc8cda8993/'
posePath = datasetPath + 'poses/'
argoverseLoader = ArgoverseTrackingLoader(rootDir)
logID = 'c6911883-1843-3727-8eaa-41dc8cda8993'  # argoverse_loader.log_list[55]
argoverseData = argoverseLoader.get(logID)

imageID = 0
imageIter = 0

worldPointsList = []
existing3Dpoints = []
cameraLength = 1
lidarLength = 3
imageLength = lidarLength * 3
voxelSize = 2
gridSize = 50 # distance based filter!!!!!!
error = 0
distanceLimit = 200
camera_poses = open("camera_poses.txt", "w")
points3DFile = open("points3D.txt", "w")
imagesFile = open("images.txt", "w")

first_pose_Ego2World = argoverseData.get_pose(0) # rotation as quaternions in scalar-first: w, x, y, z
T_Ego2World_first = first_pose_Ego2World.transform_matrix

for i in range(lidarLength):
    pcEgo = argoverseData.get_lidar(i)
    pcEgo_ds = downsample_pointcloud(pcEgo, voxelSize, True)
    pose_Ego2World = argoverseData.get_pose(i)
    T_Ego2World = pose_Ego2World.transform_matrix
    T_Ego2World[:3, 3] = T_Ego2World[:3, 3] - T_Ego2World_first[:3, 3]
    pcEgoHom = np.hstack([pcEgo, np.ones((pcEgo.shape[0], 1))])
    pcWorld = np.dot(pcEgoHom, T_Ego2World.T)
    pcWorld = pcWorld[:, :3]  # Remove homogeneous coordinate
    worldPointsList.append(pcWorld)

pcWorld = np.vstack(worldPointsList)
pcWorld_ds = downsample_pointcloud(pcWorld, voxelSize, False)
pcWorld = np.array(pcWorld, dtype=np.float32)  # Convert points to float32
indices = np.arange(pcWorld.shape[0], dtype=int).reshape(-1, 1) # Attach indices to points
pcIndexed = np.hstack((indices, pcWorld)).astype(object)  # Convert to object dtype
pcIndexed[:, 0] = pcIndexed[:, 0].astype(np.int32)  # Convert first column to int
for j in range(cameraLength):
    camera = argoverseLoader.CAMERA_LIST[j]
    print('Camera: ', camera)
    #imagePath = datasetPath + str(camera) + '/'
    T_Ego2Cam, cameraIntrinsic = Ego2Cam_Intrinsic(j)
    cameraID = j + 1

    imageList = argoverseData.get_image_list(camera)
    imageFileNames = [os.path.basename(imgPath) for imgPath in imageList]
    for i in range(imageLength):
        imageID = imageIter + i
        img = argoverseData.get_image(i, camera=camera)
        quat_Ego2World, trans_Ego2World = find_pose(imageFileNames[i], posePath)
        T_Ego2World = create_transformation_matrix(quat_Ego2World, trans_Ego2World)
        T_Ego2World[:3, 3] = T_Ego2World[:3, 3] - T_Ego2World_first[:3, 3]
        T_World2Ego = np.linalg.inv(T_Ego2World)
        T_World2Cam = np.dot(T_Ego2Cam, T_World2Ego)
        quat_World2Cam, trans_World2Cam = matrix_to_quaternion_and_translation(T_World2Cam)
        imagesLine1 = [str(imageID), " ", str(quat_World2Cam[0]), " ", str(quat_World2Cam[1]), " ",
                       str(quat_World2Cam[2]), " ", str(quat_World2Cam[3]), " ", str(trans_World2Cam[0]), " ",
                       str(trans_World2Cam[1]), " ", str(trans_World2Cam[2]), " ", str(cameraID), " ",
                       imageFileNames[i], "\n"]
        imagesFile.writelines(imagesLine1)
        print(imageFileNames[i])

        # Transform world points to camera frame
        pcWorldHom = np.hstack([pcWorld, np.ones((pcWorld.shape[0], 1))])
        pcInCamHom = np.dot(pcWorldHom, T_World2Cam.T)
        pcInCam = pcInCamHom[:, :3]  # Remove homogeneous coordinate
        pointsInFront = (pcInCam[:, 2] >= 0) & (pcInCam[:, 2] < distanceLimit)
        pcInCam = pcInCam[pointsInFront]
        # Project using intrinsic matrix
        pointsOnImage = np.dot(pcInCam, cameraIntrinsic.T)
        points2D = pointsOnImage[:, :2] / pointsOnImage[:, 2:3]  # Normalize by Z
        depthCoordinates = np.hstack((points2D, pointsOnImage[:, 2:3]))
        # Check which points are within image bounds
        image_height, image_width = img.shape[:2]
        inImage = (points2D[:, 0] >= 0) & (points2D[:, 0] < image_width) & \
                  (points2D[:, 1] >= 0) & (points2D[:, 1] < image_height)

        validInImage = points2D[inImage]
        validInDepth = depthCoordinates[inImage]
        filteredImagePoints = downsample_image_points(validInDepth, gridSize)
        filteredInImage = validInImage[filteredImagePoints]
        validPoints3D = pcIndexed[pointsInFront][inImage][filteredImagePoints]
        if existing3Dpoints:
            existing3DpointsArray = np.array(existing3Dpoints)
            newPointsMask = np.isin(validPoints3D[:, 0], existing3DpointsArray[:, 0], invert=True)
            newPoints3D = validPoints3D[newPointsMask]
            newImagePoints = filteredInImage[newPointsMask]
        else:
            newPoints3D = validPoints3D
            newImagePoints = filteredInImage

        point3DID = newPoints3D[:, 0]
        X = newPoints3D[:, 1]
        Y = newPoints3D[:, 2]
        Z = newPoints3D[:, 3]
        RGBvalues = np.zeros((len(newImagePoints), 3), dtype=np.uint8)
        for idx, (x, y) in enumerate(newImagePoints.astype(int)):
            #images_line2 = [str(x), " ", str(y), " ", str(point3d_id), " "]
            #images_file.writelines(images_line2)
            RGBvalues[idx] = img[y, x]
            BlueValue = RGBvalues[idx, 0]  # First channel
            GreenValue = RGBvalues[idx, 1]  # Second channel
            RedValue = RGBvalues[idx, 2]  # Third channel
            points3DLine = [str(point3DID[idx]), " ", str(X[idx]), " ", str(Y[idx]), " ", str(Z[idx]), " ",
                            str(RedValue), " ", str(GreenValue), " ", str(BlueValue), " ",
                            str(0), " ", str(imageID), " ", str(idx), "\n"]
            points3DFile.writelines(points3DLine)
        for point3D in newPoints3D:
            existing3Dpoints.append(point3D.tolist())
        imagesFile.writelines("\n")
    imageIter = imageID + 1

plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.scatter(filteredInImage[:, 0], filteredInImage[:, 1], c='r', s=1)
plt.axis('off')
plt.show()