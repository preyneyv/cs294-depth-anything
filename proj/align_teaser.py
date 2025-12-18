import copy
import numpy as np
import open3d as o3d
import teaserpp_python  # TEASER++ Python binding

def get_teaser_solver(noise_bound, estimate_scaling=True):
    """
    Create and return a TEASER++ robust registration solver with given parameters.
    - noise_bound: the bound on noise (must tune)
    - estimate_scaling: True to solve for scale
    """
    params = teaserpp_python.RobustRegistrationSolver.Params()
    params.cbar2 = 1  # default, can tune
    params.noise_bound = noise_bound
    params.estimate_scaling = estimate_scaling

    # choose a rotation estimation algorithm
    # params.rotation_estimation_algorithm = (
    #     teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    # )
    params.rotation_estimation_algorithm = teaserpp_python.RotationEstimationAlgorithm.GNC_TLS
    params.rotation_gnc_factor = 1.4
    params.rotation_max_iterations = 100
    params.rotation_cost_threshold = 1e-12

    # Other parameters can be set (depending on your data)
    solver = teaserpp_python.RobustRegistrationSolver(params)
    return solver

def compute_teaser_transform(src_corr: np.ndarray, dst_corr: np.ndarray, noise_bound, estimate_scaling=True):
    """
    Run TEASER++ solver on corresponding 3D points to estimate (s, R, t).
    src_corr, dst_corr: np.ndarray, shape (3, N) for N correspondences.
    estimate_scaling: If False, disable scale estimation (faster, more stable for debugging)
    Returns: scale (float), R (3x3), t (3,)
    """
    # Guard: ensure at least 3 correspondences
    n_corr = src_corr.shape[1]
    if n_corr < 3:
        raise ValueError(f"TEASER++ requires at least 3 correspondences, got {n_corr}")
    
    solver = get_teaser_solver(noise_bound=noise_bound, estimate_scaling=estimate_scaling)
    solver.solve(src_corr, dst_corr)
    solution = solver.getSolution()
    s = solution.scale
    R = solution.rotation
    t = solution.translation
    return s, R, t

def apply_sim3_to_pcd(pcd: o3d.geometry.PointCloud, s: float, R: np.ndarray, t: np.ndarray):
    """
    Apply similarity transform (scale, rotation, translation) to an Open3D point cloud.
    Returns a transformed copy.
    """
    pcd_copy = copy.deepcopy(pcd)
    pts = np.asarray(pcd_copy.points)  # (N,3)
    pts_t = s * (pts.dot(R.T)) + t  # note: row-vector convention
    pcd_copy.points = o3d.utility.Vector3dVector(pts_t)

    if pcd_copy.has_normals():
        normals = np.asarray(pcd_copy.normals)
        normals_t = normals.dot(R.T)
        pcd_copy.normals = o3d.utility.Vector3dVector(normals_t)

    return pcd_copy

def multi_scale_icp(source, target, init_transformation, voxel_sizes=(0.5, 0.2, 0.1), max_iters=(50, 30, 14)):
    """
    Multi-scale ICP to refine the alignment.
    """
    current_trans = init_transformation.copy()
    for vs, it in zip(voxel_sizes, max_iters):
        src_down = source.voxel_down_sample(vs)
        tgt_down = target.voxel_down_sample(vs)
        src_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vs * 2.0, max_nn=30))
        tgt_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vs * 2.0, max_nn=30))

        result_icp = o3d.pipelines.registration.registration_icp(
            src_down, tgt_down,
            max_correspondence_distance=vs * 2.0,  # Increased from 1.5 to 2.0 for better convergence
            init=current_trans,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=it),
        )
        current_trans = result_icp.transformation
    return current_trans, result_icp

def evaluate(source, target, transformation, max_dist):
    """
    Evaluate alignment: fitness, RMSE.
    """
    eval_res = o3d.pipelines.registration.evaluate_registration(
        source, target, max_dist, transformation
    )
    return eval_res.fitness, eval_res.inlier_rmse

def sim3_teaser_pipeline(pcd_cam, pcd_lidar, noise_bound, corr_src_pts, corr_dst_pts, verbose=True, estimate_scaling=True):
    """
    Full TEASER++ pipeline (without feature matching):
    - Takes in pre-computed correspondences (corr_src_pts, corr_dst_pts)
      in shape (N, 3) (or (3, N) after transpose)
    - Runs TEASER++ to get (s, R, t)
    - Applies to camera cloud
    - Refines with ICP
    - Returns aligned cloud and metadata
    
    estimate_scaling: If False, disable scale estimation (faster, more stable for debugging)
    """

    # TEASER++ requires shapes (3, N)
    src_corr = corr_src_pts.T
    dst_corr = corr_dst_pts.T
    
    # Guard: ensure at least 3 correspondences
    n_corr = src_corr.shape[1]
    if n_corr < 3:
        raise ValueError(f"TEASER++ requires at least 3 correspondences, got {n_corr}")

    s, R, t = compute_teaser_transform(src_corr, dst_corr, noise_bound=noise_bound, estimate_scaling=estimate_scaling)
    if verbose:
        print(f"TEASER++ result: scale {s}, translation {t}, rotation:\n{R}")

    # Build Sim(3) transformation matrix T_sim3
    T_sim3 = np.eye(4)
    T_sim3[:3, :3] = s * R
    T_sim3[:3, 3] = t

    # Apply sim(3) to get transformed cloud (for ICP input)
    cam_trans = apply_sim3_to_pcd(pcd_cam, s, R, t)

    # ICP refine: start from identity since cam_trans is already transformed
    # ICP will compute a rigid refinement T_icp on the already-transformed cloud
    T_icp, icp_res = multi_scale_icp(cam_trans, pcd_lidar, np.eye(4))

    # Compose transforms: T_total = T_icp @ T_sim3
    # This applies Sim(3) first, then ICP's rigid refinement
    T_total = T_icp @ T_sim3

    # Evaluate using the composed transform on original clouds
    # Use larger max_dist for evaluation (0.5m or 2 * LIDAR_PCD_VOXEL_SIZE)
    # LIDAR_PCD_VOXEL_SIZE is typically 0.2, so 2 * 0.2 = 0.4, use 0.5 for safety
    eval_max_dist = 0.5
    fitness, rmse = evaluate(pcd_cam, pcd_lidar, T_total, max_dist=eval_max_dist)
    if verbose:
        print(f"After ICP: fitness = {fitness:.4f}, RMSE = {rmse:.6f}")

    # Apply composed transform to original camera cloud
    cam_aligned = copy.deepcopy(pcd_cam).transform(T_total)
    metadata = {
        "s": s,
        "R": R,
        "t": t,
        "T_sim3": T_sim3,
        "T_icp": T_icp,
        "final_T": T_total,  # Keep for backward compatibility
        "icp_res": icp_res,
        "fitness": fitness,
        "rmse": rmse,
    }
    return cam_aligned, metadata

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", required=True, help="Camera point cloud (PLY/PCD)")
    parser.add_argument("--lidar", required=True, help="LiDAR point cloud (PLY/PCD)")
    parser.add_argument("--corr_cam", required=True, help="Corresponding camera points (numpy npy or txt)")
    parser.add_argument("--corr_lidar", required=True, help="Corresponding lidar points")
    parser.add_argument("--noise", type=float, default=0.05, help="Noise bound for TEASER")
    args = parser.parse_args()

    pcd_cam = o3d.io.read_point_cloud(args.cam)
    pcd_lidar = o3d.io.read_point_cloud(args.lidar)
    # load correspondences
    corr_cam = np.load(args.corr_cam)  # shape (N,3)
    corr_lidar = np.load(args.corr_lidar)

    cam_aligned, meta = sim3_teaser_pipeline(pcd_cam, pcd_lidar, args.noise, corr_cam, corr_lidar, verbose=True)
    print("TEASER final scale:", meta["s"])
    o3d.io.write_point_cloud("cam_aligned_teaser.ply", cam_aligned)
    print("Saved aligned cloud to cam_aligned_teaser.ply")
