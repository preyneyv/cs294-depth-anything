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

def multi_scale_icp(source, target, init_transformation, voxel_sizes=(0.05, 0.02, 0.01), max_iters=(50, 30, 14)):
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
            max_correspondence_distance=vs * 1.5,
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

    # Apply sim(3)
    cam_trans = apply_sim3_to_pcd(pcd_cam, s, R, t)

    # Build initial 4x4 transformation
    init_T = np.eye(4)
    init_T[:3, :3] = s * R
    init_T[:3, 3] = t

    # ICP refine
    final_T, icp_res = multi_scale_icp(cam_trans, pcd_lidar, init_T)

    fitness, rmse = evaluate(pcd_cam, pcd_lidar, final_T, max_dist=noise_bound * 1.5)
    if verbose:
        print(f"After ICP: fitness = {fitness:.4f}, RMSE = {rmse:.6f}")

    cam_aligned = copy.deepcopy(pcd_cam).transform(final_T)
    metadata = {
        "s": s,
        "R": R,
        "t": t,
        "init_T": init_T,
        "final_T": final_T,
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
