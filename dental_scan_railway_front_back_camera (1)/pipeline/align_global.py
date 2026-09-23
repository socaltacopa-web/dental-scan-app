#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


DEFAULT_EDGES = [
    ("bite_registration", "front_arc"),
    ("front_arc", "right_outer"),
    ("front_arc", "left_outer"),
    ("front_arc", "upper_bridge"),
    ("front_arc", "lower_bridge"),
    ("upper_bridge", "upper_biting"),
    ("upper_bridge", "upper_inside"),
    ("lower_bridge", "lower_biting"),
    ("lower_bridge", "lower_inside"),
    ("bite_registration", "upper_bridge"),
    ("bite_registration", "lower_bridge"),
]


def load_cloud(path: Path):
    cloud = o3d.io.read_point_cloud(str(path))
    if len(cloud.points) == 0:
        raise ValueError(f"Empty point cloud: {path}")
    return cloud


def preprocess(pcd, voxel):
    down = pcd.voxel_down_sample(voxel)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel * 3.0,
            max_nn=40,
        )
    )

    feat = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel * 6.0,
            max_nn=100,
        ),
    )

    return down, feat


def global_register(source, target, voxel):
    s_down, s_feat = preprocess(source, voxel)
    t_down, t_feat = preprocess(target, voxel)

    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        s_down,
        t_down,
        s_feat,
        t_feat,
        True,
        voxel * 2.0,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.85),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * 2.0),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(70000, 0.999),
    )


def refine(source, target, initial, voxel):
    s = source.voxel_down_sample(voxel / 2)
    t = target.voxel_down_sample(voxel / 2)

    s.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=40)
    )
    t.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=40)
    )

    return o3d.pipelines.registration.registration_icp(
        s,
        t,
        voxel * 1.3,
        initial,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80),
    )


def information_matrix(source, target, T, voxel):
    return o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source,
        target,
        voxel * 1.5,
        T,
    )


def intended_pairs(package_path: Path | None):
    if package_path and package_path.exists():
        try:
            obj = json.loads(package_path.read_text(encoding="utf-8"))
            pairs = obj.get("captureGraph", {}).get("intendedOverlaps")
            if pairs:
                return [tuple(x) for x in pairs]
        except Exception:
            pass

    return DEFAULT_EDGES


def locate_clouds(recon_dir: Path):
    clouds = {}

    for sub in recon_dir.iterdir():
        if not sub.is_dir():
            continue

        p = sub / f"dense_{sub.name}.ply"

        if p.exists():
            clouds[sub.name] = p

    return clouds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reconstruction_dir", type=Path)
    ap.add_argument("--package", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=Path("global"))
    ap.add_argument("--voxel", type=float, default=0.035)
    ap.add_argument("--min-fitness", type=float, default=0.18)
    ap.add_argument("--mesh", action="store_true")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    paths = locate_clouds(args.reconstruction_dir)

    if len(paths) < 2:
        raise SystemExit("Need at least two dense sweep clouds.")

    clouds = {}
    normalization = {}

    for name, path in paths.items():
        raw = load_cloud(path)
        pts = np.asarray(raw.points)

        center = np.median(pts, axis=0)
        d = np.linalg.norm(pts - center, axis=1)
        scale = float(np.percentile(d, 75))

        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0

        q = o3d.geometry.PointCloud(raw)
        q.translate(-center)
        q.scale(1.0 / scale, center=(0, 0, 0))

        clouds[name] = q
        normalization[name] = {
            "center": center.tolist(),
            "scale": scale,
        }

    anchor = (
        "bite_registration"
        if "bite_registration" in clouds
        else sorted(clouds)[0]
    )

    names = [anchor] + [n for n in sorted(clouds) if n != anchor]
    idx = {n: i for i, n in enumerate(names)}

    pairs = [
        p for p in intended_pairs(args.package)
        if p[0] in clouds and p[1] in clouds
    ]

    edges = []
    pair_report = []

    for a, b in pairs:
        coarse = global_register(clouds[a], clouds[b], args.voxel)
        refined = refine(
            clouds[a],
            clouds[b],
            coarse.transformation,
            args.voxel,
        )

        accepted = bool(
            refined.fitness >= args.min_fitness
            and np.isfinite(refined.inlier_rmse)
        )

        pair_report.append(
            {
                "source": a,
                "target": b,
                "coarse_fitness": float(coarse.fitness),
                "coarse_rmse": float(coarse.inlier_rmse),
                "icp_fitness": float(refined.fitness),
                "icp_rmse": float(refined.inlier_rmse),
                "accepted": accepted,
                "transformation": refined.transformation.tolist(),
            }
        )

        if accepted:
            info = information_matrix(
                clouds[a],
                clouds[b],
                refined.transformation,
                args.voxel,
            )
            edges.append(
                (
                    idx[a],
                    idx[b],
                    refined.transformation,
                    info,
                    a,
                    b,
                )
            )

    if not edges:
        raise SystemExit("No sweep pair registrations passed the fitness threshold.")

    pg = o3d.pipelines.registration.PoseGraph()

    for _ in names:
        pg.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(np.eye(4))
        )

    known = {idx[anchor]: np.eye(4)}
    changed = True

    while changed:
        changed = False

        for ia, ib, T_ab, _, _, _ in edges:
            if ib in known and ia not in known:
                known[ia] = known[ib] @ T_ab
                changed = True
            elif ia in known and ib not in known:
                known[ib] = known[ia] @ np.linalg.inv(T_ab)
                changed = True

    for i in range(len(names)):
        if i in known:
            pg.nodes[i].pose = known[i]

    for ia, ib, T_ab, info, a, b in edges:
        uncertain = not (a == anchor or b == anchor)

        pg.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                ia,
                ib,
                T_ab,
                info,
                uncertain=uncertain,
            )
        )

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=args.voxel * 1.5,
        edge_prune_threshold=0.20,
        preference_loop_closure=1.5,
        reference_node=idx[anchor],
    )

    o3d.pipelines.registration.global_optimization(
        pg,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )

    merged = o3d.geometry.PointCloud()
    transforms = {}

    for i, name in enumerate(names):
        q = o3d.geometry.PointCloud(clouds[name])
        T = pg.nodes[i].pose
        q.transform(T)
        merged += q
        transforms[name] = T.tolist()

    merged = merged.voxel_down_sample(args.voxel / 2)

    if len(merged.points) > 30:
        merged, _ = merged.remove_statistical_outlier(
            nb_neighbors=30,
            std_ratio=2.2,
        )

    if len(merged.points) > 30:
        merged.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=args.voxel * 2.5,
                max_nn=50,
            )
        )

    merged_path = args.output / "merged_mouth.ply"
    o3d.io.write_point_cloud(str(merged_path), merged)

    mesh_path = None

    if args.mesh and len(merged.points) > 500:
        try:
            merged.orient_normals_consistent_tangent_plane(30)

            mesh, densities = (
                o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    merged,
                    depth=8,
                    scale=1.1,
                    linear_fit=False,
                )
            )

            densities = np.asarray(densities)

            if len(densities):
                cutoff = np.quantile(densities, 0.04)
                mesh.remove_vertices_by_mask(densities < cutoff)

            mesh = mesh.crop(merged.get_axis_aligned_bounding_box())
            mesh.compute_vertex_normals()

            mesh_path = args.output / "merged_mouth_mesh.ply"
            o3d.io.write_triangle_mesh(str(mesh_path), mesh)

        except Exception:
            mesh_path = None

    connected = set()

    for ia, ib, *_ in edges:
        connected.add(names[ia])
        connected.add(names[ib])

    report = {
        "anchor": anchor,
        "clouds_found": names,
        "connected_clouds": sorted(connected),
        "unconnected_clouds": sorted(set(names) - connected),
        "pairwise_registration": pair_report,
        "optimized_transforms": transforms,
        "normalization": normalization,
        "merged_point_count": len(merged.points),
        "merged_point_cloud": str(merged_path),
        "mesh": str(mesh_path) if mesh_path else None,
        "warnings": [
            "The reconstruction has arbitrary scale.",
            "Registration can converge to an incorrect alignment.",
            "Always inspect the original images and the merged model.",
            "This is not a clinically validated intraoral scan.",
        ],
    }

    (args.output / "alignment_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Merged point cloud: {merged_path}")


if __name__ == "__main__":
    main()
