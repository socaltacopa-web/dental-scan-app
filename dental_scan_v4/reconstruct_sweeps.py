#!/usr/bin/env python3
"""
Dental Scan V4 - local sweep reconstruction.

Key V4 change:
Adjacent frame pairs are no longer placed next to one another with artificial offsets.
Relative camera poses are chained so all points from a sweep are expressed in one
sweep-local coordinate system. Translation scale remains arbitrary because a monocular
phone camera does not provide metric scale here.

Outputs per sweep:
  frames/
  masks/
  confidence/
  dense_<sweep>.ply
  sparse_<sweep>.ply
  cameras_<sweep>.json
  report.txt
"""

from __future__ import annotations
import argparse, base64, json, math, re
from pathlib import Path
import cv2
import numpy as np


def decode_data_url(data_url: str) -> bytes:
    m = re.match(r"^data:[^;]+;base64,(.*)$", data_url, re.DOTALL)
    if not m:
        raise ValueError("Unsupported data URL")
    return base64.b64decode(m.group(1))


def write_ply(path: Path, pts: np.ndarray, colors: np.ndarray | None = None):
    pts = np.asarray(pts, dtype=np.float64)
    good = np.all(np.isfinite(pts), axis=1)
    pts = pts[good]
    if colors is not None:
        colors = np.asarray(colors, dtype=np.uint8)[good]
        if len(colors) != len(pts):
            colors = None

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if colors is None:
            for x, y, z in pts:
                f.write(f"{x:.8f} {y:.8f} {z:.8f}\n")
        else:
            for (x, y, z), (r, g, b) in zip(pts, colors):
                f.write(f"{x:.8f} {y:.8f} {z:.8f} {int(r)} {int(g)} {int(b)}\n")


def tooth_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    b, g, r = cv2.split(img)
    redness = r.astype(np.int16) - ((g.astype(np.int16) + b.astype(np.int16)) // 2)
    mask = ((v > 105) & (s < 120) & (redness < 58)).astype(np.uint8) * 255
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return cv2.GaussianBlur(mask, (5, 5), 0)


def detector_and_norm():
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=5500), cv2.NORM_L2, "SIFT"
    return cv2.ORB_create(nfeatures=6500), cv2.NORM_HAMMING, "ORB"


def ratio_matches(d1, d2, norm):
    if d1 is None or d2 is None:
        return []
    bf = cv2.BFMatcher(norm)
    out = []
    for pair in bf.knnMatch(d1, d2, k=2):
        if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance:
            out.append(pair[0])
    return out


def K_approx(w, h):
    f = 0.95 * max(w, h)
    return np.array([[f,0,w/2],[0,f,h/2],[0,0,1]], dtype=np.float64)


def estimate_relative_pose(kp1, kp2, matches, K):
    if len(matches) < 14:
        return None
    p1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    p2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    E, mask = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.25)
    if E is None:
        return None
    _, R, t, pmask = cv2.recoverPose(E, p1, p2, K, mask=mask)
    keep = pmask.ravel() > 0
    p1, p2 = p1[keep], p2[keep]
    if len(p1) < 10:
        return None
    return R, t.reshape(3,1), p1, p2


def dense_flow_points(gray1, gray2, mask1, mask2, stride=7):
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 0.5, 4, 25, 4, 7, 1.5, 0
    )
    h, w = gray1.shape
    p1, p2, conf = [], [], []
    for y in range(stride, h-stride, stride):
        for x in range(stride, w-stride, stride):
            if mask1[y,x] < 90:
                continue
            dx, dy = flow[y,x]
            x2, y2 = x + float(dx), y + float(dy)
            if not (1 <= x2 < w-1 and 1 <= y2 < h-1):
                continue
            ix2, iy2 = int(round(x2)), int(round(y2))
            if mask2[iy2, ix2] < 70:
                continue
            mag = math.hypot(dx, dy)
            if mag < 0.35 or mag > 80:
                continue
            photo = abs(int(gray1[y,x]) - int(gray2[iy2,ix2]))
            if photo > 40:
                continue
            p1.append((x,y)); p2.append((x2,y2))
            conf.append(max(0.0, 1.0-photo/40.0))
    return (
        np.asarray(p1, np.float32),
        np.asarray(p2, np.float32),
        np.asarray(conf, np.float32),
        flow
    )


def triangulate_current_cam(K, Rrel, trel, p1, p2):
    if len(p1) == 0:
        return np.empty((0,3)), np.empty((0,), dtype=bool)
    P1 = K @ np.hstack([np.eye(3), np.zeros((3,1))])
    P2 = K @ np.hstack([Rrel, trel])
    X4 = cv2.triangulatePoints(P1, P2, p1.T, p2.T)
    valid_d = np.abs(X4[3]) > 1e-9
    X = np.zeros((len(X4[3]),3), np.float64)
    X[valid_d] = (X4[:3,valid_d] / X4[3,valid_d]).T
    X2 = (Rrel @ X.T + trel).T
    keep = (
        valid_d &
        np.all(np.isfinite(X), axis=1) &
        (X[:,2] > 0) &
        (X2[:,2] > 0) &
        (np.linalg.norm(X, axis=1) < 150)
    )
    return X[keep], keep


def cam_to_world(Xcam, Rcw, tcw):
    # Xcam = Rcw * Xworld + tcw
    return (Rcw.T @ (Xcam.T - tcw)).T


def color_points(img, pts):
    h,w = img.shape[:2]
    out=[]
    for x,y in pts:
        ix=int(np.clip(round(x),0,w-1)); iy=int(np.clip(round(y),0,h-1))
        b,g,r=img[iy,ix]
        out.append([r,g,b])
    return np.asarray(out,np.uint8)


def robust_trim(pts, cols=None, percentile=97):
    if len(pts) < 20:
        return pts, cols
    c=np.median(pts,axis=0)
    d=np.linalg.norm(pts-c,axis=1)
    lim=np.percentile(d,percentile)
    keep=d<=lim
    return pts[keep], None if cols is None else cols[keep]


def reconstruct_sweep(images, out_dir):
    h,w=images[0].shape[:2]
    images=[cv2.resize(im,(w,h)) if im.shape[:2]!=(h,w) else im for im in images]
    grays=[cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) for im in images]
    masks=[tooth_mask(im) for im in images]
    K=K_approx(w,h)
    detector,norm,det_name=detector_and_norm()
    feats=[detector.detectAndCompute(g,m) for g,m in zip(grays,masks)]

    # World-to-camera pose of frame 0.
    Rcw=np.eye(3)
    tcw=np.zeros((3,1))
    cameras=[{"frame":0,"Rcw":Rcw.tolist(),"tcw":tcw.ravel().tolist()}]

    dense_world=[]; dense_colors=[]; sparse_world=[]
    report=[f"Detector: {det_name}",f"Frames: {len(images)}"]

    for i in range(len(images)-1):
        kp1,d1=feats[i]; kp2,d2=feats[i+1]
        matches=ratio_matches(d1,d2,norm)
        pose=estimate_relative_pose(kp1,kp2,matches,K)

        if pose is None:
            report.append(f"Pair {i:02d}->{i+1:02d}: pose failed ({len(matches)} matches)")
            # Keep previous pose if pair fails.
            cameras.append({"frame":i+1,"Rcw":Rcw.tolist(),"tcw":tcw.ravel().tolist(),"pose_failed":True})
            continue

        Rrel,trel,sp1,sp2=pose

        # Sparse pair in camera-i coordinates, then convert to sweep world.
        sp_cam, _ = triangulate_current_cam(K,Rrel,trel,sp1,sp2)
        if len(sp_cam):
            sp_world=cam_to_world(sp_cam,Rcw,tcw)
            sparse_world.append(sp_world)

        fp1,fp2,fconf,flow=dense_flow_points(grays[i],grays[i+1],masks[i],masks[i+1])
        dp_cam, keep = triangulate_current_cam(K,Rrel,trel,fp1,fp2)
        if len(dp_cam):
            kp=fp1[keep]
            conf=fconf[keep]
            good=conf>0.28
            dp_cam=dp_cam[good]; kp=kp[good]
            if len(dp_cam):
                dp_world=cam_to_world(dp_cam,Rcw,tcw)
                dp_world, cols = robust_trim(dp_world, color_points(images[i],kp), 96)
                dense_world.append(dp_world); dense_colors.append(cols)

        # Chain the camera pose.
        # X_(i+1) = Rrel X_i + trel
        # X_i     = Rcw X_world + tcw
        Rcw = Rrel @ Rcw
        tcw = Rrel @ tcw + trel
        cameras.append({"frame":i+1,"Rcw":Rcw.tolist(),"tcw":tcw.ravel().tolist()})

        cv2.imwrite(str(out_dir/"masks"/f"mask_{i:03d}.png"),masks[i])
        report.append(
            f"Pair {i:02d}->{i+1:02d}: {len(matches)} matches, "
            f"{len(sp_cam)} sparse, {len(dp_cam)} dense accepted"
        )

    sparse=np.vstack(sparse_world) if sparse_world else np.empty((0,3))
    dense=np.vstack(dense_world) if dense_world else np.empty((0,3))
    colors=np.vstack(dense_colors) if dense_colors else None
    dense,colors=robust_trim(dense,colors,98)
    sparse,_=robust_trim(sparse,None,98)

    if len(sparse): write_ply(out_dir/f"sparse_{out_dir.name}.ply",sparse)
    if len(dense): write_ply(out_dir/f"dense_{out_dir.name}.ply",dense,colors)

    (out_dir/f"cameras_{out_dir.name}.json").write_text(
        json.dumps({"K":K.tolist(),"cameras":cameras},indent=2),encoding="utf-8"
    )
    report += [
        "",f"Final sparse points: {len(sparse)}",f"Final dense points: {len(dense)}",
        "",
        "Important:",
        "- All points in this sweep now share one chained sweep-local coordinate system.",
        "- Translation scale is still arbitrary and can drift.",
        "- No metric dimensions should be read from this cloud."
    ]
    (out_dir/"report.txt").write_text("\n".join(report),encoding="utf-8")
    return len(dense),len(sparse)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("package",type=Path)
    ap.add_argument("--output",type=Path,default=Path("reconstruction_v4"))
    args=ap.parse_args()
    package=json.loads(args.package.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True,exist_ok=True)
    summary=[]

    for sweep in package.get("sweeps",[]):
        if sweep.get("missing") or not sweep.get("frames"):
            continue
        out=args.output/sweep["key"]
        (out/"frames").mkdir(parents=True,exist_ok=True)
        (out/"masks").mkdir(parents=True,exist_ok=True)

        imgs=[]
        for i,frame in enumerate(sweep["frames"]):
            p=out/"frames"/f"frame_{i:03d}.jpg"
            p.write_bytes(decode_data_url(frame["dataUrl"]))
            im=cv2.imread(str(p))
            if im is not None: imgs.append(im)
        if len(imgs)<2:
            continue
        dense,sparse=reconstruct_sweep(imgs,out)
        summary.append(f'{sweep["key"]}: {dense} dense, {sparse} sparse')

    (args.output/"summary.txt").write_text("\n".join(summary),encoding="utf-8")
    print("\n".join(summary) if summary else "No completed sweeps reconstructed.")


if __name__=="__main__":
    main()
