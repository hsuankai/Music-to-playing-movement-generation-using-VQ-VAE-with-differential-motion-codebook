import numpy as np
from mir_eval.onset import f_measure
import matplotlib.pyplot as plt


# Joint indices in MOSA (for reference)
#
#  'head': 0,      'neck': 1,    'torso': 2,
#  'r_hip': 3,     'l_hip': 4,
#  'r_shoulder': 5,'r_elbow': 6, 'r_wrist': 7, 'r_finger': 8,
#  'l_shoulder': 9,'l_elbow': 10,'l_wrist': 11,'l_finger': 12,
#  'r_knee': 13,   'r_ankle': 14,'r_toe': 15,
#  'l_knee': 16,   'l_ankle': 17,'l_toe': 18,
#  'VTOP': 19,     'VBOM': 20,
#  'BTOP': 21,     'BBOM': 22

def compute_maje(
    pred: np.ndarray, targ: np.ndarray
) -> tuple[float, float]:
    """
    Mean Absolute Joint Error (MAJE).

    Args:
        pred:  Array of shape (T, J, 3)
        targ:  Array of the same shape as pred.

    Returns:
        full_maje: float = mean_abs_error over all joints/dims.
        rh_maje:   float = mean_abs_error over right‐hand joints only.
    """
    # full MAJE over all joints/dimensions
    full_maje = np.mean(np.abs(targ - pred))

    # select right‐hand columns:
    #   columns 6,7,8 correspond to r_elbow,r_wrist,r_finger
    #   columns -2,-1 correspond to BBOM, BTOP
    rh_pred = np.concatenate((pred[:, 6:9], pred[:, -2:]), axis=1)
    rh_targ = np.concatenate((targ[:, 6:9], targ[:, -2:]), axis=1)
    rh_maje = np.mean(np.abs(rh_targ - rh_pred))

    return full_maje, rh_maje

def compute_mad(pred: np.ndarray, targ: np.ndarray) -> float:
    """
    Mean Absolute Difference in right-hand accelerations (MAD).

    Args:
        pred:  Array of shape (T, J, 3)
        targ:  Array of same shape containing target keypoints.

    Returns:
        rh_mad: float = mean_abs_error between absolute accelerations of RH joints.
    """
    # select right‐hand columns same as in compute_maje
    rh_pred = np.concatenate((pred[:, 6:9], pred[:, -2:]), axis=1)
    rh_targ = np.concatenate((targ[:, 6:9], targ[:, -2:]), axis=1)

    # second derivative (acceleration)
    pred_acc = np.diff(rh_pred, n=2, axis=0)
    targ_acc = np.diff(rh_targ, n=2, axis=0)

    rh_mad = np.mean(np.abs(np.abs(targ_acc) - np.abs(pred_acc)))
    return rh_mad

def compute_velocity(data, dim=3, fps=30):
    """Compute velocity between adjacent frames

      Args:
          data:         array containing joint positions of gesture
          dim:          gesture dimensionality

      Returns:
          vel_norms:    velocities of each joint between each adjacent frame
    """
    # Flatten the array of 3d coords
    coords = data.reshape(len(data), -1)

    # First derivative of position is velocity
    vels = np.diff(coords, n=1, axis=0)
    num_vels = vels.shape[0]
    num_joints = vels.shape[1] // dim
    vel_norms = np.zeros((num_vels, num_joints))

    # calculate vector norms
    for i in range(num_vels):
        for j in range(num_joints):
            x1 = j * dim + 0
            x2 = j * dim + dim
            vel_norms[i, j] = np.linalg.norm(vels[i, x1:x2])
    # multiply on fps to compensate for the time-frame being  0.03s
    return vel_norms * fps

def compute_velocity_histogram(
    pred: np.ndarray,
    targ: np.ndarray,
    bin_width: float = 0.5,
    max_velocity: float = 50.0,
    visualize: bool = False,
    fps: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute and optionally plot histograms of joint velocity magnitudes.

    Args:
        pred:  Array of shape (T, J, 3) for predicted keypoints.
        targ:   Array of shape (T, J, 3) for target keypoints.
        bin_width:    Width of each histogram bin (default=0.5).
        max_velocity: Maximum velocity to consider (default=50).
        visualize:    If True, show a matplotlib plot.
        fps:          Frames per second (to scale velocities).

    Returns:
        pred_hist:  1D array of total counts per bin over all joints (predicted).
        targ_hist:   1D array of total counts per bin over all joints (target).
    """
    # compute per‐joint velocity norms
    pred_vel = compute_velocity(pred, fps=fps)
    targ_vel = compute_velocity(targ, fps=fps)

    # define bin edges: 0, bin_width, 2*bin_width, …, max_velocity
    bins = np.arange(0.0, max_velocity + bin_width, bin_width)

    # histogram per‐joint
    num_joints = pred_vel.shape[1]
    pred_hists = np.zeros((num_joints, len(bins) - 1), dtype=int)
    targ_hists = np.zeros((num_joints, len(bins) - 1), dtype=int)

    for j in range(num_joints):
        pred_hists[j], _ = np.histogram(pred_vel[:, j], bins=bins)
        targ_hists[j], _ = np.histogram(targ_vel[:, j], bins=bins)

    # sum across all joints → shape (bins-1,)
    pred_total = pred_hists.sum(axis=0)
    targ_total = targ_hists.sum(axis=0)

    if visualize:
        # normalize to frequencies
        pred_freq = pred_total / pred_total.sum()
        targ_freq = targ_total / targ_total.sum()

        plt.plot(bins[:-1], pred_freq, label="pred")
        plt.plot(bins[:-1], targ_freq, label="target")
        plt.xlabel("Velocity")
        plt.ylabel("Frequency")
        plt.title("Velocity Histogram (all joints)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return pred_total, targ_total

def normalize_histogram(hist: np.ndarray) -> np.ndarray:
    """
    Normalize a histogram to sum to 1.

    Args:
        hist: 1D array of counts.

    Returns:
        1D array of same shape summing to 1.
    """
    total = hist.sum()
    return hist / total if total > 0 else hist

def compute_hellinger_distance(
    pred: np.ndarray,
    targ: np.ndarray,
    bin_width: float = 0.5,
    max_velocity: float = 50.0,
    visualize: bool = False,
    fps: int = 30,
) -> float:
    """
    Compute Hellinger distance between predicted and target joint‐velocity histograms.

    Args:
        pred: Array of shape (T, J, 3) for predicted keypoints.
        targ: Array of shape (T, J, 3) for target keypoints.
        bin_width:   Bin width for histogram (default=0.5).
        max_velocity:Max velocity to consider (default=50).
        visualize:   If True, show the overlaid histograms.
        fps:         Frames per second.

    Returns:
        Hellinger distance (float).
    """
    pred_hist, targ_hist = compute_velocity_histogram(
        pred, targ, bin_width, max_velocity, visualize, fps
    )

    p_norm = normalize_histogram(pred_hist)
    t_norm = normalize_histogram(targ_hist)

    # Hellinger: sqrt(1 - sum( sqrt(p_i * t_i) ))
    return float(np.sqrt(1.0 - np.sum(np.sqrt(p_norm * t_norm))))

def compute_apd(
    pred: np.ndarray, targ: np.ndarray, fps: int = 30
) -> tuple[float, float]:
    """
    Average Pairwise Distance (APD) between 2-second non‐overlapping segments.

    Args:
        pred: Array of shape (T, J, 3) for predicted keypoints.
        targ: Array of shape (T, J, 3) for target keypoints.
        fps:  Frames per second (used to define 2‐s windows).

    Returns:
        (pred_apd, targ_apd): tuple of floats.
    """
    pred = np.concatenate((pred[:, :6], pred[:, 9:10]), axis=1)
    targ = np.concatenate((targ[:, :6], targ[:, 9:10]), axis=1)
    pred_segments = []
    targ_segments = []
    start = 0
    while start + fps*2 < len(pred):
        pred_segments.append(pred[start:start + fps*2])
        targ_segments.append(targ[start:start + fps*2])
        start += fps*2
    
    pred_segments = np.array(pred_segments)
    targ_segments = np.array(targ_segments)
    def average_pairwise(array: np.ndarray) -> float:
        """
        Compute mean(|x_i - x_j|) over all i != j, flattening each segment block.
        """
        n_seg = array.shape[0]
        total_dist = 0.0
        count = 0
        # iterate over pairs (i,j), i < j
        for i in range(n_seg):
            v_i = array[i].ravel()
            for j in range(i + 1, n_seg):
                v_j = array[j].ravel()
                total_dist += np.mean(np.abs(v_i - v_j))
                count += 1
        return total_dist / count if count > 0 else 0.0

    pred_apd = average_pairwise(pred_segments)
    targ_apd = average_pairwise(targ_segments)
    return pred_apd, targ_apd

def compute_attack_f1(
    pred: np.ndarray, targ: np.ndarray, fps: int = 30
) -> float:
    """
    “Attack” F1 score on wrist‐joint acceleration peaks.

    Steps:
      1. Extract z‐ordered wrist coords: pred[:,7] and targ[:,7].
      2. Compute velocity (diff) → sign → acceleration (diff of sign).
      3. Absolute acceleration peaks → indices / fps → attack times.
      4. Use mir_eval.onset.f_measure to compute F1 on event matches.

    Args:
        pred: Array of shape (T, J, 3) for predicted keypoints.
        targ: Array of shape (T, J, 3) for target keypoints.
        fps:  Frames per second.

    Returns:
        Mean F1 over x, y, z‐axes (float).
    """
    pred_wrist = pred[:, 7]  # shape (T, 3)
    targ_wrist = targ[:, 7]

    # compute first‐order velocity → shape (T-1, 3)
    pred_vel = np.sign(np.diff(pred_wrist, axis=0))
    targ_vel = np.sign(np.diff(targ_wrist, axis=0))

    # second‐order acc → shape (T-2, 3); take absolute
    pred_acc = np.abs(np.diff(pred_vel, axis=0))
    targ_acc = np.abs(np.diff(targ_vel, axis=0))

    # find “attack” peaks: indices where acceleration ≠ 0, converted to times
    def extract_attack_times(acc: np.ndarray) -> dict[str, np.ndarray]:
        # for each axis (x=0,y=1,z=2), get times = idx / fps
        attacks = {}
        attacks["x"] = np.where(acc[:, 0] != 0)[0] / fps
        attacks["y"] = np.where(acc[:, 1] != 0)[0] / fps
        attacks["z"] = np.where(acc[:, 2] != 0)[0] / fps
        return attacks

    pred_attacks = extract_attack_times(pred_acc)
    targ_attacks = extract_attack_times(targ_acc)

    # compute f_measure in each axis
    f1_scores = []
    for axis in ("x", "y", "z"):
        f1, _, _ = f_measure(
            targ_attacks[axis], pred_attacks[axis], window=0.1
        )
        f1_scores.append(f1)

    # average over three axes
    return float(np.mean(f1_scores))