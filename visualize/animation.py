import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from moviepy import VideoFileClip, AudioFileClip


def plot_animation(
    audio_path: str,
    plot_dir: str,
    prediction: np.ndarray,
    ground_truth: np.ndarray | None = None,
    sample_time: tuple[int, int] | None = None,
    fps: int = 30,
    dataset: str = "mosa",
) -> None:
    """
    Generate a 3D skeleton animation of predicted vs. ground-truth violin performance
    and combine it with the audio.

    Args:
        audio_path:     Path to input audio file.
        output_path:    Path where the final video will be saved.
        prediction:     np.ndarray of shape (T, J, 3) containing predicted 3D joint coords.
        ground_truth:   np.ndarray of shape (T, J, 3) containing ground-truth joint coords.
        sample_time:    Optional (start_sec, end_sec) to trim both audio and keypoints.
        fps:            Frames per second for animation and output video.
        dataset:        "mosa" or other. Controls skeleton layout and view framing.
    """
    # Trim keypoints and audio if requested
    if sample_time is not None:
        start, end = sample_time
        idx_start = start * fps
        idx_end = end * fps
        prediction = prediction[idx_start:idx_end]
        if ground_truth is not None:
            ground_truth = ground_truth[idx_start:idx_end]
        audio_clip = AudioFileClip(audio_path).subclip(start, end)
    else:
        audio_clip = AudioFileClip(audio_path)

    # Render 3D skeleton animation to a temporary MP4
    temp_video = Path("temp_animation.mp4")
    render_skeleton_animation(
        fps=fps,
        plot_path=str(temp_video),
        azim=100,
        prediction=prediction,
        ground_truth=ground_truth,
        dataset=dataset,
    )

    # Attach audio and save final video
    audio_name = audio_path.split("/")[-1]
    audio_name, ext = os.path.splitext(audio_name)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir /  f"{audio_name}.mp4"
    video_clip = VideoFileClip(str(temp_video))
    video_clip = video_clip.with_audio(audio_clip)
    video_clip.write_videofile(plot_path, fps=fps)

    # Clean up temporary file
    temp_video.unlink(missing_ok=True)

def render_skeleton_animation(
    fps: int,
    plot_path: str,
    azim: float,
    prediction: np.ndarray,
    ground_truth: np.ndarray | None = None,
    dataset: str = "mosa",
) -> None:
    """
    Render a 3D skeleton animation (prediction vs. ground_truth) and save as MP4.

    Args:
        fps:            Frames per second for the animation.
        output_path:    Path ending in .mp4.
        azim:           Azimuth angle for the 3D view (degrees).
        prediction:     np.ndarray of shape (T, J, 3) with predicted joint coords.
        ground_truth:   Optional np.ndarray of same shape for ground truth.
        dataset:        "mosa" or other, to select skeleton layout and view settings.
    """
    # Define skeleton edges and colors
    if dataset.lower() == "mosa":
        edges = [
            (0, 1), (1, 2), (2, 3), (2, 4),
            (1, 5), (5, 6), (6, 7), (7, 8),
            (1, 9), (9, 10), (10, 11), (11, 12),
            (3, 13), (13, 14), (14, 15),
            (4, 16), (16, 17), (17, 18),
            (19, 20), (21, 22),
        ]
        colors = [
            "r", "r", "r", "r",
            "b", "b", "b", "b",
            "g", "g", "g", "g",
            "c", "c", "c",
            "y", "y", "y",
            "k", "k",
        ]
    else:
        edges = [
            (0, 1), (0, 3), (0, 5),
            (1, 2), (3, 4),
            (5, 6), (6, 7), (7, 8),
            (6, 9), (9, 10), (10, 11),
            (6, 12), (12, 13), (13, 14),
        ]
        right_joints = [1, 2, 12, 13, 14]
        colors = None
        # Slightly lift skeleton off the ground plane for visibility
        prediction[..., 2] += 0.3
        if ground_truth is not None:
            ground_truth[..., 2] += 0.3

    # Prepare pose dictionaries
    poses = {"Prediction": prediction}
    if ground_truth is not None:
        poses["GroundTruth"] = ground_truth
    pose_list = list(poses.values())
    
    num_frames = prediction.shape[0]
    num_views = len(pose_list)
    limit = num_frames

    fig = plt.figure(figsize=(6 * num_views, 6))
    axes_3d = []
    lines_3d: list[list] = []

    for idx, (title, pose_data) in enumerate(poses.items()):
        ax = fig.add_subplot(1, num_views, idx + 1, projection="3d")
        if dataset.lower() == "mosa":
            ax.view_init(elev=40, azim=90)
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            ax.set_zlabel("y")
            ax.set_xlim(600, -600)
            ax.set_ylim(600, -600)
            ax.set_zlim(-400, 800)
            ax.dist = 12
        else:
            radius = 1.7
            ax.view_init(elev=15.0, azim=azim)
            ax.set_xlim3d([-radius / 2, radius / 2])
            ax.set_ylim3d([-radius / 2, radius / 2])
            ax.set_zlim3d([0, radius])
            ax.set_aspect("equal")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.dist = 7.5

        ax.set_title(title, fontsize=15)
        ax.grid(False)
        ax.axis('off')
        axes_3d.append(ax)
        lines_3d.append([])
    
    plt.tight_layout()
    plt.close(fig)  # Matplotlib will open it in FuncAnimation
    
    def update_frame(frame_idx: int) -> None:
        """
        Update the skeleton lines for frame `frame_idx`.
        """
        for view_idx, ax in enumerate(axes_3d):
            pos = pose_list[view_idx][frame_idx]

            if not lines_3d[view_idx]:
                # First frame: create line segments
                for edge_idx, (j0, j1) in enumerate(edges):
                    if dataset.lower() == "mosa":
                        color = colors[edge_idx]
                        xs = [pos[j0, 0], pos[j1, 0]]
                        ys = [pos[j0, 2], pos[j1, 2]]
                        zs = [pos[j0, 1], pos[j1, 1]]
                    else:
                        color = "red" if j0 in right_joints and j1 in right_joints else "black"
                        xs = [pos[j0, 0], pos[j1, 0]]
                        ys = [pos[j0, 1], pos[j1, 1]]
                        zs = [pos[j0, 2], pos[j1, 2]]

                    line = ax.plot(xs, ys, zs, c=color, zdir="z")
                    lines_3d[view_idx].append(line[0])
            else:
                # Subsequent frames: update existing line segments
                for edge_idx, (j0, j1) in enumerate(edges):
                    line = lines_3d[view_idx][edge_idx]
                    if dataset.lower() == "mosa":
                        xs = [pos[j0, 0], pos[j1, 0]]
                        ys = [pos[j0, 2], pos[j1, 2]]
                        zs = [pos[j0, 1], pos[j1, 1]]
                    else:
                        xs = [pos[j0, 0], pos[j1, 0]]
                        ys = [pos[j0, 1], pos[j1, 1]]
                        zs = [pos[j0, 2], pos[j1, 2]]

                    line.set_data(xs, ys)
                    line.set_3d_properties(zs, zdir="z")

        if frame_idx % 1000 == 0:
            print(f"Frame {frame_idx}/{limit}")
          
    # Create and save the animation
    anim = FuncAnimation(
        fig,
        update_frame,
        frames=range(limit),
        interval=1000 / fps,
        repeat=False,
    )

    Writer = writers["ffmpeg"]
    writer = Writer(fps=fps, metadata={}, bitrate=3000)
    anim.save(plot_path, writer=writer)
    plt.close(fig)