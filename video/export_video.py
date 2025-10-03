import os
import shutil
import numpy as np
import imageio


def array_to_uncompressed_video(arr: np.ndarray, out_path: str, fps: int = 25) -> None:
    """
    Convert a 3D numpy array (t, y, x) to an uncompressed 8-bit greyscale video
    using FFmpeg via imageio (codec='rawvideo', pix_fmt='gray').

    Args:
        arr: numpy array with shape (t, y, x)
        out_path: output path like 'output.avi' or 'output.mkv' (avoid '.mp4')
        fps: frames per second
    """
    if arr.ndim != 3:
        raise ValueError("Expected array with shape (t, y, x).")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg not found on PATH.")
    if out_path.lower().endswith(".mp4"):
        raise ValueError("Use '.avi' or '.mkv' for uncompressed rawvideo, not '.mp4'.")

    # Normalize globally to uint8 grayscale if not already
    if arr.dtype != np.uint8:
        arr = np.asarray(arr, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        amin = float(arr.min())
        amax = float(arr.max())
        if amax > amin:
            arr = (arr - amin) / (amax - amin) * 255.0
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
        arr = np.clip(arr.round(), 0, 255).astype(np.uint8)  # (t, y, x), grayscale

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # rawvideo + gray -> uncompressed 8-bit grayscale frames
    with imageio.get_writer(
        out_path,
        fps=fps,
        codec="rawvideo",
        pixelformat="gray",
        macro_block_size=None,
    ) as writer:
        for frame in arr:
            # frame shape (y, x), dtype uint8
            writer.append_data(frame)

if __name__ == "__main__":
    # Example
    for i in range(14):
        data = np.load(rf"/home/mawa/nas_a/Marvin/chicken_13_05_2025/kilosort4/4px_20Hz_shuffle_460nm_idx_2/cell_{i}/kernel_test.npy")
        # data = np.repeat(data, 5, axis=1)
        # data = np.repeat(data, 5, axis=2)
        array_to_uncompressed_video(data, out_path=rf"/home/mawa/nas_a/Marvin/chicken_13_05_2025/kilosort4/4px_20Hz_shuffle_460nm_idx_2/cell_{i}/output.mkv", fps=5)