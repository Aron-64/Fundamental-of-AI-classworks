import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_trajectory(states, desc, save_filename="trajectory.gif", edge_width=3, fps=2):
    """
    生成不使用颜色填充、标注 S/G/H/F 的轨迹 GIF。
    兼容 imageio 或 Pillow，兼容不同 Matplotlib canvas。
    """
    # 延迟导入媒体库
    have_imageio = False
    have_pil = False
    try:
        import imageio
        have_imageio = True
    except Exception:
        try:
            from PIL import Image
            have_pil = True
        except Exception:
            have_pil = False

    if not (have_imageio or have_pil):
        raise ModuleNotFoundError("Neither imageio nor Pillow (PIL) is installed. Install with: python -m pip install imageio pillow")

    results_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, save_filename)

    frames = []
    fig, ax = plt.subplots(figsize=(6, 6))

    def draw_frame(step_idx):
        ax.clear()
        ax.set_facecolor('white')
        for i in range(4):
            for j in range(4):
                rect = patches.Rectangle((j, i), 1, 1,
                                         linewidth=edge_width,
                                         edgecolor='k',
                                         facecolor='white',
                                         joinstyle='miter')
                ax.add_patch(rect)
                cell = desc[i][j].decode() if isinstance(desc[i][j], (bytes, bytearray)) else str(desc[i][j])
                label = cell if cell in ['S', 'G', 'H', 'F'] else cell.upper()[:1]
                ax.text(j + 0.5, (3 - i) + 0.5, label, fontsize=14, fontweight='bold',
                        ha='center', va='center', color='k')

        sub_states = states[:step_idx + 1]
        coords = [(s // 4, s % 4) for s in sub_states]
        xs = [c for r, c in coords]
        ys = [3 - r for r, c in coords]
        cx = [x + 0.5 for x in xs]
        cy = [y + 0.5 for y in ys]
        if len(cx) > 0:
            ax.plot(cx, cy, '-o', color='black', linewidth=2, markersize=8)
            ax.text(cx[0] - 0.25, cy[0], 'S', fontsize=12, fontweight='bold', va='center')

        g_pos = None
        for i2 in range(4):
            for j2 in range(4):
                cellv = desc[i2][j2]
                if (isinstance(cellv, (bytes, bytearray)) and cellv == b'G') or (str(cellv).upper().startswith('G')):
                    g_pos = (j2 + 0.5, (3 - i2) + 0.5)
                    break
            if g_pos:
                break
        if g_pos:
            ax.text(g_pos[0] - 0.25, g_pos[1], 'G', fontsize=12, fontweight='bold', va='center')

        if len(cx) > 0:
            cur_x, cur_y = cx[-1], cy[-1]
            ax.plot([cur_x], [cur_y], 'o', color='red', markersize=10)

        outer = patches.Rectangle((0, 0), 4, 4, linewidth=edge_width * 1.2, edgecolor='k', facecolor='none')
        ax.add_patch(outer)

        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Agent Trajectory in FrozenLake (labels only)")
        plt.tight_layout()

        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        try:
            buf = fig.canvas.tostring_rgb()
            img = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 3))
        except Exception:
            buf = fig.canvas.tostring_argb()
            arr = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))
            img = arr[:, :, [1, 2, 3]].copy()
        return img

    for t in range(len(states)):
        frames.append(draw_frame(t))

    if have_imageio:
        import imageio
        imageio.mimsave(save_path, frames, fps=fps)
    else:
        from PIL import Image
        pil_frames = [Image.fromarray(f) for f in frames]
        duration_ms = int(1000 / fps) if fps > 0 else 500
        pil_frames[0].save(save_path, save_all=True, append_images=pil_frames[1:], duration=duration_ms, loop=0)

    plt.close(fig)
    print(f"✅ Trajectory GIF saved to {save_path}")