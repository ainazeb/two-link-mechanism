# --------------------------------------------------
# Two-Link Mechanism Simulation
# Author: Ainaz Ebrahimi
#
# This script simulates a planar two-link mechanism.
# It computes the position, velocity, acceleration,
# and axial force in link AB for different scenarios.
# --------------------------------------------------

import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# ----------------------------
# 1) USER INPUTS 
# ----------------------------

# Each geometry scenario: (L_AB, L_BC, M_b, M_c)
GEOMETRY_SCENARIOS = [
    (1.0, 0.7, 2.0, 1.0),
    (1.2, 0.5, 1.0, 2.5),
]

# Each motion scenario: (omega_AB, omega_BC_relative_to_B)
# Units: rad/s
# Positive omega_AB => CCW rotation of AB
# Negative omega_BC => CW rotation of BC about B
MOTION_SCENARIOS = [
    (2.0, -3.0),
    (1.0, -1.0),
]

N_STEPS = 361  # 0..360 degrees inclusive

OUTPUT_DIR = "outputs"

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR)
# ----------------------------
# 2) SIMULATION FUNCTIONS
# ----------------------------

def simulate_case(L1, L2, Mb, Mc, w1, w2, n_steps=361):
    theta1 = np.linspace(0.0, 2.0*np.pi, n_steps)
    theta_deg = np.degrees(theta1)

    if abs(w1) < 1e-12:
        raise ValueError("omega_AB is too close to zero.")

    t = theta1 / w1
    theta2 = w2 * t

    phi = theta1 + theta2
    w_phi = w1 + w2

    Bx = -L1 * np.sin(theta1)
    By = L1 * np.cos(theta1)
    B_pos = np.column_stack([Bx, By])

    Bvx = -L1 * w1 * np.cos(theta1)
    Bvy = -L1 * w1 * np.sin(theta1)
    B_vel = np.column_stack([Bvx, Bvy])

    Bax = L1 * (w1**2) * np.sin(theta1)
    Bay = -L1 * (w1**2) * np.cos(theta1)
    B_acc = np.column_stack([Bax, Bay])

    Cx = Bx - L2 * np.sin(phi)
    Cy = By + L2 * np.cos(phi)
    C_pos = np.column_stack([Cx, Cy])

    Cvx = Bvx - L2 * w_phi * np.cos(phi)
    Cvy = Bvy - L2 * w_phi * np.sin(phi)
    C_vel = np.column_stack([Cvx, Cvy])

    Cax = Bax + L2 * (w_phi**2) * np.sin(phi)
    Cay = Bay - L2 * (w_phi**2) * np.cos(phi)
    C_acc = np.column_stack([Cax, Cay])

    F_total = Mb * B_acc + Mc * C_acc

    B_norm = np.linalg.norm(B_pos, axis=1)
    u_AB = (B_pos.T / B_norm).T

    axial_force_AB = -np.sum(F_total * u_AB, axis=1)

    return theta_deg, B_pos, B_vel, B_acc, C_pos, C_vel, C_acc, axial_force_AB


def save_force_plot(theta_deg, axial_force, title, out_png):
    plt.figure()
    plt.plot(theta_deg, axial_force)
    plt.axhline(0.0)
    plt.xlabel("AB Rotation Angle (deg)")
    plt.ylabel("Axial Force in AB (signed)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_animation_gif(B_pos, C_pos, L1, L2, out_gif, step=2, fps=30):
    idx = np.arange(0, len(B_pos), step)

    all_pts = np.vstack([np.zeros((len(idx), 2)), B_pos[idx], C_pos[idx]])
    pad = 0.2 * (L1 + L2)
    xmin, ymin = np.min(all_pts, axis=0) - pad
    xmax, ymax = np.max(all_pts, axis=0) + pad

    fig, ax = plt.subplots()
    ax.set_aspect("equal", "box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)
    ax.set_title("Two-link mechanism")

    line_ab, = ax.plot([], [], linewidth=3)
    line_bc, = ax.plot([], [], linewidth=3)
    ax.plot([0], [0], marker="o")
    pt_b, = ax.plot([], [], marker="o")
    pt_c, = ax.plot([], [], marker="o")

    def init():
        line_ab.set_data([], [])
        line_bc.set_data([], [])
        pt_b.set_data([], [])
        pt_c.set_data([], [])
        return line_ab, line_bc, pt_b, pt_c

    def update(frame_i):
        i = idx[frame_i]
        B = B_pos[i]
        C = C_pos[i]

        line_ab.set_data([0, B[0]], [0, B[1]])
        line_bc.set_data([B[0], C[0]], [B[1], C[1]])
        pt_b.set_data([B[0]], [B[1]])
        pt_c.set_data([C[0]], [C[1]])
        return line_ab, line_bc, pt_b, pt_c

    anim = FuncAnimation(fig, update, frames=len(idx), init_func=init, blit=True)
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)


# ----------------------------
# 3) RUN ALL COMBINATIONS
# ----------------------------

def main():
    if not GEOMETRY_SCENARIOS or not MOTION_SCENARIOS:
        print("Fill GEOMETRY_SCENARIOS and MOTION_SCENARIOS first.")
        return

    results = []

    for gi, (L1, L2, Mb, Mc) in enumerate(GEOMETRY_SCENARIOS, start=1):
        for mi, (w1, w2) in enumerate(MOTION_SCENARIOS, start=1):
            theta_deg, B_pos, B_vel, B_acc, C_pos, C_vel, C_acc, axial = simulate_case(
                L1, L2, Mb, Mc, w1, w2, n_steps=N_STEPS
            )

            case_name = f"G{gi}_M{mi}"
            title = f"{case_name} | AB={L1}, BC={L2}, Mb={Mb}, Mc={Mc}, wAB={w1}, wBC={w2}"

            out_png = os.path.join(OUTPUT_DIR, f"{case_name}_axial_force.png")
            save_force_plot(theta_deg, axial, title, out_png)
            print(f"Saved plot: {out_png}")

            results.append((case_name, float(np.max(axial)), float(np.min(axial))))

    L1, L2, Mb, Mc = GEOMETRY_SCENARIOS[0]
    w1, w2 = MOTION_SCENARIOS[0]
    theta_deg, B_pos, _, _, C_pos, _, _, _ = simulate_case(L1, L2, Mb, Mc, w1, w2, n_steps=N_STEPS)

    out_gif = os.path.join(OUTPUT_DIR, "animation_first_case.gif")
    save_animation_gif(B_pos, C_pos, L1, L2, out_gif, step=2, fps=30)
    print(f"Saved animation: {out_gif}")

    best_tension = max(results, key=lambda r: r[1])
    best_compr = min(results, key=lambda r: r[2])

    print("\nEXTREMES SUMMARY")
    print(f"Highest tension:     {best_tension[0]}  max = {best_tension[1]:.6f}")
    print(f"Highest compression: {best_compr[0]}  min = {best_compr[2]:.6f}")
    print(f"\nOutputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":

    main()
