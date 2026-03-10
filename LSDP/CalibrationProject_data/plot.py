import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# --- Read your estimated data ---
df_est = pd.read_csv("translation_after_rot.csv")
df_est = df_est.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)


# Fjern noise fra data med lokal interpolering
def remove_spikes_rolling(series, window=11, threshold=0.02):
    s = series.copy()
    med = s.rolling(window=window, center=True).median()
    diff = np.abs(s - med)
    s[diff > threshold] = np.nan
    return s.interpolate(limit_direction="both")

df_est["tx"] = remove_spikes_rolling(df_est["tx"], window=11, threshold=0.01)
df_est["ty"] = remove_spikes_rolling(df_est["ty"], window=100, threshold=0.01)
df_est["tz"] = remove_spikes_rolling(df_est["tz"], window=11, threshold=0.01)


# -------------------------------------------------
# Marker pose in WORLD frame from OptiTrack
# Quaternion order in scipy: [x, y, z, w]
# Position is in mm
# -------------------------------------------------
q_marker = [0.015309, 0.010602, 0.001538, 0.999825]
t_marker = np.array([-1127.92749, 1383.997314, -1616.664673])  # mm

R_W_M = R.from_quat(q_marker).as_matrix()

# -------------------------------------------------
# Read phone/camera WORLD position from OptiTrack CSV
# Columns G,H,I = indices 6,7,8
# -------------------------------------------------
df_gt_world = pd.read_csv(
    "dict_4x4_250_01.csv",
    skiprows=8,
    usecols=[6, 7, 8],
    header=None,
    names=["x_w", "y_w", "z_w"]
)

df_gt_world = df_gt_world.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

# -------------------------------------------------
# Transform phone position from WORLD frame to MARKER frame
# p_M = R_W_M.T @ (p_W - t_marker)
# -------------------------------------------------
gt_marker_positions = []

for _, row in df_gt_world.iterrows():
    p_w = np.array([row["x_w"], row["y_w"], row["z_w"]])  # mm
    p_m = R_W_M.T @ (p_w - t_marker)                      # mm in marker frame
    gt_marker_positions.append(p_m)
    # gt_marker_positions.append(p_w)


df_gt = pd.DataFrame(gt_marker_positions, columns=["tx", "ty", "tz"])



# Tilføj offset fordi marker er ved siden af a4 papir
offset = np.array([231, -105, 0])  # mm
df_gt[["tx", "ty", "tz"]] = df_gt[["tx", "ty", "tz"]] + offset

# Convert m -> mm so units match 
df_est = df_est * 1000.0

# -------------------------------------------------
# Downsample ground truth from 240 fps to 30 fps
# -------------------------------------------------
df_gt_30 = df_gt.iloc[::8].reset_index(drop=True)

# -------------------------------------------------
# Match lengths
# -------------------------------------------------
n = min(len(df_est), len(df_gt_30))
df_est = df_est.iloc[:n].reset_index(drop=True)
df_gt = df_gt_30.iloc[:n].reset_index(drop=True)

# -------------------------------------------------
# Side-by-side plots
# -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(df_est["tx"], label="tx")
axes[0].plot(df_est["ty"], label="ty")
axes[0].plot(df_est["tz"], label="tz")
axes[0].set_xlabel("Sample")
axes[0].set_ylabel("Position (mm)")
axes[0].set_title("Estimated camera position in marker frame")
axes[0].set_ylim(-500, 3500)
axes[0].legend()
axes[0].grid(True)

axes[1].plot(df_gt["tx"], label="tx")
axes[1].plot(df_gt["ty"], label="ty")
axes[1].plot(df_gt["tz"], label="tz")
axes[1].set_xlabel("Sample")
axes[1].set_ylabel("Position (mm)")
axes[1].set_title("Ground truth transformed to marker frame")
# axes[1].set_title("Ground truth in world frame")
axes[1].set_ylim(-500, 3500)
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# -------------------------------------------------
# Difference plot
# -------------------------------------------------
diff = df_est - df_gt

plt.figure(figsize=(10, 6))
plt.plot(diff["tx"], label="tx error")
plt.plot(diff["ty"], label="ty error")
plt.plot(diff["tz"], label="tz error")

plt.xlabel("Sample")
plt.ylabel("Difference (m)")
plt.title("Difference: estimated - transformed ground truth")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt

# # --- Read your translation data ---
# df_est = pd.read_csv("translation_after_rot.csv")

# # --- Read ground-truth data (columns G, H, I from row 9 and down) ---
# df_gt = pd.read_csv(
#     "dict_4x4_250_01.csv",
#     skiprows=8,
#     usecols=[6, 7, 8],
#     header=None,
#     names=["tx", "ty", "tz"]
# )

# df_gt = df_gt.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
# df_est = df_est.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

# # --- Downsample ground truth from 240 fps to 30 fps ---
# df_gt_30 = df_gt.iloc[::8].reset_index(drop=True)

# # --- Match lengths so they can be compared ---
# n = min(len(df_est), len(df_gt))
# df_est = df_est.iloc[:n].reset_index(drop=True)
# df_gt = df_gt_30.iloc[:n].reset_index(drop=True)

# # --- Side-by-side plots ---
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# # Estimated
# axes[0].plot(df_est["tx"], label="tx")
# axes[0].plot(df_est["ty"], label="ty")
# axes[0].plot(df_est["tz"], label="tz")
# axes[0].set_xlabel("Sample")
# axes[0].set_ylabel("Position")
# axes[0].set_title("Estimated translation")
# axes[0].legend()
# axes[0].grid(True)

# # Ground truth
# axes[1].plot(df_gt["tx"], label="tx")
# axes[1].plot(df_gt["ty"], label="ty")
# axes[1].plot(df_gt["tz"], label="tz")
# axes[1].set_xlabel("Sample")
# axes[1].set_ylabel("Position (mm)")
# axes[1].set_title("Ground-truth position")
# axes[1].legend()
# axes[1].grid(True)

# plt.tight_layout()
# plt.show()

# # --- Difference plot ---
# diff = df_est - df_gt

# plt.figure(figsize=(10, 6))
# plt.plot(diff["tx"], label="tx error")
# plt.plot(diff["ty"], label="ty error")
# plt.plot(diff["tz"], label="tz error")

# plt.xlabel("Sample")
# plt.ylabel("Difference")
# plt.title("Difference: estimated - ground truth")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()



# # Read CSV
# df = pd.read_csv("translation_after_rot.csv")

# # Plot tx, ty, tz
# plt.figure(figsize=(10, 6))
# plt.plot(df["tx"], label="tx")
# plt.plot(df["ty"], label="ty")
# plt.plot(df["tz"], label="tz")

# plt.xlabel("Sample")
# plt.ylabel("Position")
# plt.title("Translation data")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# # Read columns G, H, I starting from row 9
# df = pd.read_csv(
#     "dict_4x4_250_01.csv",
#     skiprows=8,
#     usecols=[6, 7, 8],
#     header=None,
#     names=["tx", "ty", "tz"]
# )

# # Convert to numeric in case there is stray text
# df = df.apply(pd.to_numeric, errors="coerce").dropna()

# # Plot like your own data
# plt.figure(figsize=(10, 6))
# plt.plot(df["tx"], label="tx")
# plt.plot(df["ty"], label="ty")
# plt.plot(df["tz"], label="tz")

# plt.xlabel("Sample")
# plt.ylabel("Position (mm)")
# plt.title("Ground-truth phone position")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()