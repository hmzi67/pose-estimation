import numpy as np
from scipy.interpolate import interp1d

class SequenceComparator:
    def compare_sequences(self, user_seq, ref_seq):
        # Extract angles and timestamps
        user_angles = np.array([[d["yaw"], d["pitch"], d["roll"]] for d in user_seq])
        ref_angles = np.array([[d["yaw"], d["pitch"], d["roll"]] for d in ref_seq])
        user_times = np.array([d["t"] for d in user_seq])
        ref_times = np.array([d["t"] for d in ref_seq])

        # Normalize times to [0, 1] (base) and prepare sampling grid
        user_duration = user_times[-1] - user_times[0] if len(user_times) > 1 else 1e-6
        ref_duration = ref_times[-1] - ref_times[0] if len(ref_times) > 1 else 1e-6
        user_times_norm = (user_times - user_times[0]) / user_duration
        ref_times_norm = (ref_times - ref_times[0]) / ref_duration

        # Fixed sampling points (e.g., 100 points for efficiency)
        num_points = 100
        sample_times = np.linspace(0, 1, num_points)

        # Try a cheap event-based piecewise linear time-warp to align local tempo differences
        def detect_phases(times_norm, angles_yaw):
            # We expect phases: center -> left (min) -> center -> right (max) -> center
            phases = []
            if len(times_norm) < 3:
                return None
            # approximate center as points where yaw crosses near zero
            # find left peak (min) and right peak (max)
            min_idx = int(np.argmin(angles_yaw))
            max_idx = int(np.argmax(angles_yaw))
            # find nearest center before left (search backwards for crossing near 0)
            def find_center_before(idx):
                for i in range(idx, -1, -1):
                    if abs(angles_yaw[i]) < 5.0:
                        return i
                return 0

            def find_center_after(idx):
                for i in range(idx, len(angles_yaw)):
                    if abs(angles_yaw[i]) < 5.0:
                        return i
                return len(angles_yaw) - 1

            c0 = find_center_before(min_idx)
            l = min_idx
            c1 = find_center_after(min_idx)
            r = max_idx if max_idx > l else None
            if r is None:
                return None
            c2 = find_center_after(r)

            idxs = [c0, l, c1, r, c2]
            # ensure strictly increasing indices
            if not all(x < y for x, y in zip(idxs, idxs[1:])):
                return None
            return times_norm[np.array(idxs)]

        user_phases = detect_phases(user_times_norm, user_angles[:, 0])
        ref_phases = detect_phases(ref_times_norm, ref_angles[:, 0])

        # Interpolate user and reference angles at sample points for all 3 dimensions
        user_interp = np.zeros((num_points, 3))
        ref_interp = np.zeros((num_points, 3))

        # If phase detection succeeded for both, build inverse piecewise mapping from ref->user times
        if user_phases is not None and ref_phases is not None and len(user_phases) == len(ref_phases):
            try:
                inv_map = interp1d(ref_phases, user_phases, kind='linear', fill_value='extrapolate', bounds_error=False)
                # For each target ref sample time, find corresponding user time to sample
                user_sample_times = inv_map(sample_times)

                for dim in range(3):
                    user_interp_func = interp1d(user_times_norm, user_angles[:, dim], kind='linear', fill_value='extrapolate')
                    ref_interp_func = interp1d(ref_times_norm, ref_angles[:, dim], kind='linear', fill_value='extrapolate')
                    user_interp[:, dim] = user_interp_func(user_sample_times)
                    ref_interp[:, dim] = ref_interp_func(sample_times)
            except Exception:
                # Fallback to uniform normalization if anything goes wrong
                user_phases = None

        if user_phases is None or ref_phases is None:
            for dim in range(3):
                user_interp_func = interp1d(user_times_norm, user_angles[:, dim], kind='linear', fill_value='extrapolate')
                ref_interp_func = interp1d(ref_times_norm, ref_angles[:, dim], kind='linear', fill_value='extrapolate')
                user_interp[:, dim] = user_interp_func(sample_times)
                ref_interp[:, dim] = ref_interp_func(sample_times)

        # Compute errors on interpolated sequences for each angle
        # Use a window-constrained DTW (Sakoe-Chiba band) on the downsampled signals to handle
        # local speed differences while keeping computation cheap. Complexity O(N * w).

        def dtw_windowed(a, b, window):
            # a, b are (N,3) and (M,3) arrays
            N = a.shape[0]
            M = b.shape[0]
            INF = 1e12
            D = np.full((N, M), INF, dtype=np.float64)
            # Precompute local distances
            local = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)

            for i in range(N):
                jmin = max(0, i - window)
                jmax = min(M - 1, i + window)
                for j in range(jmin, jmax + 1):
                    if i == 0 and j == 0:
                        D[i, j] = local[0, 0]
                    else:
                        mins = INF
                        if i > 0 and D[i - 1, j] < mins:
                            mins = D[i - 1, j]
                        if j > 0 and D[i, j - 1] < mins:
                            mins = D[i, j - 1]
                        if i > 0 and j > 0 and D[i - 1, j - 1] < mins:
                            mins = D[i - 1, j - 1]
                        D[i, j] = local[i, j] + mins

            # Backtrack path
            i, j = N - 1, M - 1
            if D[i, j] >= INF:
                return None, None
            path = []
            while i > 0 or j > 0:
                path.append((i, j))
                choices = []
                if i > 0 and j > 0:
                    choices.append((D[i - 1, j - 1], i - 1, j - 1))
                if i > 0:
                    choices.append((D[i - 1, j], i - 1, j))
                if j > 0:
                    choices.append((D[i, j - 1], i, j - 1))
                # pick min
                val, ni, nj = min(choices, key=lambda x: x[0])
                i, j = ni, nj
            path.append((0, 0))
            path.reverse()
            return D[N - 1, M - 1], path

        # Choose a small window relative to sequence length (e.g., 10% of length)
        N = user_interp.shape[0]
        window = max(1, int(0.1 * N))
        dtw_cost, path = dtw_windowed(user_interp, ref_interp, window)

        if path is None:
            # Fallback to direct sample-wise error
            errors = np.abs(user_interp - ref_interp)
            mean_error_yaw = np.mean(errors[:, 0])
            mean_error_pitch = np.mean(errors[:, 1])
            mean_error_roll = np.mean(errors[:, 2])
            mean_error = np.mean(errors)
            max_error = np.max(errors)
        else:
            # Compute errors along the alignment path
            aligned_errs = []
            for (i, j) in path:
                aligned_errs.append(np.abs(user_interp[i] - ref_interp[j]))
            aligned_errs = np.array(aligned_errs)
            mean_error_yaw = float(np.mean(aligned_errs[:, 0]))
            mean_error_pitch = float(np.mean(aligned_errs[:, 1]))
            mean_error_roll = float(np.mean(aligned_errs[:, 2]))
            mean_error = float(np.mean(aligned_errs))
            max_error = float(np.max(aligned_errs))

        time_ratio = (user_times[-1] - user_times[0]) / (ref_times[-1] - ref_times[0]) if (ref_times[-1] - ref_times[0]) != 0 else 1.0

        # Optional: penalize excessive warping (normalized by sequence length)
        if path is not None:
            warping_amount = (len(path) - N) / float(max(1, N))
            mean_error += warping_amount * mean_error * 0.5

        return {
            "mean_error_yaw": mean_error_yaw,
            "mean_error_pitch": mean_error_pitch,
            "mean_error_roll": mean_error_roll,
            "mean_error": mean_error,
            "max_error": max_error,
            "time_ratio": time_ratio,
            "dtw_cost": float(dtw_cost) if path is not None else None,
            "dtw_path_length": len(path) if path is not None else None
        }