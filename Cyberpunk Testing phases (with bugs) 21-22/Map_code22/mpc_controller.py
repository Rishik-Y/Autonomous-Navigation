import numpy as np
import math


class MPCController:
    """
    iLQR-based Model Predictive Controller with lane-aware collision avoidance.

    Key features over the original:
    - Lane-aware collision: decomposes into longitudinal/lateral so trucks on
      opposite lanes pass each other naturally without going off-road.
    - Following behaviour: trucks behind another in the same lane brake instead
      of swerving.
    - Speed clamping in dynamics (no reverse).
    - Early convergence termination.
    - Collision Hessian for better iLQR convergence.
    """

    def __init__(self, dt=0.1, N=20, wheelbase=2.8, d_safe=8.0):
        self.dt = dt
        self.N = N
        self.L = wheelbase
        self.d_safe = d_safe

        # --- Tuning Weights ---
        # [x, y, theta, v]
        self.Q = np.diag([10.0, 10.0, 20.0, 4.0])
        # [accel, steer]
        self.R = np.diag([0.5, 2.0])
        # [d_accel, d_steer] – smooth inputs, heavy steer-rate penalty
        self.R_rate = np.diag([3.0, 50.0])
        # Terminal cost
        self.Q_terminal = self.Q * 5.0

        # --- Constraints ---
        self.MAX_ACCEL = 1.5
        self.MAX_STEER = np.radians(35)
        self.MAX_SPEED = 15.0   # m/s (~54 km/h)
        self.MIN_SPEED = 0.0

        # --- Collision avoidance parameters ---
        self.LANE_WIDTH = 3.5       # lateral threshold (m)
        self.COLL_WEIGHT = 500.0    # barrier weight (strong)
        self.COLL_LONG_WEIGHT = 200.0  # longitudinal following penalty

        # --- iLQR parameters (Adaptive Regularization) ---
        self.max_iters = 10
        self.line_search_steps = 5
        self.reg_min = 1e-6
        self.reg_max = 1e4
        self.reg_factor = 10.0

        # --- Memory for warm start ---
        self.prev_u = np.zeros((N, 2))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def step_dynamics(self, x, u):
        """Bicycle kinematic model.  State [x, y, theta, v], control [a, delta]."""
        theta = x[2]
        v = max(x[3], 0.0)       # clamp speed >= 0
        a = u[0]
        delta = u[1]
        dt = self.dt
        L = self.L

        nx = np.zeros(4)
        nx[0] = x[0] + v * np.cos(theta) * dt
        nx[1] = x[1] + v * np.sin(theta) * dt
        nx[2] = x[2] + (v / L) * np.tan(delta) * dt
        nx[3] = max(x[3] + a * dt, 0.0)   # no reverse

        return nx

    def get_dynamics_jacobians(self, x, u):
        """Linearised A, B around (x, u)."""
        theta = x[2]
        v = max(x[3], 0.0)
        delta = u[1]
        dt = self.dt
        L = self.L

        A = np.eye(4)
        A[0, 2] = -v * np.sin(theta) * dt
        A[0, 3] = np.cos(theta) * dt
        A[1, 2] = v * np.cos(theta) * dt
        A[1, 3] = np.sin(theta) * dt
        tan_d = np.tan(delta)
        A[2, 3] = tan_d / L * dt

        B = np.zeros((4, 2))
        B[3, 0] = dt
        sec2 = 1.0 / (np.cos(delta) ** 2)
        B[2, 1] = (v / L) * sec2 * dt

        return A, B

    # ------------------------------------------------------------------
    # Lane-aware collision cost
    # ------------------------------------------------------------------

    def _collision_cost_and_grad(self, my_state, other_trajectories, k):
        """
        Returns (cost_scalar, grad_x4) for step k.

        Decomposition:
          - longitudinal = projection of diff onto my heading  (ahead / behind)
          - lateral      = rejection  (same lane vs opposite lane)

        Rules:
          - |lateral| > LANE_WIDTH  → skip (opposite lane, trucks pass naturally)
          - |lateral| < LANE_WIDTH AND longitudinal close → penalise longitudinally
            (truck should BRAKE, not swerve)
        """
        cost = 0.0
        grad = np.zeros(4)
        hess = np.zeros((4, 4))

        my_pos = my_state[:2]
        theta = my_state[2]
        heading = np.array([np.cos(theta), np.sin(theta)])
        normal = np.array([-np.sin(theta), np.cos(theta)])

        for traj in other_trajectories:
            if k >= traj.shape[1]:
                continue
            oth_pos = traj[:, k]
            diff = oth_pos - my_pos           # vector FROM me TO other
            dist = np.linalg.norm(diff) + 1e-6

            if dist > self.d_safe * 2.0:
                continue  # too far, skip

            # Decompose
            longitudinal = np.dot(diff, heading)   # +ve = ahead of me
            lateral = np.dot(diff, normal)          # +ve = to my left

            # --- Skip if clearly on opposite lane ---
            if abs(lateral) > self.LANE_WIDTH:
                continue

            # --- Same-lane collision avoidance ---
            if dist < self.d_safe:
                # Quadratic barrier: C = w * (d_safe - dist)^2
                pen = self.COLL_WEIGHT * (self.d_safe - dist) ** 2
                cost += pen

                # Gradient of dist w.r.t. my_pos:  dd/dpos = -(diff)/dist
                # dC/dpos = dC/dd * dd/dpos = -2*w*(d_safe - dist) * (-(diff)/dist)
                #         = 2*w*(d_safe - dist) * diff / dist
                dC_dpos = 2.0 * self.COLL_WEIGHT * (self.d_safe - dist) * (-diff / dist)
                grad[0] += dC_dpos[0]
                grad[1] += dC_dpos[1]

                # Approximate Hessian (Gauss-Newton)
                outer = np.outer(diff / dist, diff / dist)
                H_approx = 2.0 * self.COLL_WEIGHT * outer
                hess[:2, :2] += H_approx

            # --- Following penalty (longitudinal braking pressure) ---
            # If other truck is AHEAD and close, add penalty that encourages
            # deceleration (penalises high speed in the cost gradient).
            if 0 < longitudinal < self.d_safe and abs(lateral) < self.LANE_WIDTH:
                follow_gap = self.d_safe - longitudinal
                if follow_gap > 0:
                    # Penalty on velocity: encourage slowing down
                    follow_pen = self.COLL_LONG_WEIGHT * follow_gap ** 2
                    cost += follow_pen
                    # Gradient: penalise positive velocity  →  grad on v
                    grad[3] += 2.0 * self.COLL_LONG_WEIGHT * follow_gap * 0.5

        return cost, grad, hess

    # ------------------------------------------------------------------
    # iLQR Solver
    # ------------------------------------------------------------------

    def solve(self, x0, ref_traj, other_trajectories):
        """
        iLQR solver with adaptive regularisation and warm start.

        x0              : [x, y, theta, v]
        ref_traj        : (4, N+1)
        other_trajectories : list of (2, N+1)
        """
        N = self.N

        # Warm-start from shifted previous solution
        U = np.zeros((N, 2))
        U[:-1] = self.prev_u[1:]
        U[-1] = self.prev_u[-1]

        # Clamp warm-start controls
        U[:, 0] = np.clip(U[:, 0], -self.MAX_ACCEL, self.MAX_ACCEL)
        U[:, 1] = np.clip(U[:, 1], -self.MAX_STEER, self.MAX_STEER)

        # Initial rollout
        X = np.zeros((N + 1, 4))
        X[0] = np.array(x0, dtype=float)
        for k in range(N):
            X[k + 1] = self.step_dynamics(X[k], U[k])

        prev_cost = self._total_cost(X, U, ref_traj, other_trajectories)
        reg = 1e-3  # initial regularisation

        for iteration in range(self.max_iters):
            # ---------- Backward pass ----------
            k_ff = np.zeros((N, 2))
            K_fb = np.zeros((N, 2, 4))
            backward_ok = True

            # Terminal cost
            dx = X[N] - ref_traj[:, min(N, ref_traj.shape[1] - 1)]
            dx[2] = self.normalize_angle(dx[2])
            Vx = 2.0 * self.Q_terminal @ dx
            Vxx = 2.0 * self.Q_terminal.copy()

            for k in range(N - 1, -1, -1):
                A, B = self.get_dynamics_jacobians(X[k], U[k])

                ref_k = min(k, ref_traj.shape[1] - 1)
                dx = X[k] - ref_traj[:, ref_k]
                dx[2] = self.normalize_angle(dx[2])

                lx = 2.0 * self.Q @ dx
                lxx = 2.0 * self.Q.copy()

                lu = 2.0 * self.R @ U[k]
                luu = 2.0 * self.R.copy()

                # Input rate cost
                if k > 0:
                    du = U[k] - U[k - 1]
                    lu += 2.0 * self.R_rate @ du
                    luu += 2.0 * self.R_rate

                # Collision cost
                coll_cost, coll_grad, coll_hess = self._collision_cost_and_grad(
                    X[k], other_trajectories, k
                )
                lx += coll_grad
                lxx += coll_hess

                # Q-function
                Qx = lx + A.T @ Vx
                Qu = lu + B.T @ Vx
                Qxx = lxx + A.T @ Vxx @ A
                Quu = luu + B.T @ Vxx @ B
                Qux = B.T @ Vxx @ A

                # Adaptive regularisation
                Quu_reg = Quu + np.eye(2) * reg

                # Check positive-definiteness
                try:
                    np.linalg.cholesky(Quu_reg)
                except np.linalg.LinAlgError:
                    reg = min(reg * self.reg_factor, self.reg_max)
                    backward_ok = False
                    break

                Quu_inv = np.linalg.inv(Quu_reg)

                k_ff[k] = -Quu_inv @ Qu
                K_fb[k] = -Quu_inv @ Qux

                Vx = Qx + K_fb[k].T @ Quu @ k_ff[k] + K_fb[k].T @ Qu + Qux.T @ k_ff[k]
                Vxx = Qxx + K_fb[k].T @ Quu @ K_fb[k] + K_fb[k].T @ Qux + Qux.T @ K_fb[k]
                # Symmetrise Vxx to prevent numerical drift
                Vxx = 0.5 * (Vxx + Vxx.T)

            if not backward_ok:
                continue  # retry with higher reg

            # ---------- Forward pass (line search) ----------
            best_J = float('inf')
            best_U = None
            best_X = None
            alpha = 1.0
            improved = False

            for _ in range(self.line_search_steps):
                X_new = np.zeros_like(X)
                U_new = np.zeros_like(U)
                X_new[0] = X[0]

                for k in range(N):
                    ddx = X_new[k] - X[k]
                    ddx[2] = self.normalize_angle(ddx[2])
                    du = alpha * k_ff[k] + K_fb[k] @ ddx
                    u_cand = U[k] + du

                    # Constraints
                    u_cand[0] = np.clip(u_cand[0], -self.MAX_ACCEL, self.MAX_ACCEL)
                    u_cand[1] = np.clip(u_cand[1], -self.MAX_STEER, self.MAX_STEER)

                    U_new[k] = u_cand
                    X_new[k + 1] = self.step_dynamics(X_new[k], U_new[k])

                J = self._total_cost(X_new, U_new, ref_traj, other_trajectories)

                if J < best_J:
                    best_J = J
                    best_U = U_new.copy()
                    best_X = X_new.copy()

                alpha *= 0.5

            if best_U is not None and best_J < prev_cost:
                U = best_U
                X = best_X
                improved = True
                # Decrease regularisation on success
                reg = max(reg / self.reg_factor, self.reg_min)
            else:
                # Increase regularisation on failure
                reg = min(reg * self.reg_factor, self.reg_max)

            # Convergence check (relative improvement)
            if improved and prev_cost > 1e-6:
                rel_improve = (prev_cost - best_J) / (abs(prev_cost) + 1e-6)
                if rel_improve < 1e-4:
                    break
            prev_cost = min(prev_cost, best_J) if best_J < float('inf') else prev_cost

        self.prev_u = U
        return U, X

    def _total_cost(self, X, U, ref_traj, other_trajectories):
        """Evaluate total trajectory cost for line search."""
        N = self.N
        J = 0.0
        for k in range(N):
            ref_k = min(k, ref_traj.shape[1] - 1)
            err = X[k] - ref_traj[:, ref_k]
            err[2] = self.normalize_angle(err[2])
            J += err @ self.Q @ err
            J += U[k] @ self.R @ U[k]

            if k > 0:
                dr = U[k] - U[k - 1]
                J += dr @ self.R_rate @ dr

            c_cost, _, _ = self._collision_cost_and_grad(
                X[k], other_trajectories, k
            )
            J += c_cost

        # Terminal
        ref_N = min(N, ref_traj.shape[1] - 1)
        err = X[N] - ref_traj[:, ref_N]
        err[2] = self.normalize_angle(err[2])
        J += err @ self.Q_terminal @ err
        return J
