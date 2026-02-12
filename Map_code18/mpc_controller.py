import numpy as np
import math

class MPCController:
    def __init__(self, dt=0.1, N=15, wheelbase=2.8, d_safe=15.0):
        self.dt = dt
        self.N = N
        self.L = wheelbase
        self.d_safe = d_safe
        
        # Tuning Weights
        self.Q = np.diag([5.0, 5.0, 10.0, 2.0]) # [x, y, theta, v]
        self.R = np.diag([2.0, 5.0])           # [accel, steer]
        self.R_rate = np.diag([10.0, 50.0])    # [d_accel, d_steer] - High steer rate penalty prevents oscillation
        self.Q_terminal = self.Q * 10.0
        
        # Constraints
        self.MAX_ACCEL = 1.5
        self.MAX_STEER = np.radians(35)
        self.MAX_SPEED = 40.0
        
        # Memory for warm start
        self.prev_u = np.zeros((N, 2))

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def get_dynamics_jacobians(self, x, u):
        """
        Returns A, B such that x_{k+1} approx A*x_k + B*u_k
        State: [x, y, theta, v]
        Control: [a, delta]
        """
        # x, y, theta, v = x
        theta = x[2]
        v = x[3]
        delta = u[1]
        
        dt = self.dt
        L = self.L
        
        # f1 = x + v*cos(theta)*dt
        # f2 = y + v*sin(theta)*dt
        # f3 = theta + (v/L)*tan(delta)*dt
        # f4 = v + a*dt
        
        A = np.eye(4)
        A[0, 2] = -v * np.sin(theta) * dt
        A[0, 3] = np.cos(theta) * dt
        
        A[1, 2] = v * np.cos(theta) * dt
        A[1, 3] = np.sin(theta) * dt
        
        tan_delta = np.tan(delta)
        A[2, 3] = (1.0/L) * tan_delta * dt
        
        B = np.zeros((4, 2))
        B[3, 0] = dt # v_new w.r.t a
        
        sec2_delta = 1.0 / (np.cos(delta)**2)
        B[2, 1] = (v/L) * sec2_delta * dt
        
        return A, B

    def step_dynamics(self, x, u):
        theta = x[2]
        v = x[3]
        a = u[0]
        delta = u[1]
        
        dt = self.dt
        L = self.L
        
        nx = np.zeros(4)
        nx[0] = x[0] + v * np.cos(theta) * dt
        nx[1] = x[1] + v * np.sin(theta) * dt
        nx[2] = x[2] + (v / L) * np.tan(delta) * dt
        nx[3] = x[3] + a * dt
        
        # Normalize theta
        # nx[2] = self.normalize_angle(nx[2]) # Keep continuous for optimization?
        # Better to keep it continuous and normalize only for error calculation
        
        return nx

    def solve(self, x0, ref_traj, other_trajectories):
        """
        iLQR / DDP Solver (Iterative Linear Quadratic Regulator)
        
        x0: [x, y, theta, v]
        ref_traj: (4, N+1)
        other_trajectories: List of (2, N+1)
        """
        
        # Initialize U with previous solution (shifted)
        U = np.zeros((self.N, 2))
        U[:-1] = self.prev_u[1:]
        U[-1] = self.prev_u[-1] # Duplicate last
        
        X = np.zeros((self.N + 1, 4))
        X[0] = x0
        
        # Initial Rollout
        for k in range(self.N):
            X[k+1] = self.step_dynamics(X[k], U[k])
            
        max_iter = 10
        for ii in range(max_iter):
            # --- Backward Pass ---
            k_gains = np.zeros((self.N, 2))      # Feedforward
            K_gains = np.zeros((self.N, 2, 4))   # Feedback
            
            # Terminal Cost Derivatives
            dx = X[self.N] - ref_traj[:, self.N]
            dx[2] = self.normalize_angle(dx[2]) # Handle angle wrap
            
            Vx = 2 * self.Q_terminal @ dx
            Vxx = 2 * self.Q_terminal
            
            delta_J = 0.0
            
            for k in range(self.N - 1, -1, -1):
                # Linearize dynamics around current trajectory
                A, B = self.get_dynamics_jacobians(X[k], U[k])
                
                # Cost Gradients at step k
                dx = X[k] - ref_traj[:, k]
                dx[2] = self.normalize_angle(dx[2])
                
                lx = 2 * self.Q @ dx
                lxx = 2 * self.Q
                
                lu = 2 * self.R @ U[k]
                luu = 2 * self.R
                
                # Rate Cost (Soft)
                if k > 0:
                    du = U[k] - U[k-1]
                    lu += 2 * self.R_rate @ du
                    luu += 2 * self.R_rate
                
                # Collision Avoidance Barrier (Simple Soft Constraint)
                # Cost += w * exp(-gain * (dist - d_safe))
                # Just gradient accumulation
                grad_coll_x = np.zeros(4)
                hess_coll_xx = np.zeros((4,4))
                
                my_pos = X[k, :2]
                for traj in other_trajectories:
                    if k < traj.shape[1]:
                        oth_pos = traj[:, k]
                        dist_sq = np.sum((my_pos - oth_pos)**2)
                        dist = np.sqrt(dist_sq + 1e-6)
                        
                        if dist < self.d_safe * 1.5: # Only care if close
                            # Barrier: C = 100 * (d_safe - dist)^2 if dist < d_safe
                            # Smooth: C = 1000 * exp(-0.5 * (dist - d_safe)) ? 
                            # Let's use simple inverse square or quadratic penalty
                            if dist < self.d_safe:
                                pen = 100.0 * (self.d_safe - dist)**2
                                # dC/dx = dC/dd * dd/dx
                                # dC/dd = -200 * (d_safe - dist)
                                # dd/dx = (x - x_o) / dist
                                dC_dd = -200.0 * (self.d_safe - dist)
                                dd_dx = (my_pos - oth_pos) / dist
                                dC_dx = dC_dd * dd_dx
                                
                                grad_coll_x[0] += dC_dx[0]
                                grad_coll_x[1] += dC_dx[1]
                
                lx += grad_coll_x
                # Hessian approximation (optional, often skipped for collision in simple iLQR)
                
                # Q-function gradients
                Qx = lx + A.T @ Vx
                Qu = lu + B.T @ Vx
                Qxx = lxx + A.T @ Vxx @ A
                Quu = luu + B.T @ Vxx @ B
                Qux = B.T @ Vxx @ A
                
                # Regularization (Levenberg-Marquardt style) if Quu is not positive definite
                Quu += np.eye(2) * 1e-6
                
                # Feedback Gains
                # u_star = -Quu^-1 * (Qu + Qux * dx)
                try:
                    Quu_inv = np.linalg.inv(Quu)
                except np.linalg.LinAlgError:
                    Quu_inv = np.eye(2) # Fallback
                
                k_curr = -Quu_inv @ Qu
                K_curr = -Quu_inv @ Qux
                
                k_gains[k] = k_curr
                K_gains[k] = K_curr
                
                # Update Value Function for next step
                Vx = Qx + K_curr.T @ Quu @ k_curr + K_curr.T @ Qu + Qux.T @ k_curr
                Vxx = Qxx + K_curr.T @ Quu @ K_curr + K_curr.T @ Qux + Qux.T @ K_curr
                
            # --- Forward Pass (Line Search) ---
            alpha = 1.0
            best_J = float('inf')
            best_U = None
            best_X = None
            
            # Simple line search
            for _ in range(5):
                X_new = np.zeros_like(X)
                U_new = np.zeros_like(U)
                X_new[0] = X[0]
                
                valid = True
                for k in range(self.N):
                    # Feedforward + Feedback
                    dx = X_new[k] - X[k]
                    dx[2] = self.normalize_angle(dx[2]) # Important for feedback on angle
                    
                    du = alpha * k_gains[k] + K_gains[k] @ dx
                    u_cand = U[k] + du
                    
                    # Clipping (Constraints)
                    u_cand[0] = np.clip(u_cand[0], -self.MAX_ACCEL, self.MAX_ACCEL)
                    u_cand[1] = np.clip(u_cand[1], -self.MAX_STEER, self.MAX_STEER)
                    
                    U_new[k] = u_cand
                    X_new[k+1] = self.step_dynamics(X_new[k], U_new[k])
                
                # Evaluate Cost
                J_new = 0
                for k in range(self.N):
                    # Tracking
                    err = X_new[k] - ref_traj[:, k]
                    err[2] = self.normalize_angle(err[2])
                    J_new += err.T @ self.Q @ err
                    
                    # Input
                    J_new += U_new[k].T @ self.R @ U_new[k]
                    
                    # Rate
                    if k > 0:
                        drate = U_new[k] - U_new[k-1]
                        J_new += drate.T @ self.R_rate @ drate
                        
                    # Collision
                    my_pos = X_new[k, :2]
                    for traj in other_trajectories:
                         if k < traj.shape[1]:
                            oth_pos = traj[:, k]
                            dist = np.linalg.norm(my_pos - oth_pos)
                            if dist < self.d_safe:
                                J_new += 100.0 * (self.d_safe - dist)**2

                # Terminal
                err = X_new[self.N] - ref_traj[:, self.N]
                err[2] = self.normalize_angle(err[2])
                J_new += err.T @ self.Q_terminal @ err
                
                if best_U is None or J_new < best_J:
                    best_J = J_new
                    best_U = U_new
                    best_X = X_new
                
                alpha *= 0.5 # Backtracking
                
            # Update trajectory
            U = best_U
            X = best_X
            
            # Check convergence
            # if delta_J < 1e-3: break
        
        self.prev_u = U
        return U, X
