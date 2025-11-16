import numpy as np
from scipy.optimize import fsolve, brentq
from scipy.integrate import solve_ivp

class OrbitalMechanics:
    """Core orbital mechanics functions for GTOC13"""
    
    def __init__(self):
        # GTOC13 constants
        self.AU = 149597870.691  # km
        self.MU_SUN = 139348062043.343  # km^3/s^2
        self.DAY = 86400  # seconds
        self.YEAR = 365.25 * self.DAY  # seconds
    
    def kepler_equation_solver(self, M, e, tol=1e-10):
        """
        Solve Kepler's equation: M = E - e*sin(E)
        
        Args:
            M: mean anomaly (rad)
            e: eccentricity
            tol: tolerance
        
        Returns:
            E: eccentric anomaly (rad)
        """
        # Initial guess
        if e < 0.8:
            E0 = M
        else:
            E0 = np.pi
        
        # Newton-Raphson iteration
        def kepler_func(E):
            return E - e * np.sin(E) - M
        
        def kepler_deriv(E):
            return 1 - e * np.cos(E)
        
        E = E0
        for _ in range(50):
            f = kepler_func(E)
            if abs(f) < tol:
                break
            fp = kepler_deriv(E)
            E = E - f / fp
        
        return E
    
    def elements_to_cartesian(self, a, e, inc, Omega, omega, M0, t=0):
        """
        Convert orbital elements to Cartesian state
        
        Args:
            a: semi-major axis (km)
            e: eccentricity
            inc: inclination (rad)
            Omega: longitude of ascending node (rad)
            omega: argument of periapsis (rad)
            M0: mean anomaly at epoch t=0 (rad)
            t: time since epoch (s)
        
        Returns:
            r: position vector [x, y, z] (km)
            v: velocity vector [vx, vy, vz] (km/s)
        """
        # Mean motion
        n = np.sqrt(self.MU_SUN / a**3)
        
        # Current mean anomaly
        M = M0 + n * t
        M = M % (2 * np.pi)  # Wrap to [0, 2π]
        
        # Solve for eccentric anomaly
        E = self.kepler_equation_solver(M, e)
        
        # True anomaly
        nu = 2 * np.arctan2(
            np.sqrt(1 + e) * np.sin(E / 2),
            np.sqrt(1 - e) * np.cos(E / 2)
        )
        
        # Distance
        r_mag = a * (1 - e * np.cos(E))
        
        # Position in orbital plane
        x_orb = r_mag * np.cos(nu)
        y_orb = r_mag * np.sin(nu)
        
        # Velocity in orbital plane
        h = np.sqrt(self.MU_SUN * a * (1 - e**2))  # specific angular momentum
        vx_orb = -(self.MU_SUN / h) * np.sin(nu)
        vy_orb = (self.MU_SUN / h) * (e + np.cos(nu))
        
        # Rotation matrices (3-1-3 sequence)
        cos_O = np.cos(Omega)
        sin_O = np.sin(Omega)
        cos_i = np.cos(inc)
        sin_i = np.sin(inc)
        cos_w = np.cos(omega)
        sin_w = np.sin(omega)
        
        # Combined rotation matrix
        R11 = cos_O * cos_w - sin_O * sin_w * cos_i
        R12 = -cos_O * sin_w - sin_O * cos_w * cos_i
        R21 = sin_O * cos_w + cos_O * sin_w * cos_i
        R22 = -sin_O * sin_w + cos_O * cos_w * cos_i
        R31 = sin_w * sin_i
        R32 = cos_w * sin_i
        
        # Transform to inertial frame
        r = np.array([
            R11 * x_orb + R12 * y_orb,
            R21 * x_orb + R22 * y_orb,
            R31 * x_orb + R32 * y_orb
        ])
        
        v = np.array([
            R11 * vx_orb + R12 * vy_orb,
            R21 * vx_orb + R22 * vy_orb,
            R31 * vx_orb + R32 * vy_orb
        ])
        
        return r, v
    
    def propagate_keplerian(self, r0, v0, t_span):
        """
        Propagate a Keplerian orbit
        
        Args:
            r0: initial position (km)
            v0: initial velocity (km/s)
            t_span: [t_start, t_end] (s)
        
        Returns:
            r_final: final position (km)
            v_final: final velocity (km/s)
        """
        def two_body_eom(t, state):
            r = state[:3]
            v = state[3:]
            r_mag = np.linalg.norm(r)
            a = -self.MU_SUN * r / r_mag**3
            return np.concatenate([v, a])
        
        state0 = np.concatenate([r0, v0])
        sol = solve_ivp(two_body_eom, t_span, state0, 
                       method='DOP853', rtol=1e-10, atol=1e-12)
        
        final_state = sol.y[:, -1]
        return final_state[:3], final_state[3:]
    
    def lambert_universal(self, r1, r2, tof, mu, tm=1):
        """
        Solve Lambert's problem using universal variables
        
        Args:
            r1: initial position vector (km)
            r2: final position vector (km)
            tof: time of flight (s)
            mu: gravitational parameter (km^3/s^2)
            tm: transfer type (+1 for short way, -1 for long way)
        
        Returns:
            v1: initial velocity vector (km/s)
            v2: final velocity vector (km/s)
        """
        r1_mag = np.linalg.norm(r1)
        r2_mag = np.linalg.norm(r2)
        
        cos_dnu = np.dot(r1, r2) / (r1_mag * r2_mag)
        cos_dnu = np.clip(cos_dnu, -1.0, 1.0)
        
        # Transfer angle
        cross_prod = np.cross(r1, r2)
        if tm * cross_prod[2] >= 0:
            dnu = np.arccos(cos_dnu)
        else:
            dnu = 2 * np.pi - np.arccos(cos_dnu)
        
        A = tm * np.sqrt(r1_mag * r2_mag * (1 + cos_dnu))
        
        if A == 0:
            raise ValueError("Lambert problem cannot be solved - 180° transfer")
        
        # Stumpff functions
        def stumpff_c(z):
            if z > 1e-6:
                return (1 - np.cos(np.sqrt(z))) / z
            elif z < -1e-6:
                return (np.cosh(np.sqrt(-z)) - 1) / (-z)
            else:
                return 1/2 - z/24 + z**2/720
        
        def stumpff_s(z):
            if z > 1e-6:
                sz = np.sqrt(z)
                return (sz - np.sin(sz)) / (sz**3)
            elif z < -1e-6:
                sz = np.sqrt(-z)
                return (np.sinh(sz) - sz) / (sz**3)
            else:
                return 1/6 - z/120 + z**2/5040
        
        # Time of flight function
        def tof_func(z):
            C = stumpff_c(z)
            S = stumpff_s(z)
            y = r1_mag + r2_mag + A * (z * S - 1) / np.sqrt(C)
            
            if y < 0:
                return 1e10
            
            x = np.sqrt(y / C)
            t = (x**3 * S + A * np.sqrt(y)) / np.sqrt(mu)
            return t
        
        # Find z that gives correct TOF
        def tof_error(z):
            return tof_func(z) - tof
        
        # Initial bounds for z
        z_low = -4 * np.pi**2
        z_high = 4 * np.pi**2
        
        try:
            z = brentq(tof_error, z_low, z_high, xtol=1e-8)
        except:
            # If brentq fails, try fsolve
            z = fsolve(tof_error, 0)[0]
        
        # Calculate velocities
        C = stumpff_c(z)
        S = stumpff_s(z)
        y = r1_mag + r2_mag + A * (z * S - 1) / np.sqrt(C)
        
        f = 1 - y / r1_mag
        g = A * np.sqrt(y / mu)
        gdot = 1 - y / r2_mag
        
        v1 = (r2 - f * r1) / g
        v2 = (gdot * r2 - r1) / g
        
        return v1, v2
    
    def gravity_assist(self, v_sc_in, v_planet, mu_planet, rp_altitude, R_planet):
        """
        Calculate gravity assist (simplified powered model)
        
        Args:
            v_sc_in: incoming spacecraft velocity (heliocentric) (km/s)
            v_planet: planet velocity (heliocentric) (km/s)
            mu_planet: planet gravitational parameter (km^3/s^2)
            rp_altitude: periapsis altitude above surface (km)
            R_planet: planet radius (km)
        
        Returns:
            v_sc_out: outgoing spacecraft velocity (heliocentric) (km/s)
            v_inf_mag: hyperbolic excess velocity magnitude (km/s)
            delta: turn angle (rad)
        """
        # Relative velocity
        v_inf_in = v_sc_in - v_planet
        v_inf_mag = np.linalg.norm(v_inf_in)
        
        # Periapsis radius
        rp = R_planet + rp_altitude
        
        # Turn angle from vis-viva
        e_hyp = 1 + (rp * v_inf_mag**2) / mu_planet
        
        if e_hyp <= 1:
            raise ValueError(f"Not a hyperbolic orbit: e = {e_hyp}")
        
        delta = 2 * np.arcsin(1 / e_hyp)
        
        # For simplicity: rotate v_inf by delta in the plane
        # perpendicular to approach velocity
        # This is a simplified model - full B-plane targeting needed for precision
        
        v_inf_in_hat = v_inf_in / v_inf_mag
        
        # Create perpendicular vector
        if abs(v_inf_in_hat[2]) < 0.9:
            perp = np.cross(v_inf_in_hat, np.array([0, 0, 1]))
        else:
            perp = np.cross(v_inf_in_hat, np.array([1, 0, 0]))
        perp = perp / np.linalg.norm(perp)
        
        # Rodrigues rotation
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        v_inf_out = (cos_d * v_inf_in + 
                     sin_d * np.cross(perp, v_inf_in) +
                     (1 - cos_d) * np.dot(perp, v_inf_in) * perp)
        
        # Convert back to heliocentric
        v_sc_out = v_inf_out + v_planet
        
        return v_sc_out, v_inf_mag, delta