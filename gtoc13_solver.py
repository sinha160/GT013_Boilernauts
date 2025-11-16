import numpy as np
import pandas as pd
from orbital_mechanics import OrbitalMechanics
from solution_writer import SolutionWriter
from scoring import GTOC13Scorer

class GTOC13Solver:
    """Complete GTOC13 trajectory solver with visualization support"""
    
    def __init__(self):
        self.om = OrbitalMechanics()
        self.planets = self.load_planets()
        self.asteroids = self.load_asteroids()
        self.comets = self.load_comets()
        self.scorer = GTOC13Scorer()
        
    def load_planets(self):
        """Load planet data from CSV with special handling for the format"""
        try:
            # Read the raw file content to handle the multi-line header
            with open('data/gtoc13_planets.csv', 'r', encoding='latin1') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            # Skip the header line and process data lines
            data_lines = [line for line in lines if not line.startswith('#')]
            
            # Process each line, handling any line continuation issues
            planets = {}
            for line in data_lines:
                parts = [p.strip() for p in line.split(',')]
                # Handle the case where a line is split across multiple physical lines
                if len(parts) < 10:  # If we don't have enough parts, skip for now
                    continue
                    
                try:
                    planet_data = {
                        'id': int(parts[0]),
                        'name': parts[1],
                        'gm': float(parts[2]),
                        'radius': float(parts[3]),
                        'a': float(parts[4]),
                        'e': float(parts[5]),
                        'i': float(parts[6]),
                        'raan': float(parts[7]),
                        'argp': float(parts[8]),
                        'M': float(parts[9]),
                        'weight': float(parts[10]) if len(parts) > 10 else 0.0
                    }
                    
                    planets[planet_data['name']] = {
                        'id': planet_data['id'],
                        'name': planet_data['name'],
                        'a': planet_data['a'],
                        'e': planet_data['e'],
                        'inc': np.radians(planet_data['i']),
                        'Omega': np.radians(planet_data['raan']),
                        'omega': np.radians(planet_data['argp']),
                        'M0': np.radians(planet_data['M']),
                        'mu': planet_data['gm'],
                        'R': planet_data['radius'],
                        'weight': planet_data['weight']
                    }
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse line: {line}")
                    continue
                    
            if not planets:
                raise ValueError("No valid planet data found in file")
                
            return planets
            
        except FileNotFoundError:
            print("Warning: gtoc13_planets.csv not found. Using example data.")
            return self._create_example_planets()
    
    def _create_example_planets(self):
        """Create example planet data if CSV not available"""
        planets_data = {
            'Vulcan': {'id': 1, 'a': 0.05, 'e': 0.01, 'i': 0, 'Om': 0, 'w': 0, 'M': 0, 
                      'mu': 1e6, 'R': 10000, 'w_science': 0.1},
            'Yavin': {'id': 2, 'a': 0.9, 'e': 0.02, 'i': 1, 'Om': 30, 'w': 45, 'M': 120,
                     'mu': 5e6, 'R': 5000, 'w_science': 1},
            'Eden': {'id': 3, 'a': 1.2, 'e': 0.05, 'i': 2, 'Om': 60, 'w': 90, 'M': 240,
                    'mu': 1e7, 'R': 6371, 'w_science': 2},
            'Hoth': {'id': 4, 'a': 1.8, 'e': 0.08, 'i': 5, 'Om': 90, 'w': 135, 'M': 0,
                    'mu': 8e6, 'R': 6000, 'w_science': 3},
            'Yandi': {'id': 1000, 'a': 2.5, 'e': 0.1, 'i': 3, 'Om': 45, 'w': 180, 'M': 90,
                     'mu': 1e5, 'R': 500, 'w_science': 5},
            'Beyonce': {'id': 5, 'a': 5.2, 'e': 0.048, 'i': 1.3, 'Om': 100, 'w': 275, 'M': 45,
                       'mu': 1.27e8, 'R': 58232, 'w_science': 7},
            'Bespin': {'id': 6, 'a': 9.5, 'e': 0.054, 'i': 2.5, 'Om': 113, 'w': 336, 'M': 78,
                      'mu': 3.79e7, 'R': 69911, 'w_science': 10},
            'Jotunn': {'id': 7, 'a': 19.2, 'e': 0.047, 'i': 0.8, 'Om': 74, 'w': 170, 'M': 131,
                      'mu': 6.83e6, 'R': 24622, 'w_science': 15},
            'Wakonyingo': {'id': 8, 'a': 30.1, 'e': 0.009, 'i': 1.8, 'Om': 131, 'w': 44, 'M': 267,
                          'mu': 6.83e6, 'R': 12000, 'w_science': 20},
            'Rogue1': {'id': 9, 'a': 42.0, 'e': 0.25, 'i': 15, 'Om': 170, 'w': 310, 'M': 95,
                      'mu': 1.27e8, 'R': 69000, 'w_science': 35},
            'PlanetX': {'id': 10, 'a': 67.8, 'e': 0.55, 'i': 25, 'Om': 48, 'w': 152, 'M': 180,
                       'mu': 3.79e7, 'R': 50000, 'w_science': 50}
        }
        
        planets = {}
        for name, data in planets_data.items():
            planets[name] = {
                'id': data['id'],
                'name': name,
                'a': data['a'] * self.om.AU,
                'e': data['e'],
                'inc': np.radians(data['i']),
                'Omega': np.radians(data['Om']),
                'omega': np.radians(data['w']),
                'M0': np.radians(data['M']),
                'mu': data['mu'],
                'R': data['R'],
                'weight': data['w_science']
            }
        
        return planets
    
    def load_asteroids(self):
        """Load asteroid data from CSV"""
        try:
            df = pd.read_csv('data/gtoc13_asteroids.csv')
            # Similar processing as planets
            return {}  # Placeholder
        except FileNotFoundError:
            print("Warning: gtoc13_asteroids.csv not found.")
            return {}
    
    def load_comets(self):
        """Load comet data from CSV"""
        try:
            df = pd.read_csv('data/gtoc13_comets.csv')
            # Similar processing as planets
            return {}  # Placeholder
        except FileNotFoundError:
            print("Warning: gtoc13_comets.csv not found.")
            return {}
    
    def get_planet_state(self, planet_name, t):
        """Get planet state at time t"""
        if planet_name not in self.planets:
            raise ValueError(f"Planet {planet_name} not found")
        
        p = self.planets[planet_name]
        return self.om.elements_to_cartesian(
            p['a'], p['e'], p['inc'], p['Omega'], 
            p['omega'], p['M0'], t
        )
    
    def calculate_score(self, flybys_data, submission_time_days=0):
        """
        Calculate score for trajectory
        
        Args:
            flybys_data: list of dicts with flyby information
            submission_time_days: days since competition start
        
        Returns:
            dict with score breakdown
        """
        # Organize flybys by body_id
        flybys_by_body = {}
        
        for flyby in flybys_data:
            body_id = flyby['body_id']
            if body_id not in flybys_by_body:
                flybys_by_body[body_id] = []
            
            flybys_by_body[body_id].append({
                'time': flyby['time'],
                'position': flyby['position'],
                'v_inf': flyby['v_inf'],
                'is_science': flyby['is_science'],
                'weight': flyby['weight']
            })
        
        # Calculate score
        result = self.scorer.calculate_trajectory_score(
            flybys_by_body, 
            submission_time_days
        )
        
        return result
    
    def propagate_arc(self, r0, v0, t0, t_end, n_points=100):
        """
        Propagate a ballistic arc and return sampled points
        
        Args:
            r0: initial position (km)
            v0: initial velocity (km/s)
            t0: initial time (s)
            t_end: final time (s)
            n_points: number of points to sample
        
        Returns:
            positions, velocities, times
        """
        positions = []
        velocities = []
        times = []
        
        for i in range(n_points):
            t = t0 + (t_end - t0) * i / (n_points - 1)
            r, v = self.om.propagate_keplerian(r0, v0, [t0, t])
            positions.append(r)
            velocities.append(v)
            times.append(t)
        
        return positions, velocities, times
    
    def solve_simple_trajectory(self, visualize=True):
        """
        Build a simple trajectory:
        1. Start at -200 AU
        2. Capture with flyby of inner planet (Eden)
        3. Flyby of outer planet (Beyonce)
        
        Args:
            visualize: whether to generate visualization
        
        Returns:
            dict with trajectory information
        """
        writer = SolutionWriter('solution.txt')
        
        # Define timeline (in years, convert to seconds)
        t0 = 10 * self.om.YEAR
        t_eden = 50 * self.om.YEAR
        t_beyonce = 80 * self.om.YEAR
        
        print("="*70)
        print("GTOC13 SIMPLE TRAJECTORY DESIGNER")
        print("="*70)
        
        # Step 1: Initial state
        print("\n[1] Setting up initial conditions...")
        r0 = np.array([-200 * self.om.AU, 0, 0])
        
        # Get Eden position at flyby time
        r_eden, v_eden = self.get_planet_state('Eden', t_eden)
        
        print(f"  Initial position: [{r0[0]/self.om.AU:.1f}, {r0[1]/self.om.AU:.1f}, {r0[2]/self.om.AU:.1f}] AU")
        print(f"  Eden flyby at t = {t_eden/self.om.YEAR:.1f} years")
        print(f"  Eden position: [{r_eden[0]/self.om.AU:.3f}, {r_eden[1]/self.om.AU:.3f}, {r_eden[2]/self.om.AU:.3f}] AU")
        
        # Step 2: Solve Lambert to reach Eden
        print("\n[2] Solving Lambert problem to Eden...")
        tof1 = t_eden - t0
        
        try:
            v0, v_eden_arrive = self.om.lambert_universal(
                r0, r_eden, tof1, self.om.MU_SUN, tm=1
            )
            print(f"  ✓ Initial velocity: {np.linalg.norm(v0):.3f} km/s")
            print(f"  ✓ Arrival velocity: {np.linalg.norm(v_eden_arrive):.3f} km/s")
            print(f"  ✓ Time of flight: {tof1/self.om.YEAR:.2f} years")
        except Exception as e:
            print(f"  ✗ Lambert solver failed: {e}")
            return None
        
        # Step 3: Eden flyby
        print("\n[3] Performing Eden flyby...")
        eden = self.planets['Eden']
        rp_altitude = 2 * eden['R']  # 2 radii altitude
        
        try:
            v_eden_depart, v_inf, delta = self.om.gravity_assist(
                v_eden_arrive, v_eden, eden['mu'], 
                rp_altitude, eden['R']
            )
            print(f"  ✓ V-infinity: {v_inf:.3f} km/s")
            print(f"  ✓ Turn angle: {np.degrees(delta):.1f}°")
            print(f"  ✓ Periapsis altitude: {rp_altitude:.0f} km ({rp_altitude/eden['R']:.1f} R)")
            
            # Check altitude constraint
            if rp_altitude < 0.1 * eden['R'] or rp_altitude > 100 * eden['R']:
                print(f"  ⚠ Warning: Altitude outside constraints [0.1R, 100R]")
        except Exception as e:
            print(f"  ✗ Flyby calculation failed: {e}")
            return None
        
        # Step 4: Transfer to Beyoncé
        print("\n[4] Solving transfer to Beyoncé...")
        r_beyonce, v_beyonce = self.get_planet_state('Beyoncé', t_beyonce)
        tof2 = t_beyonce - t_eden
        
        print(f"  Beyonce position: [{r_beyonce[0]/self.om.AU:.3f}, {r_beyonce[1]/self.om.AU:.3f}, {r_beyonce[2]/self.om.AU:.3f}] AU")
        print(f"  Time of flight: {tof2/self.om.YEAR:.2f} years")
        
        try:
            v_eden_check, v_beyonce_arrive = self.om.lambert_universal(
                r_eden, r_beyonce, tof2, self.om.MU_SUN, tm=1
            )
            
            # Check velocity matching
            dv_error = np.linalg.norm(v_eden_depart - v_eden_check)
            print(f"  ✓ Transfer solved")
            print(f"  ✓ Velocity matching error: {dv_error:.6f} km/s")
            
            if dv_error > 1.0:  # More than 1 km/s error
                print(f"  ⚠ Warning: Large velocity mismatch!")
                print(f"     This trajectory may not be feasible.")
                print(f"     Consider adjusting flyby parameters or timing.")
        except Exception as e:
            print(f"  ✗ Lambert solver failed: {e}")
            return None
        
        # Step 5: Beyoncé flyby
        print("\n[5] Performing Beyoncé flyby...")
        beyonce = self.planets['Beyoncé']
        rp_altitude_b = 3 * beyonce['R']
        
        try:
            v_beyonce_depart, v_inf_b, delta_b = self.om.gravity_assist(
                v_beyonce_arrive, v_beyonce, beyonce['mu'],
                rp_altitude_b, beyonce['R']
            )
            print(f"  ✓ V-infinity: {v_inf_b:.3f} km/s")
            print(f"  ✓ Turn angle: {np.degrees(delta_b):.1f}°")
            print(f"  ✓ Periapsis altitude: {rp_altitude_b:.0f} km ({rp_altitude_b/beyonce['R']:.1f} R)")
        except Exception as e:
            print(f"  ✗ Flyby calculation failed: {e}")
            return None
        
        # Step 6: Calculate score
        print("\n[6] Calculating trajectory score...")
        
        flybys_data = [
            {
                'body_id': eden['id'],
                'time': t_eden,
                'position': r_eden,
                'v_inf': v_inf,
                'is_science': True,
                'weight': eden['weight'],
                'body_name': 'Eden'
            },
            {
                'body_id': beyonce['id'],
                'time': t_beyonce,
                'position': r_beyonce,
                'v_inf': v_inf_b,
                'is_science': True,
                'weight': beyonce['weight'],
                'body_name': 'Beyoncé'
            }
        ]
        
        score_result = self.calculate_score(flybys_data, submission_time_days=0)
        
        # Step 7: Write solution file
        print("\n[7] Writing solution file...")
        
        # Add initial conic arc to Eden
        writer.add_conic_arc(
            t0, r0, v0,
            t_eden, r_eden, v_eden_arrive
        )
        
        # Add Eden flyby
        v_inf_in = v_eden_arrive - v_eden
        v_inf_out = v_eden_depart - v_eden
        writer.add_flyby(
            eden['id'], True,  # is_science=True
            t_eden, r_eden,
            v_eden_arrive, v_eden_depart,
            v_inf_in, v_inf_out
        )
        
        # Add transfer arc to Beyoncé
        writer.add_conic_arc(
            t_eden, r_eden, v_eden_depart,
            t_beyonce, r_beyonce, v_beyonce_arrive
        )
        
        # Add Beyoncé flyby
        v_inf_in_b = v_beyonce_arrive - v_beyonce
        v_inf_out_b = v_beyonce_depart - v_beyonce
        writer.add_flyby(
            beyonce['id'], True,
            t_beyonce, r_beyonce,
            v_beyonce_arrive, v_beyonce_depart,
            v_inf_in_b, v_inf_out_b
        )
        
        # Write to file
        writer.write()
        
        # Step 8: Prepare visualization data
        print("\n[8] Preparing visualization data...")
        
        # Sample trajectory points
        arc1_pos, arc1_vel, arc1_times = self.propagate_arc(
            r0, v0, t0, t_eden, n_points=100
        )
        
        arc2_pos, arc2_vel, arc2_times = self.propagate_arc(
            r_eden, v_eden_depart, t_eden, t_beyonce, n_points=100
        )
        
        arcs_data = [
            {
                'type': 'conic',
                'positions': arc1_pos,
                'velocities': arc1_vel,
                'times': arc1_times
            },
            {
                'type': 'conic',
                'positions': arc2_pos,
                'velocities': arc2_vel,
                'times': arc2_times
            }
        ]
        
        trajectory_info = {
            'arcs': arcs_data,
            'flybys': flybys_data,
            'initial_state': {'r': r0, 'v': v0, 't': t0},
            'score': score_result
        }
        
        # Print summary
        print("\n" + "="*70)
        print("TRAJECTORY SUMMARY")
        print("="*70)
        print(f"Total mission time:        {(t_beyonce - t0)/self.om.YEAR:.2f} years")
        print(f"Initial epoch:             {t0/self.om.YEAR:.2f} years")
        print(f"Final epoch:               {t_beyonce/self.om.YEAR:.2f} years")
        print(f"Initial velocity:          {np.linalg.norm(v0):.3f} km/s")
        print(f"Scientific flybys:         {score_result['total_science_flybys']}")
        print(f"Bodies visited:            {score_result['bodies_visited']}")
        print(f"\n{'='*30} SCORE {'='*30}")
        print(f"Final Score:               {score_result['final_score']:.3f}")
        print(f"  Base score:              {score_result['base_score']:.3f}")
        print(f"  Grand tour bonus (b):    {score_result['grand_tour_bonus']:.2f}x")
        print(f"  Time bonus (c):          {score_result['time_bonus']:.3f}x")
        print("="*70)
        
        # Detailed score breakdown
        print("\nScore breakdown by body:")
        for body_id, info in score_result['body_scores'].items():
            body_name = 'Unknown'
            for name, planet in self.planets.items():
                if planet['id'] == body_id:
                    body_name = name
                    break
            print(f"  {body_name:15s} (w={info['weight']:4.1f}): {info['body_score']:8.3f} points")
        
        print("\n" + "="*70)
        
        # Generate visualization
        if visualize:
            try:
                from visualizer import visualize_trajectory
                print("\n[9] Generating visualizations...")
                visualize_trajectory(self, trajectory_info)
            except ImportError:
                print("\n⚠ Visualization module not available. Skipping plots.")
            except Exception as e:
                print(f"\n⚠ Visualization failed: {e}")
        
        return trajectory_info
    
    def optimize_trajectory(self, planet_sequence, t_range, max_iterations=100):
        """
        Optimize trajectory timing for a given planet sequence
        
        Args:
            planet_sequence: list of planet names to visit
            t_range: time range for each flyby [t_min, t_max] (years)
            max_iterations: maximum optimization iterations
        
        Returns:
            optimized trajectory info
        """
        from scipy.optimize import differential_evolution
        
        print(f"\n{'='*70}")
        print(f"TRAJECTORY OPTIMIZATION")
        print(f"{'='*70}")
        print(f"Planet sequence: {' → '.join(planet_sequence)}")
        print(f"Time range: {t_range[0]:.1f} - {t_range[1]:.1f} years")
        print(f"Max iterations: {max_iterations}")
        print(f"{'='*70}\n")
        
        n_planets = len(planet_sequence)
        
        def objective(x):
            """Objective function to minimize (negative score)"""
            try:
                # x contains: [t0, t1, t2, ..., tn, y0, z0, vx0]
                times = x[:n_planets+1] * self.om.YEAR
                y0 = x[n_planets+1] * self.om.AU
                z0 = x[n_planets+2] * self.om.AU
                vx0 = x[n_planets+3]
                
                r0 = np.array([-200 * self.om.AU, y0, z0])
                v0 = np.array([vx0, 0, 0])
                
                # Build trajectory
                flybys = []
                r_prev = r0
                v_prev = v0
                t_prev = times[0]
                
                for i, planet_name in enumerate(planet_sequence):
                    t_flyby = times[i+1]
                    
                    if t_flyby <= t_prev:
                        return 1e10  # Invalid timing
                    
                    # Get planet state
                    r_planet, v_planet = self.get_planet_state(planet_name, t_flyby)
                    
                    # Lambert to planet
                    tof = t_flyby - t_prev
                    v_arr_sc, v_dep_sc = self.om.lambert_universal(
                        r_prev, r_planet, tof, self.om.MU_SUN
                    )
                    
                    # Flyby
                    planet = self.planets[planet_name]
                    rp = 2 * planet['R']
                    
                    v_out, v_inf, delta = self.om.gravity_assist(
                        v_arr_sc, v_planet, planet['mu'], rp, planet['R']
                    )
                    
                    flybys.append({
                        'body_id': planet['id'],
                        'time': t_flyby,
                        'position': r_planet,
                        'v_inf': v_inf,
                        'is_science': True,
                        'weight': planet['weight']
                    })
                    
                    r_prev = r_planet
                    v_prev = v_out
                    t_prev = t_flyby
                
                # Calculate score
                score_result = self.calculate_score(flybys)
                return -score_result['final_score']  # Minimize negative score
                
            except Exception as e:
                return 1e10  # Penalty for failed trajectories
        
        # Set up bounds
        bounds = []
        
        # Time bounds for each flyby
        for i in range(n_planets + 1):
            t_min = t_range[0] + i * (t_range[1] - t_range[0]) / n_planets
            t_max = t_range[0] + (i + 1) * (t_range[1] - t_range[0]) / n_planets
            bounds.append((t_min, t_max))
        
        # Initial position bounds
        bounds.append((-50, 50))  # y0 in AU
        bounds.append((-50, 50))  # z0 in AU
        bounds.append((5, 30))    # vx0 in km/s
        
        # Run optimization
        print("Starting optimization (this may take several minutes)...\n")
        
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=max_iterations,
            popsize=15,
            workers=1,
            updating='deferred',
            disp=True
        )
        
        print(f"\nOptimization complete!")
        print(f"Best score: {-result.fun:.3f}")
        print(f"Best parameters: {result.x}")
        
        return result


def main():
    """Main entry point"""
    print("\n" + "="*70)
    print(" "*20 + "GTOC13 TRAJECTORY SOLVER")
    print("="*70 + "\n")
    
    # Create solver
    solver = GTOC13Solver()
    
    print(f"Loaded {len(solver.planets)} planets")
    print(f"Loaded {len(solver.asteroids)} asteroids")
    print(f"Loaded {len(solver.comets)} comets")
    
    # Solve simple trajectory
    trajectory = solver.solve_simple_trajectory(visualize=True)
    
    if trajectory is None:
        print("\n✗ Trajectory design failed!")
        return
    
    print("\n✓ Trajectory design complete!")
    print("✓ Solution written to: solution.txt")
    print("✓ Visualizations saved to current directory")
    
    # Optional: Run optimization
    optimize = input("\nRun trajectory optimization? (y/n): ").strip().lower()
    if optimize == 'y':
        planet_seq = ['Eden', 'Hoth', 'Beyoncé']
        result = solver.optimize_trajectory(
            planet_seq, 
            t_range=[20, 100], 
            max_iterations=50
        )


if __name__ == '__main__':
    main()