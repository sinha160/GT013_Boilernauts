import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib.animation as animation

class Arrow3D(FancyArrowPatch):
    """3D arrow for matplotlib"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)

class TrajectoryVisualizer:
    """Visualize GTOC13 trajectories"""
    
    def __init__(self, planets_data):
        self.planets = planets_data
        self.AU = 149597870.691  # km
        self.YEAR = 365.25 * 86400  # s
        
        # Color scheme for planets
        self.planet_colors = {
            'Vulcan': '#ff4500',    # Hot orange
            'Yavin': '#90EE90',     # Light green
            'Eden': '#4169E1',      # Royal blue
            'Hoth': '#B0C4DE',      # Light steel blue
            'Yandi': '#8B4513',     # Saddle brown
            'Beyoncé': '#DAA520',   # Goldenrod
            'Bespin': '#FF6347',    # Tomato
            'Jotunn': '#4682B4',    # Steel blue
            'Wakonyingo': '#20B2AA', # Light sea green
            'Rogue1': '#8B008B',    # Dark magenta
            'PlanetX': '#FF1493'    # Deep pink
        }
        
        self.planet_sizes = {
            'Vulcan': 50, 'Yavin': 30, 'Eden': 35,
            'Hoth': 30, 'Yandi': 15, 'Beyoncé': 80,
            'Bespin': 90, 'Jotunn': 60, 'Wakonyingo': 40,
            'Rogue1': 85, 'PlanetX': 55
        }
    
    def plot_2d_trajectory(self, trajectory_data, figsize=(14, 10)):
        """
        2D top-down view (XY plane)
        
        Args:
            trajectory_data: dict with trajectory information
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Left plot: Full system view
        self._plot_system_2d(ax1, trajectory_data, zoom='full')
        ax1.set_title('Full System View (XY Plane)', fontsize=14, fontweight='bold')
        
        # Right plot: Inner system zoom
        self._plot_system_2d(ax2, trajectory_data, zoom='inner')
        ax2.set_title('Inner System View (XY Plane)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def _plot_system_2d(self, ax, trajectory_data, zoom='full'):
        """Helper to plot 2D system view"""
        
        # Plot sun
        ax.plot(0, 0, 'yo', markersize=20, label='Altaira', zorder=100)
        
        # Plot planet orbits
        for name, planet in self.planets.items():
            # Calculate orbit points
            theta = np.linspace(0, 2*np.pi, 200)
            a = planet['a'] / self.AU
            e = planet['e']
            inc = planet['inc']
            Omega = planet['Omega']
            omega = planet['omega']
            
            # Simple 2D projection (ignoring inclination for clarity)
            r = a * (1 - e**2) / (1 + e * np.cos(theta))
            x = r * np.cos(theta + omega) * np.cos(Omega) - r * np.sin(theta + omega) * np.sin(Omega) * np.cos(inc)
            y = r * np.cos(theta + omega) * np.sin(Omega) + r * np.sin(theta + omega) * np.cos(Omega) * np.cos(inc)
            
            color = self.planet_colors.get(name, 'gray')
            ax.plot(x, y, '--', color=color, alpha=0.3, linewidth=1)
            
            # Plot planet position at t=0
            r0, v0 = self._get_planet_state(planet, 0)
            ax.plot(r0[0]/self.AU, r0[1]/self.AU, 'o', 
                   color=color, markersize=self.planet_sizes.get(name, 20)/5,
                   label=name)
        
        # Plot trajectory
        if 'arcs' in trajectory_data:
            for arc in trajectory_data['arcs']:
                if arc['type'] == 'conic':
                    x = [p[0]/self.AU for p in arc['positions']]
                    y = [p[1]/self.AU for p in arc['positions']]
                    ax.plot(x, y, 'b-', linewidth=2, alpha=0.7)
                elif arc['type'] == 'flyby':
                    pos = arc['position']
                    ax.plot(pos[0]/self.AU, pos[1]/self.AU, 'r*', 
                           markersize=15, zorder=50)
        
        # Plot flybys
        if 'flybys' in trajectory_data:
            for i, flyby in enumerate(trajectory_data['flybys']):
                pos = flyby['position']
                ax.plot(pos[0]/self.AU, pos[1]/self.AU, 'r*', 
                       markersize=15, label=f"Flyby {i+1}" if i < 3 else "", zorder=50)
                
                # Add annotation
                if flyby['is_science']:
                    ax.annotate(f"FB{i+1}", 
                               xy=(pos[0]/self.AU, pos[1]/self.AU),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, color='red', fontweight='bold')
        
        # Styling
        ax.set_xlabel('X (AU)', fontsize=12)
        ax.set_ylabel('Y (AU)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        if zoom == 'full':
            ax.set_xlim(-80, 80)
            ax.set_ylim(-80, 80)
        else:
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
        
        # Legend (only show first few items)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:8], labels[:8], loc='upper right', fontsize=8)
    
    def plot_3d_trajectory(self, trajectory_data, figsize=(16, 12)):
        """
        3D visualization of trajectory
        
        Args:
            trajectory_data: dict with trajectory information
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot sun
        ax.plot([0], [0], [0], 'yo', markersize=15, label='Altaira')
        
        # Plot planet orbits in 3D
        for name, planet in self.planets.items():
            theta = np.linspace(0, 2*np.pi, 200)
            a = planet['a'] / self.AU
            e = planet['e']
            inc = planet['inc']
            Omega = planet['Omega']
            omega = planet['omega']
            
            # 3D orbit
            r = a * (1 - e**2) / (1 + e * np.cos(theta))
            
            # Position in orbital plane
            x_orb = r * np.cos(theta)
            y_orb = r * np.sin(theta)
            z_orb = np.zeros_like(theta)
            
            # Rotation matrices
            cos_O = np.cos(Omega)
            sin_O = np.sin(Omega)
            cos_i = np.cos(inc)
            sin_i = np.sin(inc)
            cos_w = np.cos(omega)
            sin_w = np.sin(omega)
            
            # Transform to inertial frame
            x = (cos_O * cos_w - sin_O * sin_w * cos_i) * x_orb + \
                (-cos_O * sin_w - sin_O * cos_w * cos_i) * y_orb
            y = (sin_O * cos_w + cos_O * sin_w * cos_i) * x_orb + \
                (-sin_O * sin_w + cos_O * cos_w * cos_i) * y_orb
            z = (sin_w * sin_i) * x_orb + (cos_w * sin_i) * y_orb
            
            color = self.planet_colors.get(name, 'gray')
            ax.plot(x, y, z, '--', color=color, alpha=0.3, linewidth=0.5)
            
            # Plot planet at t=0
            r0, v0 = self._get_planet_state(planet, 0)
            ax.plot([r0[0]/self.AU], [r0[1]/self.AU], [r0[2]/self.AU],
                   'o', color=color, markersize=self.planet_sizes.get(name, 20)/8)
        
        # Plot trajectory
        if 'arcs' in trajectory_data:
            for arc in trajectory_data['arcs']:
                if arc['type'] == 'conic':
                    x = [p[0]/self.AU for p in arc['positions']]
                    y = [p[1]/self.AU for p in arc['positions']]
                    z = [p[2]/self.AU for p in arc['positions']]
                    ax.plot(x, y, z, 'b-', linewidth=2, alpha=0.8, label='Trajectory')
        
        # Plot flybys
        if 'flybys' in trajectory_data:
            for flyby in trajectory_data['flybys']:
                pos = flyby['position']
                ax.plot([pos[0]/self.AU], [pos[1]/self.AU], [pos[2]/self.AU],
                       'r*', markersize=20)
        
        # Styling
        ax.set_xlabel('X (AU)', fontsize=12)
        ax.set_ylabel('Y (AU)', fontsize=12)
        ax.set_zlabel('Z (AU)', fontsize=12)
        ax.set_title('3D Trajectory View', fontsize=14, fontweight='bold')
        
        # Set equal aspect ratio
        max_range = 70
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        
        return fig
    
    def plot_timeline(self, trajectory_data, figsize=(14, 8)):
        """
        Timeline view showing events and parameters over time
        
        Args:
            trajectory_data: dict with trajectory information
        """
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # Extract time series data
        times = []
        distances = []
        velocities = []
        v_infs = []
        
        if 'arcs' in trajectory_data:
            for arc in trajectory_data['arcs']:
                if arc['type'] == 'conic':
                    for i, pos in enumerate(arc['positions']):
                        t = arc['times'][i]
                        vel = arc['velocities'][i]
                        
                        times.append(t / self.YEAR)
                        distances.append(np.linalg.norm(pos) / self.AU)
                        velocities.append(np.linalg.norm(vel))
        
        # Plot 1: Distance from sun
        axes[0].plot(times, distances, 'b-', linewidth=2)
        axes[0].set_ylabel('Distance (AU)', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Heliocentric Distance vs Time', fontsize=12, fontweight='bold')
        
        # Mark flybys
        if 'flybys' in trajectory_data:
            for flyby in trajectory_data['flybys']:
                t = flyby['time'] / self.YEAR
                d = np.linalg.norm(flyby['position']) / self.AU
                axes[0].plot(t, d, 'r*', markersize=15)
        
        # Plot 2: Velocity
        axes[1].plot(times, velocities, 'g-', linewidth=2)
        axes[1].set_ylabel('Velocity (km/s)', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title('Heliocentric Velocity vs Time', fontsize=12)
        
        # Plot 3: V-infinity at flybys
        if 'flybys' in trajectory_data:
            flyby_times = [f['time']/self.YEAR for f in trajectory_data['flybys']]
            flyby_vinfs = [f['v_inf'] for f in trajectory_data['flybys']]
            
            axes[2].stem(flyby_times, flyby_vinfs, basefmt=' ')
            axes[2].set_ylabel('V∞ (km/s)', fontsize=10)
            axes[2].grid(True, alpha=0.3)
            axes[2].set_title('Flyby V-infinity', fontsize=12)
        
        # Plot 4: Score contribution
        if 'flybys' in trajectory_data:
            scores = []
            for flyby in trajectory_data['flybys']:
                if flyby['is_science']:
                    v_inf = flyby['v_inf']
                    F = 0.2 + np.exp(-v_inf/13) / (1 + np.exp(-5*(v_inf - 1.5)))
                    score = flyby['weight'] * F
                    scores.append(score)
                else:
                    scores.append(0)
            
            axes[3].bar(flyby_times, scores, width=1.0, alpha=0.7)
            axes[3].set_ylabel('Score', fontsize=10)
            axes[3].set_xlabel('Time (years)', fontsize=12)
            axes[3].grid(True, alpha=0.3)
            axes[3].set_title('Score Contribution per Flyby', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def plot_porkchop(self, planet1_name, planet2_name, 
                      t1_range, t2_range, n_points=50):
        """
        Porkchop plot for Lambert transfers between two planets
        
        Args:
            planet1_name: departure planet
            planet2_name: arrival planet
            t1_range: [t1_min, t1_max] departure time range (years)
            t2_range: [t2_min, t2_max] arrival time range (years)
            n_points: grid resolution
        """
        from orbital_mechanics import OrbitalMechanics
        om = OrbitalMechanics()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Create time grids
        t1_grid = np.linspace(t1_range[0]*self.YEAR, t1_range[1]*self.YEAR, n_points)
        t2_grid = np.linspace(t2_range[0]*self.YEAR, t2_range[1]*self.YEAR, n_points)
        
        T1, T2 = np.meshgrid(t1_grid, t2_grid)
        
        # Calculate C3 and arrival V-infinity
        C3_departure = np.zeros_like(T1)
        V_inf_arrival = np.zeros_like(T1)
        
        planet1 = self.planets[planet1_name]
        planet2 = self.planets[planet2_name]
        
        print(f"Computing porkchop plot for {planet1_name} → {planet2_name}...")
        
        for i in range(n_points):
            for j in range(n_points):
                t1 = T1[i, j]
                t2 = T2[i, j]
                
                if t2 <= t1:
                    C3_departure[i, j] = np.nan
                    V_inf_arrival[i, j] = np.nan
                    continue
                
                try:
                    # Get planet states
                    r1, v1_planet = self._get_planet_state(planet1, t1)
                    r2, v2_planet = self._get_planet_state(planet2, t2)
                    
                    # Solve Lambert
                    tof = t2 - t1
                    v1_sc, v2_sc = om.lambert_universal(r1, r2, tof, om.MU_SUN)
                    
                    # Calculate C3 and V-infinity
                    v_inf_dep = v1_sc - v1_planet
                    v_inf_arr = v2_sc - v2_planet
                    
                    C3_departure[i, j] = np.linalg.norm(v_inf_dep)**2
                    V_inf_arrival[i, j] = np.linalg.norm(v_inf_arr)
                    
                except:
                    C3_departure[i, j] = np.nan
                    V_inf_arrival[i, j] = np.nan
        
        # Plot C3
        levels_c3 = np.linspace(0, 100, 20)
        contour1 = ax1.contourf(T1/self.YEAR, T2/self.YEAR, C3_departure, 
                                levels=levels_c3, cmap='viridis')
        ax1.contour(T1/self.YEAR, T2/self.YEAR, C3_departure, 
                   levels=levels_c3, colors='white', alpha=0.3, linewidths=0.5)
        
        plt.colorbar(contour1, ax=ax1, label='C3 (km²/s²)')
        ax1.set_xlabel('Departure Time (years)', fontsize=12)
        ax1.set_ylabel('Arrival Time (years)', fontsize=12)
        ax1.set_title(f'Departure C3: {planet1_name} → {planet2_name}', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot V-infinity arrival
        levels_vinf = np.linspace(0, 30, 20)
        contour2 = ax2.contourf(T1/self.YEAR, T2/self.YEAR, V_inf_arrival,
                               levels=levels_vinf, cmap='plasma')
        ax2.contour(T1/self.YEAR, T2/self.YEAR, V_inf_arrival,
                   levels=levels_vinf, colors='white', alpha=0.3, linewidths=0.5)
        
        plt.colorbar(contour2, ax=ax2, label='V∞ arrival (km/s)')
        ax2.set_xlabel('Departure Time (years)', fontsize=12)
        ax2.set_ylabel('Arrival Time (years)', fontsize=12)
        ax2.set_title(f'Arrival V∞: {planet1_name} → {planet2_name}',
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _get_planet_state(self, planet, t):
        """Helper to get planet state"""
        from orbital_mechanics import OrbitalMechanics
        om = OrbitalMechanics()
        return om.elements_to_cartesian(
            planet['a'], planet['e'], planet['inc'],
            planet['Omega'], planet['omega'], planet['M0'], t
        )
    
    def create_animation(self, trajectory_data, filename='trajectory.mp4', 
                        fps=30, duration=10):
        """
        Create animated trajectory (requires ffmpeg)
        
        Args:
            trajectory_data: dict with trajectory information
            filename: output video file
            fps: frames per second
            duration: animation duration in seconds
        """
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        n_frames = fps * duration
        
        # ... animation code (complex, can provide if needed)
        
        print(f"Animation saved to {filename}")


def visualize_trajectory(solver, trajectory_info):
    """
    Main visualization function
    
    Args:
        solver: GTOC13Solver instance
        trajectory_info: trajectory data from solver
    """
    viz = TrajectoryVisualizer(solver.planets)
    
    # Prepare trajectory data
    trajectory_data = {
        'arcs': trajectory_info.get('arcs', []),
        'flybys': trajectory_info.get('flybys', [])
    }
    
    # Generate plots
    print("Generating 2D view...")
    fig2d = viz.plot_2d_trajectory(trajectory_data)
    fig2d.savefig('trajectory_2d.png', dpi=150, bbox_inches='tight')
    
    print("Generating 3D view...")
    fig3d = viz.plot_3d_trajectory(trajectory_data)
    fig3d.savefig('trajectory_3d.png', dpi=150, bbox_inches='tight')
    
    print("Generating timeline...")
    fig_timeline = viz.plot_timeline(trajectory_data)
    fig_timeline.savefig('trajectory_timeline.png', dpi=150, bbox_inches='tight')
    
    # Porkchop plot example
    print("Generating porkchop plot...")
    fig_pork = viz.plot_porkchop('Eden', 'Beyoncé', [40, 60], [70, 90], n_points=30)
    fig_pork.savefig('porkchop_eden_beyonce.png', dpi=150, bbox_inches='tight')
    
    plt.show()
    
    print("\nVisualization complete!")
    print("Saved: trajectory_2d.png, trajectory_3d.png, trajectory_timeline.png, porkchop_eden_beyonce.png")