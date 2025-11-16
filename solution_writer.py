import numpy as np

class SolutionWriter:
    """Write GTOC13 solution files"""
    
    def __init__(self, filename):
        self.filename = filename
        self.rows = []
    
    def add_conic_arc(self, t_start, r_start, v_start, t_end, r_end, v_end):
        """Add a ballistic conic arc"""
        # Start point
        self.rows.append([
            0, 0,  # body_id=0 (heliocentric), flag=0 (conic)
            t_start,
            r_start[0], r_start[1], r_start[2],
            v_start[0], v_start[1], v_start[2],
            0, 0, 0  # control (zero for conic)
        ])
        
        # End point
        self.rows.append([
            0, 0,
            t_end,
            r_end[0], r_end[1], r_end[2],
            v_end[0], v_end[1], v_end[2],
            0, 0, 0
        ])
    
    def add_flyby(self, body_id, is_science, t_flyby, r_flyby, 
                  v_in, v_out, v_inf_in, v_inf_out):
        """Add a flyby arc"""
        flag = 1 if is_science else 0
        
        # Incoming state
        self.rows.append([
            body_id, flag,
            t_flyby,
            r_flyby[0], r_flyby[1], r_flyby[2],
            v_in[0], v_in[1], v_in[2],
            v_inf_in[0], v_inf_in[1], v_inf_in[2]
        ])
        
        # Outgoing state
        self.rows.append([
            body_id, flag,
            t_flyby,
            r_flyby[0], r_flyby[1], r_flyby[2],
            v_out[0], v_out[1], v_out[2],
            v_inf_out[0], v_inf_out[1], v_inf_out[2]
        ])
    
    def write(self):
        """Write solution to file"""
        with open(self.filename, 'w') as f:
            f.write("# GTOC13 Solution File\n")
            f.write("# body_id, flag, epoch(s), x(km), y(km), z(km), ")
            f.write("vx(km/s), vy(km/s), vz(km/s), c1, c2, c3\n")
            
            for row in self.rows:
                # Format with appropriate precision
                line = f"{int(row[0])}, {int(row[1])}, "
                line += f"{row[2]:.6f}, "
                line += f"{row[3]:.6f}, {row[4]:.6f}, {row[5]:.6f}, "
                line += f"{row[6]:.9f}, {row[7]:.9f}, {row[8]:.9f}, "
                line += f"{row[9]:.9f}, {row[10]:.9f}, {row[11]:.9f}\n"
                f.write(line)
        
        print(f"Solution written to {self.filename}")
        print(f"Total rows: {len(self.rows)}")