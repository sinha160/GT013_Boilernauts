import numpy as np

class GTOC13Scorer:
    """Calculate GTOC13 trajectory score"""
    
    def __init__(self):
        self.AU = 149597870.691  # km
        self.YEAR = 365.25 * 86400  # seconds
    
    def calculate_F(self, v_inf):
        """
        Calculate flyby velocity penalty term F
        
        Args:
            v_inf: hyperbolic excess velocity magnitude (km/s)
        
        Returns:
            F: velocity penalty factor [0.2, 1.0]
        """
        F = 0.2 + np.exp(-v_inf/13) / (1 + np.exp(-5*(v_inf - 1.5)))
        return F
    
    def calculate_S(self, r_positions):
        """
        Calculate seasonal penalty term S for multiple flybys of same body
        
        Args:
            r_positions: list of heliocentric position vectors at each flyby
        
        Returns:
            S_values: list of seasonal penalties for each flyby
        """
        S_values = []
        
        for i, r_i in enumerate(r_positions):
            if i == 0:
                # First flyby always has S = 1
                S_values.append(1.0)
            else:
                # Calculate S based on previous flybys
                r_i_hat = r_i / np.linalg.norm(r_i)
                
                sum_term = 0.0
                for j in range(i):
                    r_j = r_positions[j]
                    r_j_hat = r_j / np.linalg.norm(r_j)
                    
                    # Dot product (clamped to [-1, 1])
                    cos_angle = np.clip(np.dot(r_i_hat, r_j_hat), -1.0, 1.0)
                    angle_deg = np.degrees(np.arccos(cos_angle))
                    
                    # Exponential term
                    sum_term += np.exp(-(angle_deg**2) / 50)
                
                S_i = 0.1 + 0.9 / (1 + 10 * sum_term)
                S_values.append(S_i)
        
        return S_values
    
    def calculate_time_bonus(self, submission_time_days):
        """
        Calculate time bonus c
        
        Args:
            submission_time_days: days since competition start
        
        Returns:
            c: time bonus factor
        """
        if submission_time_days <= 7:
            c = 1.13
        else:
            c = -0.005 * submission_time_days + 1.165
        
        # Clamp to reasonable range
        c = max(c, 1.0)
        
        return c
    
    def check_grand_tour(self, flybys):
        """
        Check if trajectory qualifies for grand tour bonus
        
        Args:
            flybys: dict with body_id as key, containing flyby info
        
        Returns:
            bool: True if grand tour achieved
        """
        # Need all planets (1-10), dwarf planet (1000), 
        # and at least 13 asteroids or comets (1001-1257, 2001-2042)
        
        planets_visited = set()
        asteroids_comets_visited = set()
        
        for body_id in flybys.keys():
            if 1 <= body_id <= 10:
                planets_visited.add(body_id)
            elif body_id == 1000:  # Yandi
                planets_visited.add(body_id)
            elif (1001 <= body_id <= 1257) or (2001 <= body_id <= 2042):
                asteroids_comets_visited.add(body_id)
        
        # Check criteria
        has_all_planets = len(planets_visited) == 11  # 10 planets + Yandi
        has_enough_small_bodies = len(asteroids_comets_visited) >= 13
        
        return has_all_planets and has_enough_small_bodies
    
    def calculate_trajectory_score(self, flybys, submission_time_days=0):
        """
        Calculate total trajectory score
        
        Args:
            flybys: dict of flybys organized by body_id
                    {body_id: [{
                        'time': epoch,
                        'position': [x,y,z],
                        'v_inf': magnitude,
                        'is_science': bool,
                        'weight': body weight
                    }, ...]}
            submission_time_days: days since competition start
        
        Returns:
            dict with score breakdown
        """
        # Calculate time bonus
        c = self.calculate_time_bonus(submission_time_days)
        
        # Calculate grand tour bonus
        b = 1.2 if self.check_grand_tour(flybys) else 1.0
        
        # Calculate score for each body
        total_score = 0.0
        body_scores = {}
        total_science_flybys = 0
        
        for body_id, body_flybys in flybys.items():
            # Only count science flybys (up to 13 per body)
            science_flybys = [f for f in body_flybys if f['is_science']]
            science_flybys = science_flybys[:13]  # Max 13
            
            if len(science_flybys) == 0:
                continue
            
            total_science_flybys += len(science_flybys)
            
            # Get weight
            w = science_flybys[0]['weight']
            
            # Get positions for seasonal calculation
            positions = [f['position'] for f in science_flybys]
            S_values = self.calculate_S(positions)
            
            # Calculate score for this body
            body_score = 0.0
            flyby_details = []
            
            for i, flyby in enumerate(science_flybys):
                v_inf = flyby['v_inf']
                F = self.calculate_F(v_inf)
                S = S_values[i]
                
                contribution = w * S * F
                body_score += contribution
                
                flyby_details.append({
                    'flyby_num': i + 1,
                    'v_inf': v_inf,
                    'F': F,
                    'S': S,
                    'contribution': contribution
                })
            
            body_scores[body_id] = {
                'weight': w,
                'num_flybys': len(science_flybys),
                'body_score': body_score,
                'flybys': flyby_details
            }
            
            total_score += body_score
        
        # Apply bonuses
        final_score = b * c * total_score
        
        return {
            'final_score': final_score,
            'base_score': total_score,
            'grand_tour_bonus': b,
            'time_bonus': c,
            'total_science_flybys': total_science_flybys,
            'bodies_visited': len(body_scores),
            'body_scores': body_scores
        }
    
    def print_score_report(self, score_result):
        """Print detailed score breakdown"""
        print("\n" + "="*70)
        print("GTOC13 SCORE REPORT")
        print("="*70)
        
        print(f"\n{'FINAL SCORE:':<30} {score_result['final_score']:>15.3f}")
        print(f"{'Base score:':<30} {score_result['base_score']:>15.3f}")
        print(f"{'Grand tour bonus (b):':<30} {score_result['grand_tour_bonus']:>15.2f}x")
        print(f"{'Time bonus (c):':<30} {score_result['time_bonus']:>15.3f}x")
        
        print(f"\n{'Total scientific flybys:':<30} {score_result['total_science_flybys']:>15d}")
        print(f"{'Bodies visited:':<30} {score_result['bodies_visited']:>15d}")
        
        print("\n" + "-"*70)
        print("BREAKDOWN BY BODY")
        print("-"*70)
        print(f"{'Body ID':<10} {'Weight':<10} {'Flybys':<10} {'Score':<15}")
        print("-"*70)
        
        for body_id, info in sorted(score_result['body_scores'].items()):
            print(f"{body_id:<10} {info['weight']:<10.1f} {info['num_flybys']:<10d} {info['body_score']:<15.3f}")
        
        # Detailed flyby info for each body
        print("\n" + "-"*70)
        print("DETAILED FLYBY INFORMATION")
        print("-"*70)
        
        for body_id, info in sorted(score_result['body_scores'].items()):
            print(f"\nBody ID: {body_id} (weight: {info['weight']:.1f})")
            print(f"  {'#':<5} {'V_inf (km/s)':<15} {'F':<10} {'S':<10} {'Score':<12}")
            print(f"  {'-'*5} {'-'*15} {'-'*10} {'-'*10} {'-'*12}")
            
            for flyby in info['flybys']:
                print(f"  {flyby['flyby_num']:<5} "
                      f"{flyby['v_inf']:<15.3f} "
                      f"{flyby['F']:<10.4f} "
                      f"{flyby['S']:<10.4f} "
                      f"{flyby['contribution']:<12.3f}")
        
        print("\n" + "="*70)


def example_score_calculation():
    """Example: Calculate score for a simple trajectory"""
    
    scorer = GTOC13Scorer()
    
    # Example trajectory with 2 flybys
    flybys = {
        3: [  # Eden (body_id=3, weight=2)
            {
                'time': 50 * scorer.YEAR,
                'position': np.array([1.0, 0.5, 0.1]) * scorer.AU,
                'v_inf': 12.5,  # km/s
                'is_science': True,
                'weight': 2.0
            }
        ],
        5: [  # Beyonce (body_id=5, weight=7)
            {
                'time': 80 * scorer.YEAR,
                'position': np.array([3.0, 4.0, 0.2]) * scorer.AU,
                'v_inf': 8.9,  # km/s
                'is_science': True,
                'weight': 7.0
            }
        ]
    }
    
    # Calculate score (submission at day 0 = full time bonus)
    result = scorer.calculate_trajectory_score(flybys, submission_time_days=0)
    
    # Print report
    scorer.print_score_report(result)


if __name__ == '__main__':
    example_score_calculation()