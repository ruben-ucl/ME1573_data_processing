import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import fsolve
import pandas as pd

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

class MeltPoolThermalAnalysis:
    """
    Analyzes heat transfer and accumulation in shrinking melt pools
    to test the hypothesis for keyhole flickering mechanisms
    """
    
    def __init__(self, material='aluminium'):
        """Initialize with material properties"""
        
        # Material properties database
        materials = {
            'aluminium': {
                'k_solid': 100,      # W/m·K
                'k_liquid': 91,      # W/m·K
                'rho_liquid': 2400,  # kg/m³
                'cp_liquid': 1080,   # J/kg·K
                'T_melt': 933,       # K (660°C)
                'T_ambient': 298,    # K
                'H_fusion': 3.98e5,  # J/kg
                'T_boil': 2743,      # K
                'name': 'Aluminium (AlSi10Mg)'
            },
            'steel316L': {
                'k_solid': 21.5,     # W/m·K (average)
                'k_liquid': 30.5,    # W/m·K
                'rho_liquid': 6765,  # kg/m³
                'cp_liquid': 1873,   # J/kg·K
                'T_melt': 1723,      # K
                'T_ambient': 298,    # K
                'H_fusion': 2.6e5,   # J/kg
                'T_boil': 3090,      # K
                'name': 'Stainless Steel 316L'
            },
            'ti64': {
                'k_solid': 20,       # W/m·K (average)
                'k_liquid': 33.4,    # W/m·K
                'rho_liquid': 3920,  # kg/m³
                'cp_liquid': 831,    # J/kg·K
                'T_melt': 1923,      # K
                'T_ambient': 298,    # K
                'H_fusion': 2.86e5,  # J/kg
                'T_boil': 3560,      # K
                'name': 'Ti-6Al-4V'
            }
        }
        
        self.props = materials[material]
        
        # Process parameters
        self.laser_power = 200  # W
        self.absorption = 0.15  # -
        self.Q_input = self.laser_power * self.absorption  # Net power (W)
        
        # Geometry parameters
        self.aspect_ratio = 0.5  # depth/radius ratio for hemispherical pool
        
    def calculate_pool_properties(self, radius):
        """Calculate geometric and thermal properties for a given pool radius"""
        
        R = radius
        h = R * self.aspect_ratio  # pool depth
        
        # Geometry (assuming hemispherical/ellipsoidal shape)
        volume = (2/3) * np.pi * R**2 * h  # m³
        surface_area = 2 * np.pi * R * h  # approximate free surface area
        interface_area = np.pi * R**2  # substrate interface area
        mass = volume * self.props['rho_liquid']  # kg
        
        # Thermal properties
        thermal_inertia = mass * self.props['cp_liquid']  # J/K
        stored_energy = thermal_inertia * (self.props['T_melt'] - self.props['T_ambient'])  # J
        
        # Heat transfer
        # Conduction through substrate (dominant loss mechanism)
        # Using characteristic length = pool depth
        temp_gradient = (self.props['T_melt'] - self.props['T_ambient']) / h  # K/m
        conduction_flux = self.props['k_solid'] * temp_gradient  # W/m²
        Q_conduction = conduction_flux * interface_area  # W
        
        # Net heat balance
        Q_net = self.Q_input - Q_conduction  # W
        net_ratio = Q_net / self.Q_input if self.Q_input > 0 else 0
        
        # Temperature rise rate (if net positive)
        dT_dt = Q_net / thermal_inertia if thermal_inertia > 0 else 0  # K/s
        
        # Superheat potential (temperature rise in 1 ms with constant net power)
        superheat_1ms = dT_dt * 0.001  # K
        
        # Time to reach boiling point (if net positive)
        delta_T_to_boil = self.props['T_boil'] - self.props['T_melt']  # K
        time_to_boil = delta_T_to_boil / dT_dt if dT_dt > 0 else np.inf  # s
        
        # Stefan number (importance of latent heat)
        stefan_number = (self.props['cp_liquid'] * (self.props['T_melt'] - self.props['T_ambient'])) / self.props['H_fusion']
        
        return {
            'radius': R,
            'depth': h,
            'volume': volume,
            'surface_area': surface_area,
            'interface_area': interface_area,
            'mass': mass,
            'thermal_inertia': thermal_inertia,
            'stored_energy': stored_energy,
            'temp_gradient': temp_gradient,
            'conduction_flux': conduction_flux,
            'Q_conduction': Q_conduction,
            'Q_net': Q_net,
            'net_ratio': net_ratio,
            'dT_dt': dT_dt,
            'superheat_1ms': superheat_1ms,
            'time_to_boil': time_to_boil,
            'stefan_number': stefan_number
        }
    
    def steady_state_temperature_field(self, pool_props, n_points=100):
        """
        Calculate steady-state temperature field using simplified conduction model
        within the melt pool and substrate
        """
        
        R = pool_props['radius']
        h = pool_props['depth']
        
        # Create coordinate system (cylindrical: r, z)
        r_max = 3 * R  # extend to 3× pool radius
        z_max = 5 * h  # extend deep into substrate
        
        r = np.linspace(0, r_max, n_points)
        z = np.linspace(-h, z_max, n_points)
        R_grid, Z_grid = np.meshgrid(r, z)
        
        # Temperature field using superposition of:
        # 1. Source at origin (laser input)
        # 2. Heat sink boundary at infinity
        
        T = np.zeros_like(R_grid)
        
        for i in range(len(z)):
            for j in range(len(r)):
                r_val = R_grid[i, j]
                z_val = Z_grid[i, j]
                
                # Distance from source (at surface, r=0)
                dist = np.sqrt(r_val**2 + z_val**2)
                
                if dist < 1e-9:
                    dist = R / 10  # avoid singularity
                
                # Simplified steady-state solution
                # In liquid pool region (z < 0, r < R)
                if z_val <= 0 and r_val <= R:
                    # Assume relatively uniform temperature near melting point
                    # with gradient toward edges
                    edge_dist = np.sqrt((r_val/R)**2 + (z_val/h)**2)
                    T[i, j] = self.props['T_melt'] + pool_props['superheat_1ms'] * (1 - edge_dist)
                else:
                    # In solid substrate - exponential decay
                    # Using modified Rosenthal-type solution
                    k_eff = self.props['k_solid']
                    Q_eff = self.Q_input
                    
                    T[i, j] = self.props['T_ambient'] + \
                              (Q_eff / (2 * np.pi * k_eff * dist)) * \
                              np.exp(-dist / (2 * R))
        
        return R_grid, Z_grid, T
    
    def find_critical_radius(self):
        """Find the critical radius where Q_input = Q_conduction"""
        
        def net_power(R):
            props = self.calculate_pool_properties(R)
            return props['Q_net']
        
        # Search in reasonable range
        R_crit = fsolve(net_power, x0=50e-6)[0]
        
        return R_crit
    
    def compare_pool_sizes(self, R_large, R_critical, R_small):
        """Compare large, critical, and small pool in detail"""

        print(f"\n{'='*80}")
        print(f"MELT POOL THINNING ANALYSIS: {self.props['name']}")
        print(f"{'='*80}")
        print(f"Laser Power: {self.laser_power} W")
        print(f"Absorption: {self.absorption:.2%}")
        print(f"Net Input Power: {self.Q_input:.1f} W")
        print(f"Critical Radius: {R_critical*1e6:.1f} μm")
        print(f"{'='*80}\n")

        large = self.calculate_pool_properties(R_large)
        critical = self.calculate_pool_properties(R_critical)
        small = self.calculate_pool_properties(R_small)

        # Create comparison dataframe
        comparison_data = {
            'Parameter': [],
            'Large (+50%)': [],
            'Critical': [],
            'Small (-50%)': [],
            'Ratio (L/S)': [],
            'Units': []
        }

        params = [
            ('Radius', 'radius', 1e6, 'μm'),
            ('Depth', 'depth', 1e6, 'μm'),
            ('Volume', 'volume', 1e9, 'mm³'),
            ('Mass', 'mass', 1e6, 'mg'),
            ('Interface Area', 'interface_area', 1e6, 'mm²'),
            ('Thermal Inertia', 'thermal_inertia', 1e6, 'μJ/K'),
            ('Stored Energy', 'stored_energy', 1e3, 'mJ'),
            ('Conduction Loss', 'Q_conduction', 1, 'W'),
            ('Net Power', 'Q_net', 1, 'W'),
            ('Net/Input Ratio', 'net_ratio', 100, '%'),
            ('Heating Rate', 'dT_dt', 1e-3, 'K/ms'),
            ('Superheat in 1ms', 'superheat_1ms', 1, 'K'),
            ('Time to Boiling', 'time_to_boil', 1e3, 'ms'),
        ]

        def format_val(val):
            if np.isinf(val) or np.isnan(val):
                return "N/A"
            return f"{val:.2e}" if abs(val) > 1e3 or (abs(val) < 0.01 and val != 0) else f"{val:.2f}"

        for name, key, scale, unit in params:
            large_val = large[key] * scale
            critical_val = critical[key] * scale
            small_val = small[key] * scale

            if small_val != 0 and not np.isinf(large_val) and not np.isinf(small_val):
                ratio = large_val / small_val
            else:
                ratio = np.nan

            comparison_data['Parameter'].append(name)
            comparison_data['Large (+50%)'].append(format_val(large_val))
            comparison_data['Critical'].append(format_val(critical_val))
            comparison_data['Small (-50%)'].append(format_val(small_val))
            comparison_data['Ratio (L/S)'].append(f"{ratio:.2f}" if not np.isnan(ratio) else "N/A")
            comparison_data['Units'].append(unit)

        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        print(f"\n{'='*80}\n")

        # Key insights
        print("KEY INSIGHTS:")
        print(f"1. Volume scales as R³: Ratio (L/S) = {large['volume']/small['volume']:.1f}×")
        print(f"2. Conduction loss scales as R: Ratio (L/S) = {large['Q_conduction']/small['Q_conduction']:.1f}×")
        print(f"3. Thermal inertia scales as R³: Ratio (L/S) = {large['thermal_inertia']/small['thermal_inertia']:.1f}×")

        if small['dT_dt'] > 0 and large['dT_dt'] > 0:
            print(f"4. Heating rate scales as R⁻²: Ratio (S/L) = {small['dT_dt']/large['dT_dt']:.1f}×")
            print(f"\n   → Small pool heats {small['dT_dt']/large['dT_dt']:.0f}× FASTER than large pool!")

        print(f"\n5. Heat balance at critical radius:")
        print(f"   Large pool:    Q_net = {large['Q_net']:.1f} W (net {'heating' if large['Q_net'] > 0 else 'cooling'})")
        print(f"   Critical pool: Q_net = {critical['Q_net']:.1f} W (equilibrium)")
        print(f"   Small pool:    Q_net = {small['Q_net']:.1f} W (net {'heating' if small['Q_net'] > 0 else 'cooling'})")

        if small['Q_net'] > 0 and large['Q_net'] < 0:
            print(f"\n   → Below critical radius: pool superheats toward evaporation!")
            print(f"   → Above critical radius: pool cools and stabilizes!")

        print(f"\n{'='*80}\n")

        return large, critical, small
    
    def visualize_analysis(self):
        """Create comprehensive visualization of the analysis"""

        # Find critical radius first
        try:
            R_crit = self.find_critical_radius()
            print(f"CRITICAL RADIUS (Q_net = 0): {R_crit*1e6:.1f} μm")
            print(f"Below this radius, pool will superheat toward evaporation.\n")
        except:
            R_crit = 100e-6  # Default fallback
            print("Critical radius not found, using default 100 μm.\n")

        # Calculate pool sizes based on critical radius ± 50%
        R_large = R_crit * 1.5      # Critical + 50%
        R_small = R_crit * 0.5      # Critical - 50%

        large, critical, small = self.compare_pool_sizes(R_large, R_crit, R_small)

        # Calculate temperature fields for all three cases
        R_grid_large, Z_grid_large, T_large = self.steady_state_temperature_field(large, n_points=80)
        R_grid_crit, Z_grid_crit, T_crit = self.steady_state_temperature_field(critical, n_points=80)
        R_grid_small, Z_grid_small, T_small = self.steady_state_temperature_field(small, n_points=80)

        # Create figure with subplots
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

        levels = np.linspace(self.props['T_ambient'], self.props['T_melt']+100, 20)

        # 1. Temperature field - Large pool (+50%)
        ax1 = fig.add_subplot(gs[0, 0])
        contour1 = ax1.contourf(R_grid_large*1e6, Z_grid_large*1e6, T_large,
                               levels=levels, cmap='hot', extend='both')
        ax1.contour(R_grid_large*1e6, Z_grid_large*1e6, T_large,
                   levels=[self.props['T_melt']], colors='cyan', linewidths=2, linestyles='--')
        ax1.axhline(y=0, color='white', linestyle='-', linewidth=1, alpha=0.5)
        ax1.set_xlabel('Radial Distance (μm)')
        ax1.set_ylabel('Depth (μm)')
        ax1.set_title(f'Large Pool: R={R_large*1e6:.0f}μm (+50%)')
        ax1.set_aspect('equal')
        plt.colorbar(contour1, ax=ax1, label='Temperature (K)')
        ax1.invert_yaxis()

        # 2. Temperature field - Critical pool
        ax2 = fig.add_subplot(gs[0, 1])
        contour2 = ax2.contourf(R_grid_crit*1e6, Z_grid_crit*1e6, T_crit,
                               levels=levels, cmap='hot', extend='both')
        ax2.contour(R_grid_crit*1e6, Z_grid_crit*1e6, T_crit,
                   levels=[self.props['T_melt']], colors='cyan', linewidths=2, linestyles='--')
        ax2.axhline(y=0, color='white', linestyle='-', linewidth=1, alpha=0.5)
        ax2.set_xlabel('Radial Distance (μm)')
        ax2.set_ylabel('Depth (μm)')
        ax2.set_title(f'Critical Pool: R={R_crit*1e6:.0f}μm')
        ax2.set_aspect('equal')
        plt.colorbar(contour2, ax=ax2, label='Temperature (K)')
        ax2.invert_yaxis()

        # 3. Temperature field - Small pool (-50%)
        ax3 = fig.add_subplot(gs[0, 2])
        contour3 = ax3.contourf(R_grid_small*1e6, Z_grid_small*1e6, T_small,
                               levels=levels, cmap='hot', extend='both')
        ax3.contour(R_grid_small*1e6, Z_grid_small*1e6, T_small,
                   levels=[self.props['T_melt']], colors='cyan', linewidths=2, linestyles='--')
        ax3.axhline(y=0, color='white', linestyle='-', linewidth=1, alpha=0.5)
        ax3.set_xlabel('Radial Distance (μm)')
        ax3.set_ylabel('Depth (μm)')
        ax3.set_title(f'Small Pool: R={R_small*1e6:.0f}μm (-50%)')
        ax3.set_aspect('equal')
        plt.colorbar(contour3, ax=ax3, label='Temperature (K)')
        ax3.invert_yaxis()
        
        # 4. Heat balance comparison (3 pools)
        ax4 = fig.add_subplot(gs[1, 0])
        categories = ['Large\n(+50%)', 'Critical', 'Small\n(-50%)']
        Q_in = [self.Q_input, self.Q_input, self.Q_input]
        Q_cond = [large['Q_conduction'], critical['Q_conduction'], small['Q_conduction']]
        Q_net_vals = [large['Q_net'], critical['Q_net'], small['Q_net']]

        x = np.arange(len(categories))
        width = 0.25

        ax4.bar(x - width, Q_in, width, label='Input Power', color='gold', edgecolor='black')
        ax4.bar(x, Q_cond, width, label='Conduction Loss', color='steelblue', edgecolor='black')
        ax4.bar(x + width, Q_net_vals, width, label='Net Available',
                color=['green' if q > 0 else ('gray' if abs(q) < 0.1 else 'red') for q in Q_net_vals], edgecolor='black')

        ax4.set_ylabel('Power (W)')
        ax4.set_title('Heat Balance Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories)
        ax4.legend()
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.grid(True, alpha=0.3)

        # 5. Conduction loss vs radius
        ax5 = fig.add_subplot(gs[1, 1])
        radii = np.linspace(20e-6, 200e-6, 50)
        Q_cond_array = []
        for R in radii:
            props = self.calculate_pool_properties(R)
            Q_cond_array.append(props['Q_conduction'])

        ax5.plot(radii*1e6, Q_cond_array, 'b-', linewidth=2, label='Conduction Loss')
        ax5.axhline(y=self.Q_input, color='gold', linestyle='--', linewidth=2, label='Input Power')
        ax5.axvline(x=R_large*1e6, color='blue', linestyle='--', alpha=0.5, label='Large')
        ax5.axvline(x=R_crit*1e6, color='green', linestyle='--', alpha=0.5, label='Critical')
        ax5.axvline(x=R_small*1e6, color='red', linestyle='--', alpha=0.5, label='Small')
        ax5.plot(R_crit*1e6, self.Q_input, 'go', markersize=10, label=f'Critical R={R_crit*1e6:.0f}μm')

        ax5.set_xlabel('Pool Radius (μm)')
        ax5.set_ylabel('Power (W)')
        ax5.set_title('Conduction Loss vs Pool Size (Q ∝ R)')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        ax5.fill_between(radii*1e6, 0, Q_cond_array, where=np.array(Q_cond_array) < self.Q_input,
                         alpha=0.3, color='green')
        ax5.fill_between(radii*1e6, 0, Q_cond_array, where=np.array(Q_cond_array) >= self.Q_input,
                         alpha=0.3, color='red')

        # 6. Heating rate vs radius
        ax6 = fig.add_subplot(gs[1, 2])
        heating_rates = []
        for R in radii:
            props = self.calculate_pool_properties(R)
            heating_rates.append(props['dT_dt'] * 1e-3)  # K/ms

        ax6.semilogy(radii*1e6, heating_rates, 'r-', linewidth=2)
        ax6.axvline(x=R_large*1e6, color='blue', linestyle='--', alpha=0.5, label='Large (+50%)')
        ax6.axvline(x=R_crit*1e6, color='green', linestyle='--', alpha=0.5, label='Critical')
        ax6.axvline(x=R_small*1e6, color='red', linestyle='--', alpha=0.5, label='Small (-50%)')

        ax6.set_xlabel('Pool Radius (μm)')
        ax6.set_ylabel('Heating Rate (K/ms)')
        ax6.set_title('Heating Rate vs Pool Size (dT/dt ∝ R⁻²)')
        ax6.legend()
        ax6.grid(True, alpha=0.3, which='both')

        # 7. Thermal response time
        ax7 = fig.add_subplot(gs[2, 0])
        thermal_times = []
        for R in radii:
            # Thermal diffusion time: t ~ R²/α
            alpha_thermal = self.props['k_liquid'] / (self.props['rho_liquid'] * self.props['cp_liquid'])
            t_thermal = R**2 / alpha_thermal  # s
            thermal_times.append(t_thermal * 1e3)  # ms

        ax7.plot(radii*1e6, thermal_times, 'g-', linewidth=2)
        ax7.axvline(x=R_large*1e6, color='blue', linestyle='--', alpha=0.5, label='Large (+50%)')
        ax7.axvline(x=R_crit*1e6, color='green', linestyle='--', alpha=0.5, label='Critical')
        ax7.axvline(x=R_small*1e6, color='red', linestyle='--', alpha=0.5, label='Small (-50%)')
        ax7.axhline(y=1.0, color='orange', linestyle='--', linewidth=1, label='1 ms (flickering period)')
        
        ax7.set_xlabel('Pool Radius (μm)')
        ax7.set_ylabel('Thermal Response Time (ms)')
        ax7.set_title('Thermal Diffusion Time: τ ~ R²/α')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        ax7.set_yscale('log')
        
        # 8. Time to boiling
        ax8 = fig.add_subplot(gs[2, 1])
        times_to_boil = []
        for R in radii:
            props = self.calculate_pool_properties(R)
            t_boil = props['time_to_boil'] * 1e3  # ms
            times_to_boil.append(t_boil if t_boil < 1e6 else np.nan)
        
        ax8.semilogy(radii*1e6, times_to_boil, 'm-', linewidth=2)
        ax8.axvline(x=R_large*1e6, color='blue', linestyle='--', alpha=0.5, label='Large (+50%)')
        ax8.axvline(x=R_crit*1e6, color='green', linestyle='--', alpha=0.5, label='Critical')
        ax8.axvline(x=R_small*1e6, color='red', linestyle='--', alpha=0.5, label='Small (-50%)')
        ax8.axhline(y=1.0, color='orange', linestyle='--', linewidth=1, label='1 ms')
        ax8.axhline(y=100.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        ax8.set_xlabel('Pool Radius (μm)')
        ax8.set_ylabel('Time to Reach Boiling Point (ms)')
        ax8.set_title('Time to Evaporation')
        ax8.legend()
        ax8.grid(True, alpha=0.3, which='both')
        ax8.set_ylim([0.1, 1000])
        
        # 9. Phase diagram: Pool radius vs time
        ax9 = fig.add_subplot(gs[2, 2])
        
        # Simulate dynamic shrinking
        dt = 0.01e-3  # 0.01 ms time step
        t_max = 2e-3  # 2 ms simulation
        time_array = np.arange(0, t_max, dt)
        
        # Scenario: pool starts large, shrinks, then re-grows
        R_initial = R_large
        v_shrink = 0.05  # m/s shrinkage rate
        
        radius_history = []
        temp_history = []
        T_current = self.props['T_melt']
        R_current = R_initial
        
        phase = 'shrinking'
        
        for t in time_array:
            # Calculate current pool properties
            if R_current > 10e-6:  # Minimum radius
                props = self.calculate_pool_properties(R_current)
                
                # Update temperature
                dT = props['dT_dt'] * dt
                T_current += dT
                
                # Update radius based on phase
                if phase == 'shrinking':
                    if R_current > R_small:
                        R_current -= v_shrink * dt
                    else:
                        # Transition: small pool superheats
                        if T_current >= self.props['T_boil'] * 0.95:
                            phase = 'drilling'
                            v_drill = 0.1  # m/s drilling rate (faster)
                
                if phase == 'drilling':
                    R_current += v_drill * dt
                    if R_current >= R_large:
                        phase = 'stable'
                        R_current = R_large
                        T_current = self.props['T_melt']
            
            radius_history.append(R_current * 1e6)
            temp_history.append(T_current)
        
        ax9_twin = ax9.twinx()
        line1 = ax9.plot(time_array*1e3, radius_history, 'b-', linewidth=2, label='Pool Radius')
        line2 = ax9_twin.plot(time_array*1e3, temp_history, 'r-', linewidth=2, label='Temperature')

        ax9.axhline(y=R_crit*1e6, color='green', linestyle='--', linewidth=1, alpha=0.7, label='R_critical')
        ax9_twin.axhline(y=self.props['T_boil'], color='red', linestyle='--', linewidth=1, alpha=0.5, label='T_boil')
        ax9_twin.axhline(y=self.props['T_melt'], color='orange', linestyle='--', linewidth=1, alpha=0.5, label='T_melt')
        
        ax9.set_xlabel('Time (ms)')
        ax9.set_ylabel('Pool Radius (μm)', color='b')
        ax9_twin.set_ylabel('Temperature (K)', color='r')
        ax9.set_title('Dynamic Flickering Simulation')
        ax9.tick_params(axis='y', labelcolor='b')
        ax9_twin.tick_params(axis='y', labelcolor='r')
        ax9.grid(True, alpha=0.3)
        
        lines1, labels1 = ax9.get_legend_handles_labels()
        lines2, labels2 = ax9_twin.get_legend_handles_labels()
        ax9.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
        
        # Add overall title
        fig.suptitle(f'Melt Pool Thinning Analysis: {self.props["name"]}\n' + 
                     f'Hypothesis: Smaller pools accumulate heat faster → superheat → rapid re-drilling',
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('melt_pool_analysis.png', dpi=300, bbox_inches='tight')
        print("Figure saved as 'melt_pool_analysis.png'")
        plt.show()
        
        return fig

# Run the analysis
if __name__ == "__main__":
    # Create analyzer for aluminium
    print("="*80)
    print("MELT POOL THINNING HYPOTHESIS TESTING")
    print("="*80)

    analyzer = MeltPoolThermalAnalysis(material='aluminium')

    # Run comparison and visualization (radii calculated from critical radius ± 50%)
    fig = analyzer.visualize_analysis()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey findings:")
    print("1. Check the heat balance comparison (middle left)")
    print("2. Note the heating rate scaling: dT/dt ∝ R⁻²")
    print("3. Observe the critical radius where Q_input = Q_conduction")
    print("4. See the dynamic flickering simulation (bottom right)")
    print("\nThe visualizations show whether melt pool thinning can explain")
    print("the rapid transition from conduction → keyhole mode in ~1 ms.")
    print("="*80)