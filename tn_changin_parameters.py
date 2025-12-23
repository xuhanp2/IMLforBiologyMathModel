import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class CooperationDynamics:
    """
    Simulator for cooperation dynamics with group selection.
    Implements the Traulsen-Nowak model for n=3 as shown in the poster.
    """

    def __init__(self, R=3, S=0, T=5, P=1, omega_I=1, omega_G=0.1, Lambda=0.5, n=3):
        """
        Initialize parameters.

        Parameters:
        -----------
        R : float - Reward for mutual cooperation
        S : float - Sucker's payoff
        T : float - Temptation to defect
        P : float - Punishment for mutual defection
        omega_I : float - Individual selection strength
        omega_G : float - Group selection strength
        Lambda : float - Group selection rate (0 to 1)
        n : int - Group size (must be 3 for this implementation)
        """
        if n != 3:
            raise ValueError("This implementation only supports group size n=3")

        self.R = R
        self.S = S
        self.T = T
        self.P = P
        self.omega_I = omega_I
        self.omega_G = omega_G
        self.Lambda = Lambda
        self.n = n

    def derivatives(self, state, t):
        """
        Compute derivatives using the Traulsen-Nowak model for n=3.
        Implements equations (2)-(5) from the poster.

        Parameters:
        -----------
        state : array-like - [f0, f1, f2, f3]
        t : float - Time (unused, but required by odeint)

        Returns:
        --------
        array - [df0/dt, df1/dt, df2/dt, df3/dt]
        """
        f0, f1, f2, f3 = state
        R, S, T, P = self.R, self.S, self.T, self.P
        w_I, w_G = self.omega_I, self.omega_G
        Lambda = self.Lambda

        # Compute group fitness values
        G0 = P  # All defectors
        G1 = (R + 2*S + 2*T + 4*P) / 9  # 1 cooperator
        G2 = (4*R + 2*S + 2*T + P) / 9  # 2 cooperators
        G3 = R  # All cooperators

        # Compute denominator D for normalization
        D = (f0 * np.exp(w_G * G0) +
             f1 * np.exp(w_G * G1) +
             f2 * np.exp(w_G * G2) +
             f3 * np.exp(w_G * G3))

        # Individual payoffs at different compositions
        pi_C_1 = (R + 2*S) / 3  # Cooperator payoff with 1 cooperator
        pi_D_1 = (T + 2*P) / 3  # Defector payoff with 1 cooperator
        pi_C_2 = (2*R + S) / 3  # Cooperator payoff with 2 cooperators
        pi_D_2 = (2*T + P) / 3  # Defector payoff with 2 cooperators

        # Equation (2): df0/dt
        term1_f0 = f1 * (2 * np.exp(w_I * pi_D_1)) / (np.exp(w_I * pi_C_1) + 2 * np.exp(w_I * pi_D_1))
        term2_f0 = Lambda * f0 * (np.exp(w_G * G0) / D - 1)
        df0dt = term1_f0 + term2_f0

        # Equation (3): df1/dt
        term1_f1 = -f1 * (np.exp(w_I * pi_D_1)) / (np.exp(w_I * pi_D_1) + 2 * np.exp(w_I * pi_C_1))
        term2_f1 = -2 * f1 * (np.exp(w_I * pi_C_1)) / (np.exp(w_I * pi_C_1) + 2 * np.exp(w_I * pi_D_1))
        term3_f1 = 2 * f2 * (np.exp(w_I * pi_D_2)) / (2 * np.exp(w_I * pi_C_2) + np.exp(w_I * pi_D_2))
        term4_f1 = Lambda * f1 * (np.exp(w_G * G1) / D - 1)
        df1dt = term1_f1 + term2_f1 + term3_f1 + term4_f1

        # Equation (4): df2/dt
        term1_f2 = -2 * f2 * (np.exp(w_I * pi_C_2)) / (2 * np.exp(w_I * pi_C_2) + np.exp(w_I * pi_D_2))
        term2_f2 = -f2 * (np.exp(w_I * pi_D_2)) / (2 * np.exp(w_I * pi_D_2) + np.exp(w_I * pi_C_2))
        term3_f2 = 2 * f1 * (np.exp(w_I * pi_C_1)) / (np.exp(w_I * pi_C_1) + 2 * np.exp(w_I * pi_D_1))
        term4_f2 = 2 * f3 * (np.exp(w_I * pi_D_2)) / (2 * np.exp(w_I * pi_D_2) + np.exp(w_I * pi_C_2))
        term5_f2 = Lambda * f2 * (np.exp(w_G * G2) / D - 1)
        df2dt = term1_f2 + term2_f2 + term3_f2 + term4_f2 + term5_f2

        # Equation (5): df3/dt
        term1_f3 = f2 * (2 * np.exp(w_I * pi_C_2)) / (2 * np.exp(w_I * pi_C_2) + np.exp(w_I * pi_D_2))
        term2_f3 = Lambda * f3 * (np.exp(w_G * G3) / D - 1)
        df3dt = term1_f3 + term2_f3

        return np.array([df0dt, df1dt, df2dt, df3dt])

    def simulate(self, initial_state, t_max=20, n_points=1000):
        """
        Run simulation.

        Parameters:
        -----------
        initial_state : array-like - Initial frequencies [f0, f1, f2, f3]
        t_max : float - Maximum simulation time
        n_points : int - Number of time points

        Returns:
        --------
        t : array - Time points
        solution : array - Solution trajectories
        """
        # Normalize initial state
        initial_state = np.array(initial_state)
        initial_state = initial_state / np.sum(initial_state)

        # Time points
        t = np.linspace(0, t_max, n_points)

        # Solve ODE
        solution = odeint(self.derivatives, initial_state, t)

        # Normalize at each time step to prevent drift
        solution = solution / solution.sum(axis=1, keepdims=True)

        return t, solution

    def parameter_sweep(self, param_name, param_values, initial_state,
                       t_max=100, n_points=1000):
        """
        Sweep over a parameter and return f3 trajectories.

        Parameters:
        -----------
        param_name : str - 'Lambda', 'omega_G', or 'omega_I'
        param_values : array-like - Values to sweep over
        initial_state : array-like - Initial frequencies
        t_max : float - Maximum simulation time
        n_points : int - Number of time points

        Returns:
        --------
        t : array - Time points
        f3_trajectories : array - f3 values for each parameter value
        """
        # Store original parameter value
        original_value = getattr(self, param_name)

        t = None
        f3_trajectories = []

        for value in param_values:
            # Set parameter
            setattr(self, param_name, value)

            # Run simulation
            t, solution = self.simulate(initial_state, t_max, n_points)
            f3_trajectories.append(solution[:, 3])

        # Restore original parameter
        setattr(self, param_name, original_value)

        return t, np.array(f3_trajectories)

    def plot_parameter_sweep(self, param_name, param_values, initial_state,
                            t_max=100, n_points=1000):
        """
        Plot f3 dynamics for different parameter values.

        Parameters:
        -----------
        param_name : str - 'Lambda', 'omega_G', or 'omega_I'
        param_values : array-like - Values to sweep over
        initial_state : array-like - Initial frequencies
        t_max : float - Maximum simulation time
        n_points : int - Number of time points
        """
        t, f3_trajectories = self.parameter_sweep(
            param_name, param_values, initial_state, t_max, n_points
        )

        plt.figure(figsize=(10, 6))

        # Use colormap for different parameter values
        colors = plt.cm.viridis(np.linspace(0, 1, len(param_values)))

        for i, (value, f3) in enumerate(zip(param_values, f3_trajectories)):
            label_map = {
                'Lambda': f'Λ = {value:.2f}',
                'omega_G': f'ω_G = {value:.2f}',
                'omega_I': f'ω_I = {value:.2f}'
            }
            plt.plot(t, f3, color=colors[i], linewidth=2,
                    label=label_map.get(param_name, f'{param_name} = {value:.2f}'))

        plt.xlabel('Time', fontsize=12)
        plt.ylabel('f₃ (Frequency of All Cooperators)', fontsize=12)

        title_map = {
            'Lambda': 'Effect of Group Selection Rate (Λ) on Cooperation',
            'omega_G': 'Effect of Group Selection Strength (ω_G) on Cooperation',
            'omega_I': 'Effect of Individual Selection Strength (ω_I) on Cooperation'
        }
        plt.title(title_map.get(param_name, f'Effect of {param_name} on Cooperation'),
                 fontsize=14, fontweight='bold')

        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()

        return plt.gcf()

    def plot_all_states(self, initial_state, t_max=100, n_points=1000):
        """
        Plot all four state variables (f0, f1, f2, f3) over time.
        """
        t, solution = self.simulate(initial_state, t_max, n_points)

        plt.figure(figsize=(10, 6))

        labels = ['f₀ (All Defectors)', 'f₁ (1 Cooperator)',
                  'f₂ (2 Cooperators)', 'f₃ (All Cooperators)']
        colors = ['red', 'orange', 'lightblue', 'blue']

        for i in range(4):
            plt.plot(t, solution[:, i], label=labels[i],
                    linewidth=2.5, color=colors[i])

        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Evolution of Group Compositions', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()

        return plt.gcf()

    def plot_all_parameter_sweeps(self, initial_state, t_max=100, n_points=1000):
        """
        Create a 3-panel plot showing sweeps over Lambda, omega_G, and omega_I.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Define parameter ranges
        lambda_values = np.linspace(0.0, 70.0, 15)
        omega_G_values = np.linspace(0, 4, 15)
        omega_I_values = np.linspace(0, 4, 15)

        param_configs = [
            ('Lambda', lambda_values, 'Λ'),
            ('omega_G', omega_G_values, 'ω_G'),
            ('omega_I', omega_I_values, 'ω_I')
        ]

        for ax, (param_name, param_values, param_label) in zip(axes, param_configs):
            t, f3_trajectories = self.parameter_sweep(
                param_name, param_values, initial_state, t_max, n_points
            )

            # Use colormap
            colors = plt.cm.viridis(np.linspace(0, 1, len(param_values)))

            for i, (value, f3) in enumerate(zip(param_values, f3_trajectories)):
                ax.plot(t, f3, color=colors[i], linewidth=2.5,
                       label=f'{param_label} = {value:.2f}')

            ax.set_xlabel('Time', fontsize=11)
            ax.set_ylabel('f₃ (All Cooperators)', fontsize=11)
            title_map = {
                'Lambda': 'Effect of Group Selection Rate (Λ)',
                'omega_G': 'Effect of Group Selection Strength (ω_G)',
                'omega_I': 'Effect of Individual Selection Strength (ω_I)'
            }
            ax.set_title(title_map.get(param_name, f'Effect of {param_label}'),
                        fontsize=13, fontweight='bold')
            ax.legend(fontsize=8, loc='best', ncol=1)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)

        fig.suptitle(f'Traulsen-Nowak Model: Evolution of Cooperation (n=3)',
                    fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()

        return fig


# Example usage and demonstration
if __name__ == "__main__":
    print("="*60)
    print("TRAULSEN-NOWAK MODEL: PARAMETER SWEEP ANALYSIS (n=3)")
    print("="*60)

    # Initialize system with default parameters
    system = CooperationDynamics(
        R=3,      # Reward for mutual cooperation
        S=0,      # Sucker's payoff
        T=5,      # Temptation to defect
        P=1,      # Punishment for mutual defection
        omega_I=1,    # Individual selection strength
        omega_G=0.1,  # Group selection strength
        Lambda=0.5,   # Group selection rate
        n=3       # Group size
    )

    # Initial state (equal frequencies)
    initial_state = [0.25, 0.25, 0.25, 0.25]

    print("\nBase Parameters:")
    print(f"  R={system.R}, S={system.S}, T={system.T}, P={system.P}")
    print(f"  ω_I={system.omega_I}, ω_G={system.omega_G}, Λ={system.Lambda}, n={system.n}")
    print(f"\nInitial state: f₀={initial_state[0]}, f₁={initial_state[1]}, "
          f"f₂={initial_state[2]}, f₃={initial_state[3]}")

    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)

    # Plot all state variables
    print("\nCreating all-states trajectory plot...")
    fig0 = system.plot_all_states(initial_state, t_max=100)
    plt.savefig('cooperation_all_states.png', dpi=300, bbox_inches='tight')
    print("Saved: cooperation_all_states.png")

    # Create comprehensive 3-panel plot
    print("\nCreating 3-panel comparison plot...")
    fig = system.plot_all_parameter_sweeps(initial_state, t_max=100, n_points=1000)
    plt.savefig('cooperation_parameter_sweeps.png', dpi=300, bbox_inches='tight')
    print("Saved: cooperation_parameter_sweeps.png")

    # Create individual plots for each parameter
    print("\nCreating individual parameter sweep plots...")

    # Lambda sweep
    lambda_values = np.linspace(0.0, 70.0, 15)
    fig1 = system.plot_parameter_sweep('Lambda', lambda_values, initial_state)
    plt.savefig('cooperation_lambda_sweep.png', dpi=300, bbox_inches='tight')
    print("Saved: cooperation_lambda_sweep.png")

    # omega_G sweep
    omega_G_values = np.linspace(0, 4.0, 15)
    fig2 = system.plot_parameter_sweep('omega_G', omega_G_values, initial_state)
    plt.savefig('cooperation_omegaG_sweep.png', dpi=300, bbox_inches='tight')
    print("Saved: cooperation_omegaG_sweep.png")

    # omega_I sweep
    omega_I_values = np.linspace(0, 4.0, 15)
    fig3 = system.plot_parameter_sweep('omega_I', omega_I_values, initial_state)
    plt.savefig('cooperation_omegaI_sweep.png', dpi=300, bbox_inches='tight')
    print("Saved: cooperation_omegaI_sweep.png")

    print("\n" + "="*60)
    print("INTERPRETATION GUIDE")
    print("="*60)
    print("\nΛ (Lambda) - Group selection rate:")
    print("  • Higher Λ → More group selection → Faster cooperation")
    print("  • Lower Λ → More individual selection → Slower/less cooperation")
    print("  • Λ=0: Pure individual selection")
    print("  • Λ=1-10: Strong group selection")
    print("  • Λ>10: Very strong group selection dominance")

    print("\nω_G (omega_G) - Group selection strength:")
    print("  • Higher ω_G → Stronger fitness differences between groups")
    print("  • Amplifies the effect of group selection")
    print("  • ω_G<0: Inverts group fitness (defector groups favored)")
    print("  • ω_G=0: No fitness amplification (neutral)")
    print("  • ω_G>0: Standard group selection (cooperator groups favored)")

    print("\nω_I (omega_I) - Individual selection strength:")
    print("  • Higher ω_I → Stronger within-group selection")
    print("  • Amplifies differences in individual payoffs")
    print("  • ω_I<0: Cooperation favored at individual level")
    print("  • ω_I=0: Neutral individual selection")
    print("  • ω_I>0: Defection favored at individual level")

    print("\n" + "="*60)
    print("MODEL DETAILS")
    print("="*60)
    print("\nThis implementation follows the Traulsen-Nowak model")
    print("equations (2)-(5) from the poster for group size n=3:")
    print("  • f₀: Groups with 0 cooperators (all defectors)")
    print("  • f₁: Groups with 1 cooperator")
    print("  • f₂: Groups with 2 cooperators")
    print("  • f₃: Groups with 3 cooperators (all cooperators)")
    print("\nThe model includes:")
    print("  • Within-group frequency-dependent selection")
    print("  • Between-group selection based on group fitness")
    print("  • Exponential fitness functions (Fermi update rule)")

    print("\n" + "="*60)

    # Show all plots
    plt.show()

    print("\nAnalysis complete!")