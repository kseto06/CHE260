import numpy as np
import matplotlib.pyplot as plt

def load_experiment_data(filename: str) -> np.ndarray:
    with open(filename, 'r') as f:
        lines = f.readlines()

    data_lines = []
    for line in lines:
        if line.strip() and line[0].isdigit(): #look for valid number
            data_lines.append(line)

    data = np.loadtxt(data_lines)
    return data

def analyze_mass_flow(filename: str, start_time: int = None, end_time: int = None) -> None:
    # Load data
    data = load_experiment_data(f'data/{filename}')

    P_ATM_KPA = 99.3
    time = data[:, 0] # Time (s)
    T1 = data[:, 1] # Temperature (deg C)
    P1 = data[:, 5] # P1 (psi)
    mass_flow = data[:, 7] # Mass Flowrate (g/min)

    # Convert flow rate to g/s for integration
    mass_flow_gs = mass_flow / 60.0

    # Integrate to find total mass added (uses trapezoidal rule)
    m_added = np.trapz(mass_flow_gs, time)

    # Start and end indices for flow, convert to kPa and K
    start_idx = np.argmin(np.abs(time - start_time))
    end_idx = np.argmin(np.abs(time - end_time))
    P1_i = np.mean(P1[start_idx]) * 6.89476 + P_ATM_KPA
    P2_f = np.mean(P1[end_idx]) * 6.89476 + P_ATM_KPA
    T1_i = np.mean(T1[start_idx]) + 273.15
    T2_f = np.mean(T1[end_idx]) + 273.15

    # Compute using formula
    ratio = (P2_f * T1_i) / (P1_i * T2_f)
    m_left_tank = m_added * (1 + 1/(ratio - 1))

    # Plot mass flow rate vs. time
    plt.figure(figsize=(8,5))
    plt.plot(time, mass_flow, '-', color='red', alpha=0.7)
    plt.title("Mass Flow Rate vs. Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Mass Flow Rate (g/min)")
    plt.grid(True)
    plt.savefig(f'figures/{filename}.png')
    plt.show()

    print(f"--- Results for {filename} ---")
    print(f"m_added = {m_added:.2f} g")
    print(f"P1 = {P1_i:.2f} kPa, T1 = {T1_i:.2f} K")
    print(f"P2 = {P2_f:.2f} kPa, T2 = {T2_f:.2f} K")
    print(f"m_left_tank = {m_left_tank:.2f} g\n")

def part1() -> None:
    analyze_mass_flow('Lab 2 - Part 1a', 20, 53)
    analyze_mass_flow('Lab 2 - Part 1b', 26, 90)
    analyze_mass_flow('Lab 2 - Part 1c', 10, 50)
    analyze_mass_flow('Lab 2 - Part 1d', 12, 70)
    """
    Part 1 Results:

    --- Results for Lab 2 - Part 1a ---
    m_added = 22.80 g
    P1 = 127.57 kPa, T1 = 299.35 K
    P2 = 370.95 kPa, T2 = 304.25 K
    m_left_tank = 35.05 g

    --- Results for Lab 2 - Part 1b ---
    m_added = 46.94 g
    P1 = 119.29 kPa, T1 = 304.25 K
    P2 = 635.71 kPa, T2 = 309.45 K
    m_left_tank = 58.01 g

    --- Results for Lab 2 - Part 1c ---
    m_added = 24.09 g
    P1 = 104.13 kPa, T1 = 307.95 K
    P2 = 375.09 kPa, T2 = 312.45 K
    m_left_tank = 33.54 g

    --- Results for Lab 2 - Part 1d ---
    m_added = 40.58 g
    P1 = 119.29 kPa, T1 = 309.45 K
    P2 = 580.55 kPa, T2 = 312.75 K
    m_left_tank = 51.22 g
    """

def analyze_heater_energy(filename, steady_start=None, mass_left_tank=None) -> None:
    with open(f"data/{filename}", "r") as f:
        lines = f.readlines()

    data_lines = [line for line in lines if line.strip() and line[0].isdigit()] #check for valid number
    data = np.loadtxt(data_lines)

    time = data[:, 0] # Time (s)
    temp = data[:, 1] # Temperature (°C)
    heater_energy = data[:, 8] # Heater Energy (kJ)

    # steady start time
    steady_index = np.argmin(np.abs(time - steady_start))

    # Q's
    Q_added = heater_energy[steady_index]
    Q_final = heater_energy[-1]
    Q_lost = (Q_final - Q_added)

    T_ambient = 24.3 # deg C
    k_acrylic = 0.19 # W/m*K
    l = 0.28575 # height of tank (m)
    r1 = 0.09208 # inner radius (m)
    r2 = 0.10610 # outer radius (m)

    # Compute heat transfer rate
    delta_T = np.mean(temp[steady_index:]) - T_ambient
    Qdot_wall = 2 * np.pi * k_acrylic * l * delta_T / np.log(r2 / r1)
    dt = time[-1] - time[steady_index]
    Q_wall = Qdot_wall * dt / 1000

    Q_plate = Q_lost - Q_wall

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Left y-axis (Heater Energy)
    l1, = ax1.plot(time, heater_energy, color='red', linewidth=2.5, label='Heater Energy (kJ)')
    steady_line = ax1.axvline(steady_start, color='blue', linestyle='--', linewidth=1.2, label=f'Steady-state start = {steady_start}s')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Heater Energy (kJ)", color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True)
    ax1.legend()

    # Right y-axis (Temperature)
    ax2 = ax1.twinx()
    l2, = ax2.plot(time, temp, color='blue', linewidth=2, label='Temperature (°C)')
    ax2.set_ylabel("Temperature (°C)", color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    ax1.legend(handles=[l1, l2, steady_line], loc='best')
    plt.title(f"Heater Energy vs Time")
    fig.tight_layout()
    plt.savefig(f"figures/{filename}.png", dpi=300)
    plt.show()

    # compute specific heat capacity
    if mass_left_tank is not None:
        c_v = Q_lost / (mass_left_tank * 1e-3 * (np.mean(temp[steady_index:])+273.15 - T_ambient+273.15)) # kJ/kgK

    # results
    print(f"--- Results for {filename} ---")
    print(f"Steady-state start: {steady_start:.2f} s")
    print(f"T_ambient = {T_ambient:.2f} °C")
    print(f"ΔT across wall = {delta_T:.2f} °C")
    print(f"Q_dot_wall = {Qdot_wall:.2f} W")
    print(f"Q_wall = {Q_wall:.2f} kJ")
    print(f"Q_added = {Q_added:.2f} kJ")
    print(f"Q_final = {Q_final:.2f} kJ")
    print(f"Q_lost = Q_final - Q_added = {Q_lost:.2f} kJ")
    print(f"Q_plate = Q_lost - Q_wall = {Q_plate:.2f} kJ")
    print(f"c_v: {c_v:.2f} kJ/kgK\n")

def part2() -> None:
    analyze_heater_energy("Lab 2 - Part 2a", 90, 35.03)
    analyze_heater_energy("Lab 2 - Part 2b", 75, 58.01)
    analyze_heater_energy("Lab 2 - Part 2c", 120, 33.54)
    analyze_heater_energy("Lab 2 - Part 2d", 105, 51.22)
    '''
    --- Results for Lab 2 - Part 2a ---
    Steady-state start: 90.00 s
    T_ambient = 24.30 °C
    ΔT across wall = 15.75 °C
    Q_dot_wall = 37.90 W
    Q_wall = 10.80 kJ
    Q_added = 46.20 kJ
    Q_final = 66.40 kJ
    Q_lost = Q_final - Q_added = 20.20 kJ
    Q_plate = Q_lost - Q_wall = 9.40 kJ
    c_v: 1.03 kJ/kgK

    --- Results for Lab 2 - Part 2b ---
    Steady-state start: 75.00 s
    T_ambient = 24.30 °C
    ΔT across wall = 15.70 °C
    Q_dot_wall = 37.79 W
    Q_wall = 10.72 kJ
    Q_added = 32.40 kJ
    Q_final = 58.60 kJ
    Q_lost = Q_final - Q_added = 26.20 kJ
    Q_plate = Q_lost - Q_wall = 15.48 kJ
    c_v: 0.80 kJ/kgK

    --- Results for Lab 2 - Part 2c ---
    Steady-state start: 120.00 s
    T_ambient = 24.30 °C
    ΔT across wall = 35.46 °C
    Q_dot_wall = 85.34 W
    Q_wall = 20.87 kJ
    Q_added = 92.90 kJ
    Q_final = 138.50 kJ
    Q_lost = Q_final - Q_added = 45.60 kJ
    Q_plate = Q_lost - Q_wall = 24.73 kJ
    c_v: 2.34 kJ/kgK

    --- Results for Lab 2 - Part 2d ---
    Steady-state start: 105.00 s
    T_ambient = 24.30 °C
    ΔT across wall = 35.13 °C
    Q_dot_wall = 84.56 W
    Q_wall = 24.39 kJ
    Q_added = 87.30 kJ
    Q_final = 151.10 kJ
    Q_lost = Q_final - Q_added = 63.80 kJ
    Q_plate = Q_lost - Q_wall = 39.41 kJ
    c_v: 2.14 kJ/kgK
    '''

if __name__ == "__main__": 
    part2()