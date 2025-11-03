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

def part2() -> None:
    raise NotImplementedError()

if __name__ == "__main__": 
    part1()