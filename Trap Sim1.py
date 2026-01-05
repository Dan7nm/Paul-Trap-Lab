import numpy as np
import matplotlib.pyplot as plt


def trap_solver(m,q,V0,Omega,r0,g,gamma1,gamma2,dt,T,x0,vx0):
    steps = int(T/dt)
    x = np.zeros(steps)
    vx = np.zeros(steps)
    t = np.arange(0, T, dt)
    x[0] = x0
    vx[0] = vx0

    def acceleration(x, vx, t):
        E = V0 * np.cos(Omega * t) * x / r0**2  # E = d(phi)/dx = (V0/r0^2) * x cos(Omega t)
        F = q * E - m * g
        # Nonlinear drag: F_drag = -gamma * |v|^(n-1) * v
        F_drag = -gamma1 * vx - gamma2 * np.abs(vx) * vx
        return (F + F_drag) / m

    for i in range(steps-1):
        # RK4 step
        k1_v = acceleration(x[i], vx[i], t[i]) * dt
        k1_x = vx[i] * dt

        k2_v = acceleration(x[i] + 0.5*k1_x, vx[i] + 0.5*k1_v, t[i] + 0.5*dt) * dt
        k2_x = (vx[i] + 0.5*k1_v) * dt

        k3_v = acceleration(x[i] + 0.5*k2_x, vx[i] + 0.5*k2_v, t[i] + 0.5*dt) * dt
        k3_x = (vx[i] + 0.5*k2_v) * dt

        k4_v = acceleration(x[i] + k3_x, vx[i] + k3_v, t[i] + dt) * dt
        k4_x = (vx[i] + k3_v) * dt

        vx[i+1] = vx[i] + (k1_v + 2*k2_v + 2*k3_v + k4_v)/6
        x[i+1] = x[i] + (k1_x + 2*k2_x + 2*k3_x + k4_x)/6



    # mask = t > T*0.2  #T / 2
    # t = t[mask]
    # x = x[mask]
    # plt.figure(figsize=(8,4))
    # plt.plot(t, x)
    # plt.xlabel('Time (s)')
    # plt.ylabel('x (m)')
    # plt.title('Particle motion in Paul trap with nonlinear drag')
    # plt.grid(True)
    # plt.show()

    # plt.figure(figsize=(10, 6))
    #
    # plt.subplot(2, 1, 1)
    # plt.plot(t, x)
    # plt.xlabel('Time (s)')
    # plt.ylabel('x (m)')
    # plt.title('Particle motion in Paul trap with nonlinear drag')
    # plt.grid(True)
    # plt.show()

    # FFT of the (detrended) signal
    # N = len(x)
    # X = np.fft.rfft(x - np.mean(x))
    # freqs = np.fft.rfftfreq(N, dt)
    # mag = np.abs(X) / N
    #
    # plt.plot(freqs, mag/max(mag),linewidth=4)
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Amplitude')
    # plt.title('FFT of particle motion')
    # plt.grid(True)
    #
    # q_z=2*q/m * V0/r0**2 / Omega**2
    # omega_z=q_z*Omega/np.sqrt(8)
    # x_pred=np.cos(omega_z*t) * (1 - 0.5 * q_z * np.cos(Omega*t))
    # X_pred = np.fft.rfft(x_pred - np.mean(x_pred))
    # mag_pred = np.abs(X_pred) / N
    # plt.plot(freqs, mag_pred / max(mag_pred),alpha=0.5)
    #
    # plt.tight_layout()
    # plt.show()
    return (t,x)

def plot_position(t, x, T_cutoff=0.7):
    plt.figure(figsize=(12, 4))
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(t, x)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('x (m)')
    ax1.set_title('Particle motion in Paul trap')
    ax1.grid(True)

    mask = t > T * T_cutoff
    t = t[mask]
    x = x[mask]
    ax2 = plt.subplot(1, 2, 2)
    N = len(x)
    X = np.fft.rfft(x - np.mean(x))
    freqs = np.fft.rfftfreq(N, dt)
    mag = np.abs(X) / N
    ax2.plot(freqs, mag)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('FFT of particle motion')
    ax2.grid(True)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()


def test_FFT_time_convergence():
    (t,x)=trap_solver(m,q,V0,Omega,r0,g,gamma1,gamma2,dt,T,0.0,0.0)
    plot_position(t, x, T_cutoff=0.9)
    plot_position(t, x, T_cutoff=0.7)
    plot_position(t, x, T_cutoff=0.5)




def test_V0_dependence():
    V0_list=[500,700,1000,1500,2000,2200,2600,3000,4000,4400]
    V0_list=np.array(V0_list)
    x_std_list=[0]*len(V0_list)
    x_avg=[0]*len(V0_list)
    for i,V0_var in enumerate(V0_list):
        (t, x)=trap_solver(m,q,V0_var,Omega,r0,g,gamma1,gamma2,dt,T,0.0,0.0)
        plot_position(t, x)
        x_std_list[i]=np.std(x)
        x_avg[i]=np.mean(x)

    plt.loglog(V0_list,x_std_list,'o-')
    plt.loglog(V0_list,1/V0_list)
    plt.legend(['Simulation','1/V0'])
    plt.xlabel('AC Voltage Amplitude V0')
    plt.ylabel('Position Standard Deviation')
    plt.show()


def test_gamma_dependence():
    gamma_list=gamma1*np.array([0.02,0.05,0.1,0.15,0.3,0.8,1.0,1.5,2.0,3.0,4.0,5.0,6.0,10.0])
    gamma_list=np.array(gamma_list)
    x_std_list=[0]*len(gamma_list)
    x_avg_list=[0]*len(gamma_list)
    for i,gamma_var in enumerate(gamma_list):
        (t, x)=trap_solver(m,q,V0,Omega,r0,g,gamma_var,gamma2,dt,T,0.0,0.0)
        plot_position(t, x)
        x_std_list[i]=np.std(x)
        x_avg_list[i]=np.mean(x)

    plt.loglog(gamma_list,x_std_list,'o-')
#    plt.loglog(gamma_list,gamma_list)

    plt.legend(['Simulation','1/V0'])
    plt.xlabel('Gamma')
    plt.ylabel('Position Standard Deviation')
    plt.tight_layout()
    plt.show()

    plt.plot(gamma_list,x_avg_list,'o-')
    plt.xlabel('Gamma')
    plt.ylabel('Position Average')
    plt.tight_layout()
    plt.show()

#test_gamma_dependence()

# test_V0_dependence()


# Amplitude is linear in g. Obvious from the EOM: g --> 2g means z(t) --> 2z(t) (for the particular solution).
# g_list=g*np.array([0.1,0.3,0.7,0.9,1.0,1.2,1.4,1.7,2.0])
# g_list=np.array(g_list)
# x_std_list=[0]*len(g_list)
# for i,g_var in enumerate(g_list):
#     (t, x)=trap_solver(m,q,V0,Omega,r0,g_var,gamma1,gamma2,dt,T,0.0,0.0)
#     plot_position(t, x)
#     x_std_list[i]=np.std(x)




# plt.plot(V0_list,x_std_list,'o-')
# plt.loglog(g_list,x_std_list,'o-')
# plt.loglog(g_list,g_list)
# plt.xlabel('g')
# plt.ylabel('Position Standard Deviation')
# plt.show()

if __name__ == "__main__":
    q = 1.5e-6  # particle charge
    m = 1.5e-3  # particle mass
    V0 = 3000.0  # amplitude of AC voltage
    Omega = 2 * np.pi * 50  # AC drive frequency (rad/s)
    r0 = 1e-2  # trap size scale
    g = 9.81  # gravitational acceleration

    gamma1 = 0.1  # 2#0.1  # drag coefficient
    gamma2 = 0.0  # 0.1  # nonlinear drag coefficient

    dt = 0.001  # time step (s)
    T = 10.0  # total simulation time (s)

    (t, x) = trap_solver(m, q, V0, Omega, r0, g, gamma1, gamma2, dt, T, 0.0, 0.0)
    plot_position(t, x)
