import numpy as np
import matplotlib.pyplot as plt

total_time = int(10001)
all_speed = [6, 8, 10, 12, 14, 16, 18]
file_name = "C:/Users/29904/Desktop/car_following/results_for_Huang_LSTM_0310/"
for equi_speed in all_speed:
    speed = np.loadtxt(file_name + "Huang_LSTM_speed_simulation_" + str(equi_speed) + ".csv", delimiter=',')
    position = np.loadtxt(file_name + "Huang_LSTM_position_simulation_" + str(equi_speed) + ".csv", delimiter=',')
    acce = np.loadtxt(file_name + "Huang_LSTM_acce_simulation_" + str(equi_speed) + ".csv", delimiter=',')

    # # Plot speed deviation, acce and spacing
    # fig, axs = plt.subplots(1, 3, sharex=True)
    # for ind, acc in enumerate(acce.T):
    #     axs[0].plot(range(total_time), acc, label="veh " + str(ind), lw=1)
    # axs[0].plot(range(total_time), [0] * total_time, label="equilibrium acce", ls='--', lw=1)
    # for ind, veh in enumerate(speed.T):
    #     axs[1].plot(range(total_time), veh, label="veh " + str(ind), lw=1)
    # axs[1].plot(range(total_time), [equi_speed] * total_time, label="equilibrium speed", ls='--', lw=1)
    # all_spacing = np.abs(np.diff(position))
    # for ind, veh in enumerate(all_spacing.T):
    #     axs[2].plot(range(total_time), veh, label="veh " + str(ind), lw=1)
    # axs[2].plot(range(total_time), [equi_gap] * total_time, label="equilibrium gap", ls='--', lw=1)
    # # plt.legend()

    # Plot max speed deviation
    all_speed_deviation = []
    valid_speed = speed[4900:-1, :]
    for veh_speed in valid_speed.T:
        max_deviation = max(np.abs(veh_speed - equi_speed))
        all_speed_deviation.append(max_deviation)
    np.savetxt(file_name + "Huang_LSTM_" + str(equi_speed) + "_speed_deviation_simulation.csv",
               np.array(all_speed_deviation).T,
               delimiter=',')
    plt.figure()
    plt.plot(range(101), all_speed_deviation)
    plt.xlabel("platoon")
    plt.ylabel("max speed deviation (m/s)")
    plt.title("Huang_LSTM_" + str(equi_speed) + "_speed_deviation_simulation")
    plt.show()
