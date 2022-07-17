import pandas as pd
import argparse
import math
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Arguments for home assignment selection')
parser.add_argument('--path_to_data', default='/Users/ramtahor/Desktop/acceleration_tabel.csv',
                    help='Path to a csv file')
parser.add_argument('--initial_altitude', default=100, help="initial altitude of the drone at time 0")


# this function gets a Dataframe and the name of the acceleration in the axis we want to calculate the velocity
def get_velocity(df: pd.DataFrame, acc_col: str) -> list:
    return [sum(list(df[acc_col])[:i]) for i in range(len(list(df[acc_col])))]


# this function gets a Dataframe and the name of the omega in the axis we want to calculate the total angel
# the correction is a number to fit the lot to the axis in the SW(python)
def get_angle(df: pd.DataFrame, omega_col: str, correction: float) -> list:
    return [sum(list(df[omega_col])[:i + 1]) + correction for i in range(len(list(df[omega_col])))]


# this function gets a velocity, azimuth and pitch returns a tuple of th velocity list in xy plane and the one in z axis
def get_velocity_vector(velocity, azimuth, pitch) -> tuple:
    u0_xy = [(velocity[i] * math.cos(math.radians(pitch[i])) * math.sin(math.radians(azimuth[i])),
              velocity[i] * math.cos(math.radians(pitch[i])) * math.cos(math.radians(azimuth[i]))) for i in
             range(len(list(azimuth)))]
    u0_z = [velocity[i] * math.sin(math.radians(pitch[i])) for i in range(len(list(azimuth)))]

    return u0_xy, u0_z


# This function calculates the position of the drone in every axis
# the calculation is done by the equation of motion in two stages:
# 1. x = u*t + 0.5 * a * t ** 2
# 2. sum the position list until a certain index every time to add the initial distance
def position(u0_xy: list, u0_z: list, ax: list, az: list, azimuth, pitch, init_alt) -> tuple:
    pos_x = [u0_xy[i][0] + 0.5 * ax[i] * math.sin(math.radians(azimuth[i - 1])) for i in range(len(azimuth))]
    pos_y = [u0_xy[i][1] + 0.5 * ax[i] * math.cos(math.radians(azimuth[i - 1])) for i in range(len(azimuth))]
    pos_x = [sum(pos_x[:i]) for i in range(len(pos_x))]
    pos_y = [sum(pos_y[:i]) for i in range(len(pos_y))]
    pos_z = [init_alt + u0_z[i] + 0.5 * az[i] * math.sin(math.radians(pitch[i - 1])) for i in range(len(pitch))]

    return pos_x, pos_y, pos_z


def main():
    args = parser.parse_args()
    df = pd.read_csv(args.path_to_data)
    init_alt = args.initial_altitude
    velo = get_velocity(df, 'ax')
    azimuth = get_angle(df, 'wz', 90)
    pitch = get_angle(df, 'wy', 0)
    u_xy, u_z = get_velocity_vector(velocity=velo, azimuth=azimuth, pitch=pitch)
    pos_x, pos_y, pos_z = position(u_xy, u_z, list(df['ax']), list(df['az']), azimuth, pitch, init_alt)
    t = list(df['t'])

    # plot the results
    # It was simpler to just switch between the axis instead transpose the plot
    plt.figure('position')
    plt.plot(pos_y, pos_x)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()

    plt.figure('Azimuth[deg]')
    plt.plot(t, [i - 90 for i in azimuth])
    plt.ylabel('Azimuth[deg]')
    plt.xlabel('Time')
    plt.grid()

    plt.figure('Velocity Vs Time')
    plt.plot(t, velo)
    plt.ylabel('Velocity')
    plt.xlabel('Time')
    plt.grid()

    plt.show()


if __name__ == '__main__':
    main()
