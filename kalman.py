#-------------------------------------------------------------------------------
# FILE NAME:      kalman.py
# DESCRIPTION:    uses python3 to perform KF iterations
# USAGE:          python3 kalman.py
#                 
# notes:          Converted Matlab to Python
#                 
#
# MODIFICATION HISTORY
# Author               Date           version
#-------------------  ------------    ---------------------------------------
# Annette McDonough   2022-02-16      1.0 first version 
# Annette McDonough   2022-02-18      1.1 still converting
# Annette McDonough   2022-02-19      1.2 initialization works
# Annette McDonough   2022-02-22      1.3 working on bugs
# Annette McDonough   2022-02-24      1.4 implementing KF
# Annette McDonough   2022-02-25      1.5 changing arrays to a class
#                                         to mimmic a struct
# Annette McDonough   2022-02-27      1.6 adding second graph
# Annette McDonough   2022-03-01      1.7 cleaning up code
#-----------------------------------------------------------------------------


import numpy as np
import csv
import pandas as pd 
import math
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt 


# load data
data = pd.read_csv("EKF_DATA_circle.txt")

time = data["%time"]

Odom_x = data["field.O_x"] 

Odom_y = data["field.O_y"] 

Odom_theta = data["field.O_t"]

IMU_heading = data["field.I_t"]

IMU_Co_heading = data["field.Co_I_t"]

#IMU_Co_heading = data["field.Co_I_t"] + .01
#IMU_Co_heading[200:800] = IMU_Co_heading[200:800] + .01
#IMU_Co_heading[2500:3000] = IMU_Co_heading[2500:3000] + .01

Gps_x = data["field.G_x"]

Gps_y = data["field.G_y"]

Gps_Co_x = data["field.Co_gps_x"]
#Gps_Co_x = data["field.Co_gps_x"].add(.01)
#Gps_Co_x[200:1000] = Gps_Co_x[200:1000] + .01
#Gps_Co_x[2000:3000] = Gps_Co_x[2000:3000] + .01

Gps_Co_y = data["field.Co_gps_y"]
#Gps_Co_y = data["field.Co_gps_y"].add(.01)
#Gps_Co_y[200:1000] = Gps_Co_y[200:1000] + .01
#Gps_Co_y[2000:3000] = Gps_Co_y[2000:3000] + .01

V = 0.44
L = 1

Omega = V*np.tan(Odom_theta[0])/L
delta_t = 0.001
total = range(0, len(Odom_x))
total2 = len(Odom_x)

#calibrate IMU
IMU_heading = IMU_heading + (.32981 -.237156)

class robot_data:

    x = np.array([[Odom_x[0], Odom_y[0], V, Odom_theta[0], Omega]])

    Q = np.array([[.00001, 0, 0, 0, 0],
                 [0, .00001, 0, 0, 0], 
                 [0, 0, .001, 0, 0], 
                 [0, 0, 0, .001, 0],
                 [0, 0, 0, 0, .001]])

    H = np.array([[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]])

    R = np.array([[.1, 0, 0, 0, 0],
                  [0, .1, 0, 0, 0],
                  [0, 0, .01, 0, 0],
                  [0, 0, 0, .01, 0],
                  [0, 0, 0, 0, .01]])

    B = np.array([[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]])

    U = np.array([[0, 0, 0, 0, 0]])

    P = np.array([[.01, 0, 0, 0, 0],
                  [0, .01, 0, 0, 0],
                  [0, 0, .01, 0, 0],
                  [0, 0, 0, .01, 0],
                  [0, 0, 0, 0, .01]])

s = []

for i in range(len(total)):
    s.append(robot_data())


noise = np.ones((total2, 2), float)

for t in range(0, total2):
    noise[t][0] = np.random.normal(.5, .1)
    noise[t][1] = np.random.normal(.5, .1)

#Gps_x = Gps_x + noise[:, 0]
#Gps_y = Gps_y + noise[:, 1]

#Gps_x[200:1000] = Gps_x[200:1000] + noise[200:1000, 0]
#Gps_x[2000:3000] = Gps_x[2000:3000] + noise[2000:3000, 0]
#Gps_y[200:1000] = Gps_y[200:1000] + noise[200:1000, 1]
#Gps_y[2000:3000] = Gps_y[2000:3000] + noise[2000:3000, 1]

x1 = []
x2 = []
x3 = []

for t in range(0, len(total)):

    s[t].A = np.array([[1, 0, delta_t*np.cos(Odom_theta[t]), 0, 0],
                       [0, 1, delta_t*np.sin(Odom_theta[t]), 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 0, 1, delta_t],
                       [0, 0, 0, 0, 1]])

    s[t].R = np.array([[Gps_Co_x[t], 0, 0, 0, 0],
                      [0, Gps_Co_y[t], 0, 0, 0],
                      [0, 0, .01, 0, 0],
                      [0, 0, 0, IMU_Co_heading[t], 0],
                      [0, 0, 0, 0, .01]])

    s[t].z = np.array([[Gps_x[t], Gps_y[t], V, IMU_heading[t], Omega]])

    # kalman filter function call will go here 
    #s[t+1] = Kalman_Filters(s[t])
    s_x_priori = s[t].A * s[max(t-1, 0)].x
    s_P_priori = s[t].A * s[max(t-1, 0)].P * np.transpose(s[t].A) + s[t].Q
    s[t].K = (s_P_priori * np.transpose(s[t].H)
           * np.linalg.inv(s[t].H * s_P_priori * np.transpose(s[t].H) + s[t].R))
    s[t].x = s_x_priori + s[t].K * (s[t].z - s[t].H * s_x_priori)
    s[t].P = s_P_priori - s[t].K * s[t].H * s_P_priori





    x1 = np.append(x1, s[t].x[0, 0])
    x2 = np.append(x2, s[t].x[1, 1])
    x3 = np.append(x3, s[t].x[3, 3])


plt.scatter(x1, x2, label="kalman output", s=.1, color="blue")
#gps output
plt.scatter(Gps_x, Gps_y, label="GPS", s=.1, color="red")
# odometry output
plt.scatter(Odom_x, Odom_y, label="odom", s=.1, color="green")


plt.title("Fusion of GPS + IMU and Odometry in Position\n")
plt.xlabel("X(m)")
plt.ylabel("Y(m)")
plt.legend()

plt.grid()
#plt.savefig("imageE5.jpg", dpi=750)
plt.show()

plt.title("Fusion of GPS + IMU and Odometry in Heading\n")
plt.plot(total, x3, label="Kalman output", color="blue")
plt.plot(total, Odom_theta, label="Odom heading", color="green")
plt.plot(total, IMU_heading, label="IMU", color="red")
plt.xlabel("Iteration")
plt.ylabel("Radian")
plt.legend()
plt.grid()
#plt.savefig("imageE6.jpg", dpi=750)
plt.show()






 