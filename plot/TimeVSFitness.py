import matplotlib.pyplot as plt
import math

# line 1 points
x1 = [0,7.533244,14.825801,22.095287,29.202148,36.365345]
y1 = [8.01193055569637e-05,4.5995974206957e-06,6.12445785036437e-08,2.82198322874722e-11,6.76810157852423e-15,1.03410445299729e-15]
# plotting the line 1 points
plt.plot(x1, [math.log10(i) for i in y1], label = "GMRES", marker='x')

# line 2 points
x2 = [0,0.954248,1.212765,1.459785,1.702903,1.956918,2.199977,2.437768,2.678581,2.931413,3.173243,3.421287,3.699985]
y2 = [8.0119305556963734e-05,4.60022370570733e-06,4.71205963304785e-07,1.39229863355653e-08,7.76557402860303e-10, 7.91928687438543e-11,4.20097679757974e-12,2.48316163959499e-13,1.57681970741051e-14,2.26003556408881e-15,1.10690016724612e-15,1.03608910278579e-15,1.03033477548411e-15]
# plotting the line 2 points
plt.plot(x2, [math.log10(i) for i in y2], label = "BFGS", marker= 'o')

# naming the x axis
plt.xlabel('Time (s)')
# naming the y axis
plt.ylabel('log(Fitness)')
# giving a title to my graph
plt.title('Fitness vs Time Plot')

# show a legend on the plot
plt.legend()

# function to show the plot
plt.show()

