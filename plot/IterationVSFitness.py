import matplotlib.pyplot as plt
import math

# line 1 points
x1 = [0,1,2,3,4,5]
y1 = [8.01193055569637e-05,4.5995974206957e-06,6.12445785036437e-08,2.82198322874722e-11,6.76810157852423e-15,1.03410445299729e-15]
# plotting the line 1 points
plt.plot(x1, [math.log10(i) for i in y1], label = "GMRES", marker='x')

# line 2 points
x2 = [0,1,2,3,4,5,6,7,8,9,10,11,12]
y2 = [8.0119305556963734e-05,4.60022370570733e-06,4.71205963304785e-07,1.39229863355653e-08,7.76557402860303e-10, 7.91928687438543e-11,4.20097679757974e-12,2.48316163959499e-13,1.57681970741051e-14,2.26003556408881e-15,1.10690016724612e-15,1.03608910278579e-15,1.03033477548411e-15]
# plotting the line 2 points
plt.plot(x2, [math.log10(i) for i in y2], label = "BFGS", marker= 'o')

# naming the x axis
plt.xlabel('Iterations')
# naming the y axis
plt.ylabel('log(Fitness)')
# giving a title to my graph
plt.title('Fitness vs Iterations Plot')

# show a legend on the plot
plt.legend()

# function to show the plot
plt.show()

