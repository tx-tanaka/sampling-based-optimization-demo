# Sampling from Boltzmann distribution

Numerical demonstration of the algorithm below.

![This is an alt text.](/image/algorithm.png "Monte Carlo sampling.")

For demonstration purposes, we have set $n=2$, $\alpha = 1,000$, and $N=100,000,000$. For each two-dimensional vector $x=(x_1, x_2)$, 
the cost function $C$ is defined by $C(x)=4(20x_1-10)^4 + 4(20x_1-10)^3(20x_2-10) - 7(20x_1-10)^2(20x_2-10)^2 - 2(20x_1-10)(20x_2-10)^3 + 10(20x_2-10)^4$.

![This is an alt text.](/image/Figure_1.png "Result.")
