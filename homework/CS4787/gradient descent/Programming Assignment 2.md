Bryant Har
# To Address
### Not converging faster for shuffle?


### Q1
Gradient derivation:
$$R(h_W) = \frac{1}{\mathcal{D}}\sum L(h_W(x), y)+\frac{\gamma}{2}||W||^2$$
$$\nabla_W R(h_W) = \frac{1}{\mathcal{D}}\sum \nabla_WL(h_W(x), y)+\gamma W$$
$$\nabla_W R(h_W) = \frac{1}{\mathcal{D}}\sum_{\mathcal{D}} (\mathrm{softmax}(Wx_i)-y_i)x_i^T+\gamma W$$
Gradient step derivation:
$$w_{n+1} = w_n - \alpha \nabla R(h_W)$$
$$w_{n+1} = w_n - \alpha \bigg[ 
\frac{1}{\mathcal{D}}\sum_{\mathcal{D}} (\mathrm{softmax}(Wx_i)-y_i)x_i^T+\gamma W
\bigg]$$
### Q2
Below, observe it took 21.781 seconds for 10 iterations. Multiplying by 100, this would take 2,178 seconds or around half an hour for 1000 iterations.

![[Screen Shot 2023-09-29 at 12.27.15 AM.png |500]]


It is significantly faster with numpy optimizations (30x). Due to using C structures and vectorization abstracted away by numpy libraries.
![[Pasted image 20230929011850.png|500]]

### Q5
We clearly see that there is a tradeoff between batch size, step size, and speed. Batch size = 1 usually took triple the time as B=60, but because there were more total updates to the weights and a finer step size, it could converge to a better value.

We see the GD error and loss graphs go down with more training iterations, as accuracy increases. The error graphs are very slightly less steep due to the L2 regularization term penalizing complexity.

![[Pasted image 20231002113528.png|500]]
![[Pasted image 20231002114317.png|500]]
![[Pasted image 20231002114409.png|500]]
![[Pasted image 20231002114432.png|500]]
![[Pasted image 20231002115054.png|500]]
![[Pasted image 20231002115125.png|500]]
![[Pasted image 20231002115213.png|500]]



### Q7
1000 Iterations,
![[Pasted image 20230929162402.png|500]]
![[Pasted image 20231002114743.png|500]]
![[Pasted image 20231002114800.png|500]]
![[Pasted image 20231002114816.png|500]]
![[Pasted image 20230929162500.png|500]]
![[Pasted image 20230929162507.png|500]]
![[Pasted image 20230929162516.png|500]]
