### Q1
We begin,
$$y_{xx}-\frac{1}{c^2}y_{tt}=0,\ \ \ y(x,0)=y_0 \text{ for } -x_0<x<x_0$$
We define green's function as satisfying
$$g_{xx}-\frac{1}{c^2}g_{tt}=0$$
$$c^2g_{xx}=g_{tt}$$
Taking the Fourier transform wrt $x$,
$$-k^2c^2G=G_{tt}$$
This is a well known differential equation (Hooke's Law). The solution is,
$$G = G_0e^{\pm ikct}$$
As usual,  
$$G_0 = \mathcal{F}(g(x|\xi, 0)) = \mathcal{F}(\delta(x-\xi)) = \frac{1}{\sqrt{2\pi}}e^{-ik\xi}$$
$$\implies G = \frac{1}{\sqrt{2\pi}}e^{\pm ik(ct-\xi)}$$
We recover $g$ using the inverse fourier transform (can verify below are equivalent),
$$g = \mathcal{F}^{-1}\bigg(\frac{1}{\sqrt{2\pi}}e^{\pm ik(ct-\xi)}\bigg)=\delta(\xi)e^{\pm ik(ct-x)}$$
Applying initial conditions, we recover $y(x,t)$,
$$y(x,t)=\int d\xi\ y_0(\xi)g(x|\xi, t)$$
$$y(x,t)=\int_{-x_0}^{x_0} d\xi\ y_0\delta(x-\xi)e^{\pm ikct}$$





### Q31

### Q33



Q31, 33, 34, 35, and 36


![[Pasted image 20231001231146.png]]



