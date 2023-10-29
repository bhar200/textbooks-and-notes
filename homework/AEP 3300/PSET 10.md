#physics 
# Q1
### Q1a
We begin,
$$\mathcal L = \frac{1}{2}\mu (\dot r ^2 +r^2\dot \phi^2) - k r^n$$
By Euler-Lagrange,
$$0 = \frac{d L }{d\phi}-\frac{d}{dt}\frac{dL}{d\dot \phi} = 
\boxed{-\mu r^2 \ddot \phi = 0}
$$
$$0 = \frac{d L }{dr}-\frac{d}{dt}\frac{dL}{d\dot r} = \mu r \dot \phi^2-knr^{n-1}-\mu \ddot r \implies \boxed{\mu \ddot r = \mu r \dot \phi^2-knr^{n-1}}$$
Combining, 
$$\dot \phi = L \implies \boxed{\mu \ddot r = \mu r L^2-knr^{n-1}}$$
### Q1b
If $\dot r = 0$, then
$$\mu \ddot r = \mu r L^2-knr^{n-1}=0 \implies 

\boxed{
r_{eq} = 
\bigg(\frac{\mu L^2}{kn}\bigg)^{1/(n-2)}
}$$
### Q1c
We have
$$\mu \ddot r = \mu r L^2-knr^{n-1},\ \ \ kn>0$$
To be stable, this plot for $\ddot r$ must cross zero from positive to negative. The derivative at equilibrium, then, is
$$\mu L^{2}-knr^{n-2}\left(n-1\right)=
\mu L^{2}-kn(\frac{\mu L^2}{kn})\left(n-1\right)
=(2-n)\mu L^2
$$
So to have a stable equilibrium, we require $2-n \leq 0 \implies 2\leq n$.