## Q1



## Q2 - Nesterov Momentum
##### Part A
We begin,
$$
\begin{cases}
v_{t+1} = w_t-\alpha \nabla f(w_t)\\
w_{t+1} = v_{t+1}+\beta(v_{t+1}-v_t)
\end{cases}
$$
$$
\implies
v_{t+1} = v_{t}+\beta(v_{t}-v_{t-1})-\alpha \nabla f(v_{t}+\beta(v_{t}-v_{t-1})) 
$$
$$
\nabla f(w_t) = \gamma w_t\implies v_{t+1} = v_{t}+\beta(v_{t}-v_{t-1})-\alpha \gamma (v_{t}+\beta(v_{t}-v_{t-1}))
$$
$$
v_{t+1} = v_{t}+\beta v_{t}- \beta v_{t-1}-\alpha \gamma v_{t}-\alpha \gamma \beta v_{t}+\alpha \gamma \beta v_{t-1}
$$
$$
v_{t+1} = (1+\beta-\alpha \gamma-\alpha \gamma \beta)v_{t}+(\alpha \gamma \beta-\beta) v_{t-1}
$$
In matrix form, this last equation is 
$$
\begin{bmatrix}
v_{t+1}\\v_{t}
\end{bmatrix}
=
\begin{bmatrix}
(1+\beta-\alpha \gamma-\alpha \gamma \beta) & \alpha \gamma \beta-\beta
\\
1 & 0
\end{bmatrix}

\begin{bmatrix}
v_{t}\\v_{t-1}
\end{bmatrix}
$$
$$
\begin{bmatrix}
v_{t+1}\\v_{t}
\end{bmatrix}
=
\begin{bmatrix}
(1+\beta)(1-\alpha \gamma) & -\beta(1-\alpha \gamma)
\\
1 & 0
\end{bmatrix}

\begin{bmatrix}
v_{t}\\v_{t-1}
\end{bmatrix}
$$
We can apply this definition repeatedly to the rightmost vector:
$$
\begin{bmatrix}
v_{t+1}\\v_{t}
\end{bmatrix}
=
\begin{bmatrix}
(1+\beta)(1-\alpha \gamma) & -\beta(1-\alpha \gamma)
\\
1 & 0
\end{bmatrix}
\begin{bmatrix}
(1+\beta)(1-\alpha \gamma) & -\beta(1-\alpha \gamma)
\\
1 & 0
\end{bmatrix}

\begin{bmatrix}
v_{t-1}\\v_{t-2}
\end{bmatrix}
$$
Unraveling this recurrence until we get to $v_1, v_0$,
$$\begin{bmatrix}
v_{t+1}\\v_{t}
\end{bmatrix}
=
\begin{bmatrix}
(1+\beta)(1-\alpha \gamma) & -\beta(1-\alpha \gamma)
\\
1 & 0
\end{bmatrix}
(\dots)

\begin{bmatrix}
v_{1}\\v_{0}
\end{bmatrix}
$$
As desired, we arrive at
$$
\boxed{\begin{bmatrix}
v_{t+1}\\v_{t}
\end{bmatrix}
=
\begin{bmatrix}
(1+\beta)(1-\alpha \gamma) & -\beta(1-\alpha \gamma)
\\
1 & 0
\end{bmatrix}
^t
\begin{bmatrix}
v_{1}\\v_{0}
\end{bmatrix}}
$$
##### Part B
We begin,
$$
\begin{vmatrix}
(1+\beta)(1-\alpha \gamma)-\lambda & -\beta(1-\alpha \gamma)
\\
1 & -\lambda
\end{vmatrix}=\lambda^2-\lambda(1+\beta)(1-\alpha \gamma) +\beta(1-\alpha \gamma)
$$
$$\lambda = \frac{(1+\beta)(1-\alpha \gamma)\pm\sqrt{(1+\beta)^2(1-\alpha \gamma)^2-4\beta(1-\alpha \gamma)}}{2}$$
$$\lambda = \frac{(1+\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1})
(1-\frac{\gamma}{L})\pm\sqrt{(1+\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1})^2(1-\frac{\gamma}{L})^2-
4\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}(1-\frac{\gamma}{L})}}{2}$$
$$\lambda = \frac{
(1-\frac{\gamma}{L})\sqrt{\kappa}\pm\sqrt{\kappa(1-\frac{\gamma}{L})^2-
(\kappa-1)(1-\frac{\gamma}{L})}}
{\sqrt{\kappa}+1}$$
$$\lambda = \frac{
(1-\frac{\gamma}{L})\sqrt{\kappa}\pm\sqrt{\frac{L}{\mu}(1-\frac{\gamma}{L})^2-
(\frac{L}{\mu}-1)(1-\frac{\gamma}{L})}}
{\sqrt{\kappa}+1}$$
$$\lambda = \frac{
(1-\frac{\gamma}{L})\sqrt{\kappa}\pm\sqrt{
(\frac{L}{\mu}-\frac{\gamma}{\mu})(1-\frac{\gamma}{L})-
(\frac{L}{\mu}-1)(1-\frac{\gamma}{L})}}
{\sqrt{\kappa}+1}$$
As desired, we arrive at
$$\boxed{\lambda = \frac{
(1-\frac{\gamma}{L})\sqrt{\kappa}\pm\sqrt{
(1-\frac{\gamma}{L})(1-\frac{\gamma}{\mu})}}
{\sqrt{\kappa}+1}}$$
##### Part C
The determinant of a matrix is the product of its eigenvalues, and $\lambda_1 \lambda_2=|\lambda|^2$ for complex eigenvalues (which $\mu \leq \gamma \leq L$ ensures). We find the determinant as before,
$$|\lambda|^2=\begin{vmatrix}
(1+\beta)(1-\alpha \gamma) & -\beta(1-\alpha \gamma)
\\
1 & 0
\end{vmatrix}=(1-\alpha \gamma)\beta$$
As desired, we arrive at
$$\boxed{|\lambda|^2=(1-\frac{\gamma}{L})\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}}$$
Further, to maximize $\lambda$ within this range, we clearly must minimize $1-\gamma/L$. 
Therefore, in the upper bound, we require $\gamma=\mu$.
$$\implies |\lambda|^2\leq(1-\frac{\mu}{L})\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}=(1-\frac{1}{\kappa})\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}=
\frac{\kappa-1}{\kappa}\frac{\kappa-1}{(\sqrt{\kappa}+1)^2}$$
$$\implies |\lambda|\leq\frac{\kappa-1}{\sqrt{\kappa}(\sqrt\kappa+1)}=\frac{\sqrt\kappa-1}{\sqrt{\kappa}}=1-\frac{1}{\sqrt{\kappa}}$$
As desired, we arrive at an upper bound,
$$\implies \boxed{\lambda \leq 1-\frac{1}{\sqrt{\kappa}}}$$
##### Part D
In effect, we combine the results from parts a-c. We take the gradient, 
$$f = \frac{1}{2}w^TAw \implies \nabla f = Aw$$
Using our scalar result from a, we replace $\gamma$ with $A$ and ones with the identity matrix accordingly,
$$\begin{bmatrix}
v_{t+1}\\ v_{t}
\end{bmatrix}
=
\begin{bmatrix}
(1+\beta)(I-\alpha A) & -\beta(I-\alpha A)
\\
1 & 0
\end{bmatrix}
^t
\begin{bmatrix}
v_{1}\\ v_{0}
\end{bmatrix}$$
Note, however, that $A$'s singular values are bounded above and below by $\mu$ and $L$. Therefore, the operator norm of $A$ is bounded the same, and $A$ does not scale any vector by more than $\mu$ or $L$. Therefore, we can use our result from part C to similarly surmise,
$$\lambda \leq 1-\frac{1}{\sqrt{\kappa}}$$
As desired,
$$\boxed{\begin{bmatrix}
v_{t+1}\\ v_{t}
\end{bmatrix}
=
\begin{bmatrix}
(1+\beta)(I-\alpha A) & -\beta(I-\alpha A)
\\
1 & 0
\end{bmatrix}
^t
\begin{bmatrix}
v_{1}\\ v_{0}
\end{bmatrix}}$$
$$\boxed{\lambda \leq 1-\frac{1}{\sqrt{\kappa}}}$$

## Q3 - Dimension Reduction
##### Part A
We compute k-nearest neighbors with k=1 as follows with mostly valid python:
```python
def knn(x): #k = 1
	nearest = float('inf')
	label = 0
	# iterate over (xi, yi) dataset
	for (xi, yi) in training: 
		#compute distance
		dist = sum((x[j]-xi[j])**2 for j in range(d))
		if dist < nearest:
			nearest = dist
			label = yi
	return label
```
##### Part B
Subtracting elementwise and getting the norm costs $d+d+d-1$. Adding the comparison check after, this costs $2d$. This must be done for every training dataset example, or $3dn$. Finally, this routine must be run for every test example, or $3dnm$.
$$\boxed{3dmn \text{ computations}=\mathcal{O}(dmn)}$$
##### Part C
We repeat in mostly valid python but compute distances more efficiently:
```python
def knn(x): #k = 1
	nearest = float('inf')
	label = 0
	# iterate over (xi, yi) dataset
	for (xi, yi) in training: 
		#get index and value arrays
		idxs = xi['index']
		vals = xi['vals']
		#compute dist heuristic in O(nonzero elements)
		dist = sum((x[idx]-val)**2 for (idx, val) in zip(idxs, vals))
		if dist < nearest:
			nearest = dist
			label = yi
	return label
```
##### Part D
The code is identical to before, except computing distances only requires iterating over the number of nonzero vector entries instead of all dimensions. Then, ignoring python quirks and zip() cost, we can replace $d$ in our previous computation with $pd$, where $0<p<1$.
$$\boxed{3(pd)mn \text{ computations}=\mathcal{O}(pdmn)}$$
If $p\approx 1$, using sparse data formats has no benefit, but for very sparse $p<<1$, this is far fewer computations.