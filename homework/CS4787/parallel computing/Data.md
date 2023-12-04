### Part 1
![[Pasted image 20231203235416.png]]
![[Pasted image 20231204000306.png]]
![[Pasted image 20231204002340.png]]

### Part 2

2.1. 
I have 8 cores. On Mac, opening "System Information" reads,
  Total Number of Cores: 8 (4 performance and 4 efficiency)
I used 4 threads when implicitly multithreading.
![[Pasted image 20231204002815.png]]

### Part 3
![[Pasted image 20231204003825.png]]

### Part 4



![[Pasted image 20231204004842.png]]
![[Pasted image 20231204005340.png]]


The float32 decreases in time much faster and doesn't suffer as much from small batch sizes.
However, at a certain point, they all become effectively the same speed, since the benefits of higher batch sizes supercedes any alternative

can fit more data, and data transmits faster. 

Probably because modern hardware is usually specialized to perform 64bit computations faster.

And especially when parallelization, the effects of this hardware specialization is even more prominent, 

32 bit is cheaper and lower power, but modern cpus are specialized to perform 64 bit computations and can therefore compute them faster.

For implicit threading, the difference is small and almost imperceptible, with 64 bit being slightly faster in the final 3000 size batch.

Modern CPUs trend toward 64 bit computation.

More modern CPUs like my computer (designed in 2021)


