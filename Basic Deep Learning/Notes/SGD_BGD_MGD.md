### 1. Batch Gradient Descent
- We go through all training samples and calculate sum of error (error of all samples) in each epoch
- Now we back propagate and adjust the weights
**Cons:**
- For large dataset i.e 10Million samples, to find cumulative error for first round epoch we need to do a forward pass for 10M samples!
- For just 2 features, this requires finding 20M derivatives on each epoch.

### 2. Stochastic Gradient Descent(SGD):
- Instead of picking all training samples, SGD takes randomly one sample for forward pass and then adjust its weights
- Efficient for large dataset, does less computation.
**Cons:**
- Time consuming as it takes only on sample on each epoch.
## Differences of BGD & SGD:![alt text](<src/Screenshot from 2025-05-03 06-22-43.png>)

### 3. Mini Batch Gradient Descent:
- Instead of taking one random sample, takes a batch of samples randomly for forward pass.

## Quick Summary:![alt text](<src/Screenshot from 2025-05-03 06-25-26.png>)

