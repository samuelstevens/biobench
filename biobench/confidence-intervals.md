# Confidence Intervals

Recommended Reading:

* [Sebastian Raschka's blogpost on confidence intervals for ML](https://sebastianraschka.com/blog/2022/confidence-intervals-for-ml.html)

## BioBench Confidence Intervals

We resample from the test predictions $N$ times.
Then we re-calculate the statistic of interest (typically just mean accuracy) for each resampling.
Over $N$ re-samples, we collect a distribution of possible mean accuracies.
Then, if we repeated this process 100 times, we could assume that in 95 of the cases, the true mean accuracy is in the sampled confidence interval.

> For me personally, this is pretty convoluted logic.
> I just assume that my true mean is 95% likely to be in my confidence interval.[^wrong]

In practice, we use $N = 500$.

[^wrong]: I know this is wrong. I know. You don't need to tell me. I know it's wrong. But it's way freakin' easier to reason about and it's only subtley wrong and it's still better than just a mean. I know it's wrong. I know.

## Note on Confidence Interval Types

For different tasks, we *could* use different kinds of bootstrapping.
Non-parametric methods like nearest centroid classifiers or KNN are particularly easy to re-evaluate and so we could bootstrap the entire process.
Parametric methods like linear classifiers or SVMs are slow to train, so we don't want to repeat the training step 200+ times.

In practice, however, we simply bootstrap test set predictions because it can be cheaply applied to all benchmarks regardless of training or inference cost.
It requires only that we have a $N$-dimensional vector of scores, where $N$ is the number of test examples, which is always under 10M, so at most a 40MB vector.
