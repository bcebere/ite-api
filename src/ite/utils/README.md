# Helpers&metrics

This module contains several helpers for measuring the quality of the results. The performance metrics are adapted to Tensorflow, PyTorch or Numpy, depending on their use case.

## Performance Metrics
1. Precision in Estimation of Heterogeneous Effect (PEHE).
2. Average treatment effect(ATE).
3. Policy risk(RPol).
4. Average treatment effect on the treated(ATT).

## Wrappers
The algorithms use two wrappers for accessing the performance metrics:

1.`Metrics` class. Generates performance metrics between the generated and the expected output.

API:

 - `sqrt_PEHE` - returns the squared root PEHE measure of the two entities.
 - `ATE` - returns the ATE measure of the two entities.
 - `MSE` - returns the Mean squared error of the two entities.
 - `worst_mistakes` - Returns the indices of the top k biggest PEHE errors.
 - `print` - print all the metrics.



2.`HistoricMetrics` class. Helpers for visualizing the evolution of the performance metrics. Useful for monitoring the training.

API:

 - `add` - For a group and key and a new observed value.
 - `mean_confidence_interval` - For a group and key generated the mean and the confidence interval of the observed data.
 - `print` - print all the metrics.
 - 'plot' - plot the current metrics. The plots are split into figures around the `group` keys.


## References:
1. [Reference code](https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/68e4f7d13e4368eba655132a73ff9f278da5d3af/alg/ganite/ganite.py#lines-61).
