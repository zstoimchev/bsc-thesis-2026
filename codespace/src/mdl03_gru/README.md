# Model 03: GRU

This model is currently optional and disabled.

A GRU model should only be used if the dataset is prepared as meaningful temporal sequences. The current dataset is
tabular flow-based data, where each row is treated independently.

## Status

Not implemented as of 6.7.2026.

## Reason

Applying a GRU directly to independent tabular rows would not be methodologically correct. A proper GRU experiment would
require sequence construction, such as grouping flows by time, host, or connection context.
