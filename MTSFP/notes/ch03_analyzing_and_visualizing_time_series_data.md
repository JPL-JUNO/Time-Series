# Analyzing and Visualizing Time Series Data

## Components of a time series

任何时间序列都可以包含以下部分或全部组成部分：

- Trend
- Seasonal
- Cyclical
- Irregular

这些成分可以以不同的方式混合，但两种非常常见的假设方式是加法（$Y = Trend + Seasonal + Cyclical + Irregular$）和乘法（$Y = Trend *Seasonal* Cyclical *Irregular$）。

### 趋势成分

趋势是时间序列均值的长期变化。它是时间序列在特定方向上平滑稳定的运动。当时间序列向上移动时，我们说存在上升或增加趋势，而当时间序列向下移动时，我们说存在下降或减少趋势。
