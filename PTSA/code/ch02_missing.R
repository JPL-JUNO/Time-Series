require(zoo) # zoo provides time series functionality
require(data.table) # data.table is a high performance data frame

unemp <- fread("data/UNRATE.csv")
unemp[, DATE := as.Date(DATE)]
setkey(unemp, DATE)

# generate a data set where data is randomly missing
rand.unemp.idx <- sample(1:nrow(unemp), .1 * nrow(unemp))
rand.unemp <- unemp[-rand.unemp.idx]

# generate a data set where data is more likely to be missing if it's high
high.unemp.idx <- which(unemp$UNRATE > 8)
num.to.select <- .5 * length(high.unemp.idx)
high.unemp.idx <- sample(high.unemp.idx, num.to.select)
bias.unemp <- unemp[-high.unemp.idx]


all.dates <- seq(from = unemp$DATE[1], to = tail(unemp$DATE, 1), by = "months")
rand.unemp <- rand.unemp[J(all.dates), roll = 0]
bias.unemp <- bias.unemp[J(all.dates), roll = 0]
rand.unemp[, rpt := is.na(UNRATE)]
# here we label the missing data for easy plotting


rand.unemp[, impute.ff := zoo::na.locf(UNRATE, na.rm = FALSE)]
bias.unemp[, impute.ff := zoo::na.locf(UNRATE, na.rm = FALSE)]

unemp[350:400, plot(DATE, UNRATE, col = 1, lwd = 2, type = "b")]
rand.unemp[350:400, lines(DATE, impute.ff, col = 2, lwd = 2, lty = 2)]
rand.unemp[350:400][
  rpt == TRUE,
  points(DATE, impute.ff, col = 2, pch = 6, cex = 2)
]

png("img/forward_fill_plot.png")
unemp[350:400, plot(DATE, UNRATE, col = 1, type = "b")]
rand.unemp[350:400, lines(DATE, impute.ff, col = 2)]
rand.unemp[350:400][rpt == TRUE, points(DATE, impute.ff, col = 2, lwd = 3)]
dev.off()

rand.unemp[, impute.rm.nolookahead := rollapply(
  c(NA, NA, UNRATE), 3,
  function(x) {
    if (!is.na(x[3])) x[3] else mean(x, na.rm = TRUE)
  }
)]
bias.unemp[, impute.rm.nolookahead := rollapply(
  c(NA, NA, UNRATE), 3,
  function(x) {
    if (!is.na(x[3])) x[3] else mean(x, na.rm = TRUE)
  }
)]

rand.unemp[, complete.rm := rollapply(
  c(NA, UNRATE, NA), 3,
  function(x) {
    if (!is.na(x[2])) {
      x[2]
    } else {
      mean(x, na.rm = TRUE)
    }
  }
)]

png("img/moving_average_plot.png")
use.idx <- 150:200
unemp[use.idx, plot(DATE, UNRATE), col = 1, type = "b"]
