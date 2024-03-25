require(lubridate)

eventTime <- as.POSIXct("2018-02-06 15:41:23.102")

eventTime1 <- as.POSIXct("2018-02-06 15:41:23.102", tz = "EST")
eventTime2 <- as.POSIXct("2018-02-06 15:41:23.102", tz = "EE")
eventTime3 <- as.POSIXct("2018-02-06 15:41:23.102")

attr(eventTime1, "tzone")
attr(eventTime2, "tzone")
attr(eventTime3, "tzone")

format(eventTime, tz = "GMT", usetz = TRUE)
format(eventTime, tz = "America/Los_Angeles", usetz = TRUE)

eventTime1Converted <- as.POSIXct(format(eventTime1, tz = "GMT", usetz = TRUE), tz = "GMT")
eventTime1Copy <- eventTime1
attributes(eventTime1Copy)$tzone <- "GMT"

eventTime1 - eventTime
eventTime1 - with_tz(eventTime1, "GMT")

t1 <- as.POSIXct("2018-02-06 15:41:23.102", tz = "EST")
t2 <- as.POSIXct("2018-02-06 15:41:23.102", tz = "GMT")

difftime(t1, t2, units = "hours")
difftime(t1, t2, units = "days")

Sys.setenv(TZ = "GMT")

with_tz(eventTime1, "GMT")
# You can also keep the 'label' on the time and simply swap the time zone
force_tz(eventTime1, "GMT")
