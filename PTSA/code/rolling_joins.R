require(data.table)

donations <- data.table(
  amt = c(99, 100, 5, 15, 11, 1200),
  dt = as.Date(c(
    "2019-2-27", "2019-3-2", "2019-6-13",
    "2019-8-1", "2019-8-31", "2019-9-15"
  ))
)


publicity <- data.table(
  identifier = c("q4q42", "4299hj", "bbg2"),
  dt = as.Date(c(
    "2019-1-1",
    "2019-4-1",
    "2019-7-1"
  ))
)

setkey(publicity, "dt")
setkey(donations, "dt")

想要找到每次捐款前的营销活动
publicity[donations, roll = TRUE]
# > publicity[donations, roll = TRUE]
#    identifier         dt  amt
# 1:      q4q42 2019-02-27   99
# 2:      q4q42 2019-03-02  100
# 3:     4299hj 2019-06-13    5
# 4:       bbg2 2019-08-01   15
# 5:       bbg2 2019-08-31   11
# 6:       bbg2 2019-09-15 1200
