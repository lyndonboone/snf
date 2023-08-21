library(SNFtool)

digits0 <- read.csv("digits0.csv", header=FALSE)
digits1 <- read.csv("digits1.csv", header=FALSE)
digits2 <- read.csv("digits2.csv", header=FALSE)
digits3 <- read.csv("digits3.csv", header=FALSE)

digits0 = standardNormalization(digits0)
digits1 = standardNormalization(digits1)
digits2 = standardNormalization(digits2)
digits3 = standardNormalization(digits3)

K = 20  # change
alpha = 0.8  # change

Dist_digits0 = dist2(as.matrix(digits0), as.matrix(digits0))
Dist_digits1 = dist2(as.matrix(digits1), as.matrix(digits1))
Dist_digits2 = dist2(as.matrix(digits2), as.matrix(digits2))
Dist_digits3 = dist2(as.matrix(digits3), as.matrix(digits3))

AM_digits0 = affinityMatrix(Dist_digits0, K, alpha)
write.csv(AM_digits0, "0.csv", row.names = FALSE)
AM_digits1 = affinityMatrix(Dist_digits1, K, alpha)
write.csv(AM_digits1, "1.csv", row.names = FALSE)
AM_digits2 = affinityMatrix(Dist_digits2, K, alpha)
write.csv(AM_digits2, "2.csv", row.names = FALSE)
AM_digits3 = affinityMatrix(Dist_digits3, K, alpha)
write.csv(AM_digits3, "3.csv", row.names = FALSE)