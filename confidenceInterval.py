import math
from NaiveBayes import error_rate as errorRateNB
from DecisionTree import error_rate as errorRateDT

sampleSize = 15060

errorDifference = abs(errorRateDT - errorRateNB)
variance = ((errorRateNB * (1-errorRateNB)) / sampleSize) + ((errorRateDT * (1-errorRateDT)) / sampleSize)

confidenceLevel = 0.95
zscore = -1.96

difference = zscore * math.sqrt(variance)

print("Interval = " + str(round(errorDifference,3)) + " \u00B1 " + str(round(difference,3)))

