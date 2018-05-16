

sOutput = open("outputCosine.txt", 'r').read()
sTruth =  open("video_small_ground_truth_cosine.csv", 'r').read()

outputTuples = []
for line in sOutput.split('\n'):
    if len(line) > 0:
        lineContentList = line.split(",")
        if (float(lineContentList[2]) >= 0.5):
            if (lineContentList[0] < lineContentList[1]):
                outputTuples.append((lineContentList[0], lineContentList[1]))
            else:
                outputTuples.append((lineContentList[1], lineContentList[0]))

truthTuples = []
for line in sTruth.split('\n'):
    if len(line) > 0:
        lineContentList = line.split(",")
        truthTuples.append((lineContentList[0], lineContentList[1]))
outPutSet = set(outputTuples)
truthSet = set(truthTuples)
falsePos = 0
truePos = 0
for outputTuple in outputTuples:
    if (outputTuple not in truthSet):
        falsePos += 1
    else:
        truePos += 1
falseNeg = 0
for truthTuple in truthTuples:
    if (truthTuple not in outPutSet):
        falseNeg += 1

print("precision:" + str(truePos/(truePos + falsePos)))
print("recall:" + str(truePos/(truePos + falseNeg)))
