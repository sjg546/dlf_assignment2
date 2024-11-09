from matplotlib import pyplot as plot
from numpy import log10

lr = [
8,16,32,64
]
accuracy = [
0.8931,
0.8936,
0.8952,
0.8959
]

precision = [
0.8928796866866147,
0.8955118993761291,
0.8958394094448912,
0.8958043638068657
]

recall = [
0.8931,
0.8936,
0.8952,
0.8959
]

f1score = [
0.8925227551440094,
0.8938361924066334,
0.89513136281232,
0.8956742811492222
]

plot.plot(lr,accuracy,label="Accuracy")
plot.plot(lr,precision,label="Precision")
plot.plot(lr,recall,label="Recall")
plot.plot(lr,f1score,label="F1 Score")
plot.xlabel("Batch Size")
plot.title("VGG16 Batch Size Investigation")
plot.legend()
plot.show()

