import csv
import numpy as np

def read_csv(csv_file):
    f_write = open('output.csv', 'w')
    f_write.write('阈值' + '\t' + '准确率' + '\t' + '召回率' + '\n')

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)

    label = []
    prob = []
    for line in reader:
        label.append(line[1])
        prob.append(line[2])

    label = np.array(label)
    prob = np.array(prob)
    thres = np.arange(0, 1, 0.001)
    
    for thre in thres:
        pred = (prob>thre)
        TP = ((pred==1) * (label==1)).sum()
        FP = ((pred==1) * (label==0)).sum()
        TN = ((pred==0) * (label==0)).sum()
        FN = ((pred==0) * (label==1)).sum()

        precision = TP/(TP+FP)
        recall = TP/(TP+FN)

        f_write.write(str('%.3f' % thre) + '\t' + str('%.4f' % precision) + '\t' + str('%.4f' % recall) + '\n')

    f_write.close()

if __name__ == '__main__':
    csv_file = ''
    read_csv(csv_file)