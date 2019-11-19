from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import datasets, svm, metrics
import csv
import numpy as np

value_true = []
value_pred = []

gt_vs_pred_file =  '/path/to/your/eval.txt'
with open(gt_vs_pred_file) as f:
	lines = f.readlines()
	for line in lines:
		#line
		gt = (line.strip().split(' ')[0])
		pred = (line.strip().split(' ')[1])
		value_true.append(gt)
		value_pred.append(pred)


#target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', #'32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54']

target_names = [str(i) for i in range(1, 29)] #max of gt will be the range of gt, obscures readability


print(classification_report(value_true, value_pred))
print("Confusion matrix:" + "\n")

values = []

# Referred - https://gist.github.com/zachguo/10296432
def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels]+[2]) # 2 is value length
    empty_cell = " " * columnwidth
    # Print header
    print "    " + empty_cell,
    for label in labels: 
        print "%{0}s".format(columnwidth) % label,
    print
    # Print rows
    for i, label1 in enumerate(labels):
        print "    %{0}s".format(columnwidth) % label1,
        for j in range(len(labels)): 
            cell = "%{0}.0f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print cell,
            if j == i:
               values.append(cell)
        print

# first generate with specified labels
labels = target_names
cm = confusion_matrix(value_true, value_pred, labels)

# then print it in a pretty way
print_cm(cm, labels)

print values


