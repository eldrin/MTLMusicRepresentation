"""by zachguo's github"""
def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels]+[5]) # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    line = '\n'
    line +=  "    " + empty_cell
    for label in labels:
        line += "%{0}s".format(columnwidth) % label
    line += '\n'
    # Print rows
    for i, label1 in enumerate(labels):
        line += "    %{0}s".format(columnwidth) % label1
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            line += cell
        line += '\n'
    return line
