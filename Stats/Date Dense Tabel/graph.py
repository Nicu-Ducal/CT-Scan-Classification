train_accuracy, train_losses, validation_accuracy, validation_losses = [], [], [], []


def parse_data(lst):
    global train_accuracy, train_losses, validation_accuracy, validation_losses
    hopa1 = False
    for i, row in enumerate(lst):
        if 'Epoch' in row:
            hopa1 = True
            train_accuracy.append(float(row[7]))
            train_losses.append(float(row[4][:-1]))
        elif hopa1 and 'Loss' in row:
            # print(row)
            hopa1 = False
            validation_accuracy.append(float(row[5]))
            validation_losses.append(float(row[2][:-1]))


if __name__ == "__main__":
    with open("LR4 ADAM.txt") as file:
        lst = [row.rstrip().split() for row in file.readlines()]

    parse_data(lst)
    print(validation_losses)
