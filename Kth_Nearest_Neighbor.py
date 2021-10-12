# Ducal Nicolae, grupa 234
# Model: KNN
import torch
import numpy as np
import os
import cv2
import csv
import matplotlib.pyplot as plt


# Fisierele de inputs
TRAIN_FOLDER = "./train"
TEST_FOLDER = "./test"
VALIDATION_FOLDER = "./validation"
TRAIN_LABELS = "./train.txt"
TEST_LABELS = "./test.txt"            # No labels
VALIDATION_LABELS = "./validation.txt"
RESHAPE_SIZE = 2500
EXISTA_GPU = torch.cuda.is_available()


# Citirea imaginilor si datelor din fisiere
def read_images(path, files):
    if not os.path.isdir(path):
        raise Exception("Path-ul specificat nu este un folder")
    return torch.tensor([cv2.imread(f"{path}/{img}", 0).reshape(RESHAPE_SIZE) for img in files])


def read_csv_file(path):
    if not os.path.isfile(path):
        raise Exception("Path-ul specificat nu este un fisier")
    with open(path) as csv_file:
        data = np.array(list(csv.reader(csv_file)))
    return data


def split_csv_data(data):
    return data[:, 0], torch.tensor(data[:, 1].astype(int))


# Citirea fisierelor problemei
train_data = read_csv_file(TRAIN_LABELS)
train_file_names, train_labels = split_csv_data(train_data)
train_images = read_images(TRAIN_FOLDER, train_file_names)

validation_data = read_csv_file(VALIDATION_LABELS)
validation_file_names, validation_labels = split_csv_data(validation_data)
validation_images = read_images(VALIDATION_FOLDER, validation_file_names)

test_file_names = read_csv_file(TEST_LABELS)[:, 0]
test_images = read_images(TEST_FOLDER, test_file_names)

# print(train_images.shape)
# print(train_labels.shape)
# print(torch.cuda.is_available())

# Schimbam totul pe GPU (daca avem)
def tranfera_obiect_pe_gpu(data):
    if EXISTA_GPU:
        return data.cuda()
    return data


train_images = tranfera_obiect_pe_gpu(train_images)
train_labels = tranfera_obiect_pe_gpu(train_labels)
validation_images = tranfera_obiect_pe_gpu(validation_images)
validation_labels = tranfera_obiect_pe_gpu(validation_labels)
test_images = tranfera_obiect_pe_gpu(test_images)


# Clasificatorul KNN
class KnnClassifier:
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

    def classify_image(self, test_image, num_neighbors=3, metric='L1'):
        if metric == 'L1':
            distances = self.manhattan(self.train_images, test_image)
        elif metric == 'L2':
            distances = self.euclidian(self.train_images, test_image)
        else:
            k = int(metric[1:])
            distances = self.minkowski(self.train_images, test_image, k)

        sorted_indices = torch.argsort(distances)
        k_indices = sorted_indices[:num_neighbors]
        nearest_labels = self.train_labels[k_indices]
        return torch.argmax(torch.bincount(nearest_labels))

    def classify_images(self, test_images, num_neighbors=3, metric='L1'):
        #         predicted_labels = [self.classify_image(image, num_neighbors, metric) for image in test_images]
        predicted_labels = []
        for i, image in enumerate(test_images):
            predicted_labels.append(self.classify_image(image, num_neighbors, metric))
            if i % 1000 == 0:
                print(f"Done: {i}/{len(test_images)}")

        return torch.tensor(predicted_labels)

    def manhattan(self, x, y):
        return torch.sum(torch.abs(x - y), dim=1)

    def euclidian(self, x, y):
        return torch.sqrt(torch.sum(torch.pow((x - y), 2), dim=1))

    def minkowski(self, x, y, k):
        return torch.pow(torch.sum(torch.pow((x - y), k), dim=1), 1 / k)


knn = KnnClassifier(train_images, train_labels)


# Accuracy calculator
def accuracy(ground_truth_labels, predicted_labels):
    if EXISTA_GPU:
        return np.mean(np.array(ground_truth_labels.cpu() == predicted_labels))
    return np.mean(np.array(ground_truth_labels == predicted_labels))


# Pentru plotare:
def metric_to_num(x):
    return int(x[1:])

ks = [1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 101, 103, 105, 201, 203, 205, 1001]
metrics = ['L1', 'L2', 'L3', 'L4']
results = [[] for i in range(len(metrics) + 1)]


# Antrenarea modelului
with open('trainer_knn_big_test.csv', "w") as out:
    for k in ks:  # , 33, 35, 37, 39, 41, 201, 203, 205,
        for metric in metrics:  # 'L1', 'L2', 'L3', 'L4'
            print(f'Starting: k={k}, metric={metric}')
            acc = accuracy(validation_labels, knn.classify_images(validation_images, k, metric))
            results[metric_to_num(metric)].append(acc)
            print(f"Result: k={k}, metric={metric} -> {acc}")
            out.write(f"{k},{metric},{acc}\n")

# Plotare:
for metric in metrics:
    n = metric_to_num(metric)
    plt.plot(ks, results[n])

plt.xlabel('Valorile K')
plt.ylabel('Acuratetea')
plt.title('Kth - Nearest Neighbor')
plt.show()


# Matricea de confuzie
def confunsion_matrix(predicted_labels, ground_truth_labels):
    num_labels = ground_truth_labels.max() + 1
    conf_mat = np.zeros((num_labels, num_labels))

    for i in range(len(predicted_labels)):
        conf_mat[ground_truth_labels[i], predicted_labels[i]] += 1
    return conf_mat


# Voi face matricea de confuzie doar pentru cel mai bun rezultat: K=205 si L2
conf_mat = confunsion_matrix(knn.classify_images(validation_images, 205, 'L2'), validation_labels)
print(conf_mat)