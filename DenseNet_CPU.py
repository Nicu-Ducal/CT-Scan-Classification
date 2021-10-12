import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
from PIL import Image
import csv
from tqdm import trange, tqdm
import os
import matplotlib.pyplot as plt


# Fisierele de inputs
TRAIN_FOLDER = "./train"
TEST_FOLDER = "./test"
VALIDATION_FOLDER = "./validation"
TRAIN_LABELS = "./train.txt"
TEST_LABELS = "./test.txt"  # No labels
VALIDATION_LABELS = "./validation.txt"
RESHAPE_SIZE = 2500



# Citirea imaginilor si datelor din fisiere
# def read_images(path, files):
#     if not os.path.isdir(path):
#         raise Exception("Path-ul specificat nu este un folder")
#     return torch.tensor([cv2.imread(f"{path}/{img}", 0).reshape(RESHAPE_SIZE) for img in files])


def read_csv_file(path):
    if not os.path.isfile(path):
        raise Exception("Path-ul specificat nu este un fisier")
    with open(path) as csv_file:
        data = np.array(list(csv.reader(csv_file)))
    return data


def split_csv_data(data):
    return data[:, 0], torch.tensor(data[:, 1].astype(int))


# Transformations
train_data_transform = torchvision.transforms.Compose([
    # torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize(270),
    torchvision.transforms.CenterCrop(256),
    torchvision.transforms.RandomRotation(23),
    torchvision.transforms.RandomHorizontalFlip(0.075),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.487, 0.446, 0.415],
                                     std=[0.227, 0.225, 0.228])
    # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

validation_data_transform = torchvision.transforms.Compose([
    # torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize(270),
    torchvision.transforms.CenterCrop(256),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.487, 0.446, 0.415],
                                     std=[0.227, 0.225, 0.228])
    # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


# Custom dataset
class DatasetCTScanKaggleNicuDucal(torch.utils.data.Dataset):
    def __init__(self, date_despre_set, folder_poze, transformari=None):
        self.nume_labeluri = read_csv_file(date_despre_set)
        self.folder_poze = folder_poze
        self.transformari = transformari

    def __len__(self):
        return len(self.nume_labeluri)

    def __getitem__(self, poz):
        path_imagine = os.path.join(self.folder_poze, self.nume_labeluri[poz, 0])
        imagine = Image.open(path_imagine).convert('RGB')
        if self.nume_labeluri[poz].size == 1:
            label = torch.tensor(-1)
        else:
            label = torch.tensor(int(self.nume_labeluri[poz, 1]))
        if self.transformari:
            imagine = self.transformari(imagine)
        obiect = (imagine, label)
        return obiect



trainset = DatasetCTScanKaggleNicuDucal("train.txt", "train", train_data_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

validationset = DatasetCTScanKaggleNicuDucal("validation.txt", "validation", validation_data_transform)
validationloader = torch.utils.data.DataLoader(validationset, batch_size=32)

testset = DatasetCTScanKaggleNicuDucal("test.txt", "test", validation_data_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32)


##################### MODELE (Din Pytorch) ###############################
# DenseNet
advanced_network = torchvision.models.densenet161(pretrained=True)
last_good_dense_layer = advanced_network.classifier.in_features
advanced_network.classifier = nn.Linear(in_features=last_good_dense_layer, out_features=3)

# Definirea datelor
# advanced_network = VGG_Simple()
loss_function = nn.CrossEntropyLoss()

# Optimizer
learning_rate = 1e-5
optimizer_function = torch.optim.Adam(advanced_network.parameters(), lr=learning_rate)
# optimizer_function = torch.optim.Adadelta(advanced_network.parameters(), lr=learning_rate)
# optimizer_function = torch.optim.Adagrad(advanced_network.parameters(), lr=learning_rate)
# optimizer_function = torch.optim.SGD(advanced_network.parameters(), lr=learning_rate, nesterov=True, momentum=0.9)


# Loss and accuracy pentru plot
train_losses = []
train_accuracy = []
validation_losses = []
validation_accuracy = []


# Matricea de confuzie (de la laborator)
def confunsion_matrix(predicted_labels, ground_truth_labels):
    num_labels = ground_truth_labels.max() + 1
    conf_mat = torch.from_numpy(np.zeros((num_labels, num_labels)))

    # print(predicted_labels, ground_truth_labels)
    for i in range(len(predicted_labels)):
        conf_mat[ground_truth_labels[i], predicted_labels[i]] += 1
    return conf_mat


all_confusion_matrices = []


# Functie pentru evaluarea modelului pe validation data
def evaluaza_validare_model(data):
    advanced_network.eval()
    total_accuracy = 0
    total = 0
    total_loss_rate = 0.0
    with torch.no_grad():
        confusion_matrice = torch.from_numpy(np.zeros((3, 3)))
        for photos, labels in tqdm(data):
            y_predictions = advanced_network(photos)
            loss = loss_function(y_predictions, labels)
            confusion_matrice += confunsion_matrix(y_predictions.argmax(dim=1), labels)
            total_loss_rate += loss.item()
            total_accuracy += torch.sum(y_predictions.argmax(dim=1) == labels)
            total += len(photos)
        print(f"Loss = {total_loss_rate / len(data)}, Accuracy = {total_accuracy / total}")
        validation_losses.append(total_loss_rate / len(data))
        validation_accuracy.append(total_accuracy / total)
        all_confusion_matrices.append(confusion_matrice)


# Functie care face predictiile pentru test data
def preziceri_test_si_scriere_csv(data):
    advanced_network.eval()
    labels = torch.tensor([])
    with torch.no_grad():
        for photos, fake_labels in tqdm(data):
            y_predictions = advanced_network(photos)
            labels = torch.cat((labels, torch.argmax(y_predictions, dim=1)))
        return labels


# Write to submission file
if not os.path.isdir("submissions_folder"):
    os.mkdir("submissions_folder")


def scrie_predictii(epoch, test_labels):
    with open(f"submissions_folder/submission_epoch_{epoch}.csv", "w") as csvfile:
        csvfile.write("id,label\n")
        for i in range(len(test_labels)):
            file_name = testloader.dataset.nume_labeluri[i][0]
            label = int(test_labels[i])
            # print(file_name, label)
            csvfile.write(f"{file_name},{label}\n")


# Functia care antreneaza modelul
def full_antrenator_model(data, epochs):
    global train_losses, train_accuracy, validation_losses, validation_accuracy, all_confusion_matrices
    # Initializarea datelor
    train_losses = []
    train_accuracy = []
    validation_losses = []
    validation_accuracy = []
    all_confusion_matrices = []
    loss_rate = 0.0
    correct_labels = 0
    total_data = 0
    total_batches = len(data)
    for epoch in trange(epochs, desc="Epoch:"):
        advanced_network.train()
        for photos, labels in tqdm(data):
            y_predictions = advanced_network(photos)
            correct_labels += torch.sum(y_predictions.argmax(dim=1) == labels)
            total_data += len(photos)
            loss = loss_function(y_predictions, labels)
            loss_rate += loss.item()

            # Learning
            optimizer_function.zero_grad()
            loss.backward()
            optimizer_function.step()
        print(f"Epoch {epoch}, Loss = {loss_rate / total_batches}, Accuracy = {correct_labels / total_data}")
        train_losses.append(loss_rate / total_batches)
        train_accuracy.append(correct_labels / total_data)
        loss_rate = 0.0
        correct_labels = 0
        total_data = 0

        # Testing on validation
        print(f"Validation after epoch {epoch}: ")
        evaluaza_validare_model(validationloader)

        # Making submissions
        test_labels = preziceri_test_si_scriere_csv(testloader).cpu().numpy()
        scrie_predictii(epoch, test_labels)


# Antrenarea
full_antrenator_model(trainloader, 30)


# Plotare Loss
epochs = range(30)
plt.plot(epochs, train_losses, label='Train', marker='o')
plt.plot(epochs, validation_losses, label='Val', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('losses.png')


# Plotare Loss
epochs = range(30)
plt.plot(epochs, train_accuracy, label='Train', marker='o')
plt.plot(epochs, validation_accuracy, label='Val', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy.png')