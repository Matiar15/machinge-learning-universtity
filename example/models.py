import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    """
    Convolutional Neural Network (CNN) for image classification.

    Architektura składa się z dwóch warstw konwolucyjnych z funkcjami aktywacji ReLU
    i warstwami MaxPooling, po których następują dwie w pełni połączone warstwy (fully connected).
    Model jest przeznaczony do przetwarzania obrazów wejściowych o rozmiarze 28x28 pikseli i 1 kanale (np. MNIST).

    Warstwy:
    --------
    conv1 : nn.Conv2d
        Pierwsza warstwa konwolucyjna (1 → 16 kanałów), kernel 3x3, padding=1.
    pool : nn.MaxPool2d
        Warstwa maksymalnego próbkowania o kernelu 2x2 i kroku 2.
    conv2 : nn.Conv2d
        Druga warstwa konwolucyjna (16 → 32 kanały), kernel 3x3, padding=1.
    fc1 : nn.Linear
        W pełni połączona warstwa: 32 * 7 * 7 → 128.
    fc2 : nn.Linear
        W pełni połączona warstwa: 128 → 10 (liczba klas).

    Przykład:
    ---------
    >>> import torch
    >>> model = CNNModel()
    >>> images = torch.randn(64, 1, 28, 28)  # batch 64 obrazów 28x28
    >>> outputs = model(images)
    >>> outputs.shape
    torch.Size([64, 10])
    """

    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNNModel2(nn.Module):
    """
    Convolutional Neural Network (CNN) with Batch Normalization.

    Rozszerzona wersja bazowego modelu, zawierająca warstwy normalizacji wsadowej (BatchNorm)
    po każdej warstwie konwolucyjnej oraz jednej w pełni połączonej (fc1).
    BatchNorm przyspiesza trening i poprawia stabilność poprzez normalizację aktywacji.

    Architektura modelu zakłada wejściowe obrazy 28x28 pikseli z jednym kanałem (np. MNIST).

    Warstwy:
    --------
    conv1 : nn.Conv2d
        Pierwsza warstwa konwolucyjna (1 → 16 kanałów), kernel 3x3, padding=1.
    bn1 : nn.BatchNorm2d
        Normalizacja wsadowa po conv1.
    pool : nn.MaxPool2d
        Warstwa maksymalnego próbkowania o kernelu 2x2 i kroku 2.
    conv2 : nn.Conv2d
        Druga warstwa konwolucyjna (16 → 32 kanały), kernel 3x3, padding=1.
    bn2 : nn.BatchNorm2d
        Normalizacja wsadowa po conv2.
    fc1 : nn.Linear
        W pełni połączona warstwa: 32 * 7 * 7 → 128.
    bn3 : nn.BatchNorm1d
        Normalizacja wsadowa po fc1.
    fc2 : nn.Linear
        W pełni połączona warstwa: 128 → 10 (liczba klas).

    Przykład:
    ---------
    >>> import torch
    >>> model = CNNModel2()
    >>> images = torch.randn(64, 1, 28, 28)  # batch 64 obrazów
    >>> outputs = model(images)
    >>> outputs.shape
    torch.Size([64, 10])
    """
    def __init__(self):
        super(CNNModel2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x

class CNNModel3(nn.Module):
    """
    Convolutional Neural Network (CNN) with Batch Normalization and Dropout.

    Rozszerzona wersja bazowego modelu zawierająca:
    - normalizację wsadową (BatchNorm) po warstwach konwolucyjnych i jednej w pełni połączonej,
    - regularizację Dropout (p=0.5) po pierwszej warstwie w pełni połączonej (fc1),
      co pomaga redukować przeuczenie (overfitting).

    Model przeznaczony do klasyfikacji obrazów 28x28 z 1 kanałem (np. MNIST).

    Warstwy:
    --------
    conv1 : nn.Conv2d
        Pierwsza warstwa konwolucyjna (1 → 16 kanałów), kernel 3x3, padding=1.
    bn1 : nn.BatchNorm2d
        Normalizacja wsadowa po conv1.
    pool : nn.MaxPool2d
        Warstwa maksymalnego próbkowania o kernelu 2x2 i kroku 2.
    conv2 : nn.Conv2d
        Druga warstwa konwolucyjna (16 → 32 kanały), kernel 3x3, padding=1.
    bn2 : nn.BatchNorm2d
        Normalizacja wsadowa po conv2.
    fc1 : nn.Linear
        W pełni połączona warstwa: 32 * 7 * 7 → 128.
    bn3 : nn.BatchNorm1d
        Normalizacja wsadowa po fc1.
    dropout : nn.Dropout
        Dropout z prawdopodobieństwem p=0.5 po fc1.
    fc2 : nn.Linear
        W pełni połączona warstwa: 128 → 10 (klasy wyjściowe).

    Przykład:
    ---------
    >>> import torch
    >>> model = CNNModel3()
    >>> images = torch.randn(64, 1, 28, 28)
    >>> outputs = model(images)
    >>> outputs.shape
    torch.Size([64, 10])
    """
    def __init__(self):
        super(CNNModel3, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


