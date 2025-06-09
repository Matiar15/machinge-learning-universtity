import pandas as pd
import torch


class FashionMNISTDataset(torch.utils.data.Dataset):
    """
    Własny zbiór danych MNIST z pliku CSV dla PyTorch.

    Klasa ładuje dane z pliku CSV, w którym każda linia reprezentuje jeden przykład:
    - kolumna "label" zawiera etykietę (cyfra 0–9),
    - pozostałe kolumny zawierają wartości pikseli (0–255) z obrazu 28x28.

    Dane są normalizowane (podzielone przez 255) i przekształcane w tensory typu float32
    o kształcie (1, 28, 28) dla obrazów oraz long dla etykiet.

    Parametry:
    ----------
    file : str
        Ścieżka do pliku CSV zawierającego dane MNIST.

    Atrybuty:
    ---------
    X_train : torch.Tensor
        Tensor wejściowy o kształcie (N, 1, 28, 28), znormalizowane obrazy.
    y_train : torch.Tensor
        Tensor etykiet o kształcie (N,), liczby całkowite 0–9.

    Przykład użycia:
    ----------------
    >>> dataset = FashionMNISTDataset("mnist_train.csv")
    >>> len(dataset)
    60000
    >>> image, label = dataset[0]
    >>> image.shape
    torch.Size([1, 28, 28])
    >>> label
    tensor(5)
    """

    def __init__(self, file, **kwargs):
        self.transform = kwargs.get("transform")

        csv = pd.read_csv(file)
        x = csv.drop("label", axis=1).to_numpy()
        y = csv["label"].to_numpy()

        self.X_train = torch.tensor(x / 255.0, dtype=torch.float32).reshape(-1, 1, 28, 28)
        self.y_train = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y_train )

    def __getitem__(self, idx):
        image = self.X_train[idx]
        label = self.y_train[idx]
        if self.transform:
            image = self.transform(image)
        return image, label