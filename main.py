import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import torch.nn as nn
import torch.optim as optim

# Установим глобальную переменную для устройства
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Класс для загрузки данных
class AnimeDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 1])
        image = Image.open(img_name).convert("RGB")  # Конвертируем изображение в RGB
        label = self.dataframe.iloc[idx, 2] if 'class' in self.dataframe.columns else -1

        if self.transform:
            image = self.transform(image)

        return image, label


# Функция для подготовки данных
def prepare_data(train_csv, train_dir, test_dir, submission_csv):
    train_df = pd.read_csv(train_csv)
    submission_df = pd.read_csv(submission_csv)

    # Убедимся, что пути корректны
    print("Train DataFrame head:\n", train_df.head())
    print("Submission DataFrame head:\n", submission_df.head())

    # Проверяем существование всех файлов
    for path in train_df['path']:
        if not os.path.isfile(os.path.join(train_dir, path)):
            print(f"File not found: {os.path.join(train_dir, path)}")

    for path in submission_df['path']:
        if not os.path.isfile(os.path.join(test_dir, path)):
            print(f"File not found: {os.path.join(test_dir, path)}")

    # Трансформации с аугментацией для тренировочного набора данных
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Трансформации без аугментации для тестового набора данных
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = AnimeDataset(train_df, train_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = AnimeDataset(submission_df, test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, submission_df


# Функция для создания и обучения модели
def train_model(train_loader, num_classes, num_epochs=25):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            # Убедимся, что inputs и labels являются переменными
            print(f"Inputs type: {type(inputs)}, Labels type: {type(labels)}")
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    return model


# Функция для предсказаний и сохранения результатов
def predict_and_save(model, test_loader, submission_df, output_csv):
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    submission_df['class'] = predictions
    submission_df[['id', 'class']].to_csv(output_csv, index=False)


# Пример использования:
train_csv = 'source/train.csv'
train_dir = 'source/train'
test_dir = 'source/test'
submission_csv = 'source/submission.csv'

# Подготовка данных
train_loader, test_loader, submission_df = prepare_data(train_csv, train_dir, test_dir, submission_csv)

# Обучение модели
num_classes = len(pd.read_csv(train_csv)['class'].unique())
model = train_model(train_loader, num_classes)

# Предсказание и сохранение результатов
output_csv = 'submission.csv'
predict_and_save(model, test_loader, submission_df, output_csv)
