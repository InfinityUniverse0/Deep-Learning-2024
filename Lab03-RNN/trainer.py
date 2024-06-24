"""
Trainer class for training the model
"""

import torch
from sklearn.metrics import confusion_matrix
import time


class Trainer:
    """
    Trainer
    """
    def __init__(self, model, criterion, optimizer, device):
        """
        Constructor

        :param model: model
        :param criterion: loss function
        :param optimizer: optimizer
        :param device: device
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)

    def train(self, train_loader, valid_loader, num_epochs):
        """
        Train

        :param train_loader: train data loader
        :param valid_loader: valid data loader
        :param num_epochs: number of epochs
        :return: train loss list, train accuracy list, valid loss list, valid accuracy list
        """
        train_loss_list = []
        train_accuracy_list = []
        valid_loss_list = []
        valid_accuracy_list = []
        for epoch in range(num_epochs):
            start_time = time.time()  # Start Time

            self.model.train()
            train_loss = 0.0
            train_accuracy = 0.0
            total = 0
            for i, (names, labels, lengths) in enumerate(train_loader):
                batch_size = labels.size(0)
                names = names.to(self.device)
                labels = labels.to(self.device)
                # lengths should be a 1D CPU int64 tensor

                self.optimizer.zero_grad()
                outputs, h = self.model(names, h0=None, lengths=lengths)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * batch_size
                total += batch_size

                _, predictions = torch.max(outputs, 1)
                train_accuracy += (predictions == labels).sum().item()

            train_loss /= total
            train_accuracy /= total

            valid_loss, valid_accuracy, conf_mat = self.evaluate(valid_loader, verbose=False)

            train_loss_list.append(train_loss)
            train_accuracy_list.append(train_accuracy)
            valid_loss_list.append(valid_loss)
            valid_accuracy_list.append(valid_accuracy)

            end_time = time.time()  # End Time
            elapsed_time = end_time - start_time  # Elapsed Time

            print(f'Epoch {epoch + 1}/{num_epochs}, '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Train Accuracy: {train_accuracy:.4f}, '
                  f'Valid Loss: {valid_loss:.4f}, '
                  f'Valid Accuracy: {valid_accuracy:.4f}, '
                  f'Time: {elapsed_time:.2f}s')

        print('Training Finished')
        return train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list

    def evaluate(self, test_loader, verbose=False):
        """
        Evaluate

        :param test_loader: test data loader
        :param verbose: verbose flag
        :return: test loss, test accuracy, confusion matrix
        """
        test_loss = 0.0
        test_accuracy = 0.0
        total = 0
        all_preds = []
        all_labels = []
        self.model.eval()
        with torch.no_grad():
            for i, (names, labels, lengths) in enumerate(test_loader):
                batch_size = labels.size(0)
                names = names.to(self.device)
                labels = labels.to(self.device)
                # lengths should be a 1D CPU int64 tensor

                outputs, h = self.model(names, h0=None, lengths=lengths)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item() * batch_size
                total += batch_size

                _, predictions = torch.max(outputs, 1)
                test_accuracy += (predictions == labels).sum().item()

                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss /= total
        test_accuracy /= total

        conf_mat = confusion_matrix(all_labels, all_preds)

        if verbose:
            print('Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(test_loss, test_accuracy))
            print('Confusion Matrix:')
            print(conf_mat)

        return test_loss, test_accuracy, conf_mat
