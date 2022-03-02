import time

import numpy as np
import torch


class Trainer:
    def __init__(
            self,
            model,
            device,
            optimizer,
            criterion,
            loss_meter,
            score_meter
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.loss_meter = loss_meter
        self.score_meter = score_meter

        self.best_valid_score = -np.inf
        self.n_patience = 0

        self.messages = {
            "epoch": "[Epoch {}: {}] loss: {:.5f}, score: {:.5f}, time: {} s",
            "checkpoint": "The score improved from {:.5f} to {:.5f}. Save model to '{}'",
            "patience": "\nValid score didn't improve last {} epochs."
        }

    def fit(self, epochs, train_loader, valid_loader, save_path, patience, train_num, val_num):
        for n_epoch in range(1, epochs + 1):
            self.info_message("EPOCH: {}", n_epoch)

            train_loss, train_score, train_time = self.train_epoch(train_loader, train_num)
            valid_loss, valid_score, valid_time = self.valid_epoch(valid_loader, val_num)

            self.info_message(
                self.messages["epoch"], "Train", n_epoch, train_loss, train_score, train_time
            )

            self.info_message(
                self.messages["epoch"], "Valid", n_epoch, valid_loss, valid_score, valid_time
            )

            if True:
                #             if self.best_valid_score < valid_score:
                self.info_message(
                    self.messages["checkpoint"], self.best_valid_score, valid_score, save_path
                )
                self.best_valid_score = valid_score
                self.save_model(n_epoch, save_path)
                self.n_patience = 0
            else:
                self.n_patience += 1

            if self.n_patience >= patience:
                self.info_message(self.messages["patience"], patience)
                break

    def train_epoch(self, train_loader, train_num):
        self.model.train()
        t = time.time()
        train_loss = self.loss_meter()
        train_score = self.score_meter()

        acc = 0.0

        for step, batch in enumerate(train_loader, 1):
            X = batch["X"].to(self.device)
            targets = batch["y"].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X).squeeze(1)

            
            predict_y = torch.max(outputs, dim=0)[1]
            acc += torch.eq(predict_y, targets).sum().item()


            loss = self.criterion(outputs, targets)
            loss.backward()

            train_loss.update(loss.detach().item())
            train_score.update(targets, outputs.detach())

            self.optimizer.step()

            # print(self.optimizer.param_groups[0]['lr'])

            _loss, _score = train_loss.avg, train_score.avg
            message = 'Train Step {}/{}, train_loss: {:.5f}, train_score: {:.5f}'
            self.info_message(message, step, len(train_loader), _loss, _score, end="\r")
        train_accuracy = acc / train_num
        print('train_accuracy: %.3f' % train_accuracy)


        return train_loss.avg, train_score.avg, int(time.time() - t)

    def valid_epoch(self, valid_loader, val_num):
        self.model.eval()
        t = time.time()
        valid_loss = self.loss_meter()
        valid_score = self.score_meter()

        acc = 0.0

        for step, batch in enumerate(valid_loader, 1):
            with torch.no_grad():
                X = batch["X"].to(self.device)
                targets = batch["y"].to(self.device)

                outputs = self.model(X).squeeze(1)

                predict_y = torch.max(outputs, dim=0)[1]
                acc += torch.eq(predict_y, targets).sum().item()

                loss = self.criterion(outputs, targets)

                valid_loss.update(loss.detach().item())
                valid_score.update(targets, outputs)

            _loss, _score = valid_loss.avg, valid_score.avg
            message = 'Valid Step {}/{}, valid_loss: {:.5f}, valid_score: {:.5f}'
            self.info_message(message, step, len(valid_loader), _loss, _score, end="\r")

        val_accuracy = acc / val_num
        print('val_accuracy: %.3f' % val_accuracy)
        return valid_loss.avg, valid_score.avg, int(time.time() - t)

    def save_model(self, n_epoch, save_path):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_valid_score,
                "n_epoch": n_epoch,
            },
            save_path,
        )

    @staticmethod
    def info_message(message, *args, end="\n"):
        print(message.format(*args), end=end)

