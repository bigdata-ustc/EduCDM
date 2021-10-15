# coding: utf-8
# 2021/6/19 @ tongshiwei

import torch
from torch import nn
from tqdm import tqdm
from EduCDM.IRT.GD import IRT as PointIRT
import numpy as np
import pandas as pd
from .loss import PairSCELoss, HarmonicLoss, loss_mask
from longling.ML.metrics import ranking_report

__all__ = ["IRT"]


class IRT(PointIRT):
    def __init__(self, user_num, item_num, knowledge_num, value_range=10, zeta=0.5):
        super(IRT, self).__init__(user_num, item_num, value_range=value_range)
        self.knowledge_num = knowledge_num
        self.zeta = zeta

    def train(self, train_data, test_data=None, *, epoch: int, device="cpu", lr=0.001) -> ...:
        point_loss_function = nn.BCELoss()
        pair_loss_function = PairSCELoss()
        loss_function = HarmonicLoss(self.zeta)

        trainer = torch.optim.Adam(self.irt_net.parameters(), lr, weight_decay=1e-4)

        for e in range(epoch):
            point_losses = []
            pair_losses = []
            losses = []
            for batch_data in tqdm(train_data, "Epoch %s" % e):
                user_id, item_id, _, score, n_samples, *neg_users = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                predicted_pos_score: torch.Tensor = self.irt_net(user_id, item_id)
                score: torch.Tensor = score.to(device)
                neg_score = 1 - score

                point_loss = point_loss_function(predicted_pos_score, score)
                predicted_neg_scores = []
                if neg_users:
                    for neg_user in neg_users:
                        predicted_neg_score = self.irt_net(neg_user, item_id)
                        predicted_neg_scores.append(predicted_neg_score)

                    # prediction loss
                    pair_pred_loss_list = []
                    for i, predicted_neg_score in enumerate(predicted_neg_scores):
                        pair_pred_loss_list.append(
                            pair_loss_function(
                                predicted_pos_score,
                                predicted_neg_score,
                                score - neg_score
                            )
                        )

                    pair_loss = sum(loss_mask(pair_pred_loss_list, n_samples))
                else:
                    pair_loss = 0

                loss = loss_function(point_loss, pair_loss)

                # back propagation
                trainer.zero_grad()
                loss.backward()
                trainer.step()

                point_losses.append(point_loss.mean().item())
                pair_losses.append(pair_loss.mean().item() if not isinstance(pair_loss, int) else pair_loss)
                losses.append(loss.item())
            print(
                "[Epoch %d] Loss: %.6f, PointLoss: %.6f, PairLoss: %.6f" % (
                    e, float(np.mean(losses)), float(np.mean(point_losses)), float(np.mean(pair_losses))
                )
            )

            if test_data is not None:
                eval_data = self.eval(test_data)
                print("[Epoch %d]\n%s" % (e, eval_data))

    def eval(self, test_data, device="cpu"):
        self.irt_net.eval()
        y_pred = []
        y_true = []
        items = []
        for batch_data in tqdm(test_data, "evaluating"):
            user_id, item_id, _, response = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            pred: torch.Tensor = self.irt_net(user_id, item_id)
            y_pred.extend(pred.tolist())
            y_true.extend(response.tolist())
            items.extend(item_id.tolist())

        df = pd.DataFrame({
            "item_id": items,
            "score": y_true,
            "pred": y_pred,
        })

        ground_truth = []
        prediction = []

        for _, group_df in tqdm(df.groupby("item_id"), "formatting item df"):
            ground_truth.append(group_df["score"].values)
            prediction.append(group_df["pred"].values)

        self.irt_net.train()

        return ranking_report(
            ground_truth,
            y_pred=prediction,
            coerce="padding"
        )
