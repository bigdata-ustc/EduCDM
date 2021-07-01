# coding: utf-8
# 2021/7/1 @ tongshiwei

import pandas as pd
import numpy as np
import torch
from torch import nn
from EduCDM import GDDINA
from .loss import PairSCELoss, HarmonicLoss, loss_mask
from tqdm import tqdm
from longling.ML.metrics import ranking_report


class DINA(GDDINA):
    def __init__(self, user_num, item_num, knowledge_num, ste=False, zeta=0.5):
        super(DINA, self).__init__(user_num, item_num, knowledge_num, ste)
        self.zeta = zeta

    def train(self, train_data, test_data=None, *, epoch: int, device="cpu", lr=0.001) -> ...:
        point_loss_function = nn.BCELoss()
        pair_loss_function = PairSCELoss()
        loss_function = HarmonicLoss(self.zeta)

        trainer = torch.optim.Adam(self.dina_net.parameters(), lr, weight_decay=1e-4)

        for e in range(epoch):
            point_losses = []
            pair_losses = []
            losses = []
            for batch_data in tqdm(train_data, "Epoch %s" % e):
                user_id, item_id, knowledge, score, n_samples, *neg_users = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                knowledge: torch.Tensor = knowledge.to(device)
                predicted_pos_score: torch.Tensor = self.dina_net(user_id, item_id, knowledge)
                score: torch.Tensor = score.to(device)
                neg_score = 1 - score

                point_loss = point_loss_function(predicted_pos_score, score)
                predicted_neg_scores = []
                if neg_users:
                    for neg_user in neg_users:
                        predicted_neg_score = self.dina_net(neg_user, item_id, knowledge)
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
        self.dina_net.eval()
        y_pred = []
        y_true = []
        items = []
        for batch_data in tqdm(test_data, "evaluating"):
            user_id, item_id, knowledge, response = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            pred: torch.Tensor = self.dina_net(user_id, item_id, knowledge)
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

        self.dina_net.train()

        return ranking_report(
            ground_truth,
            y_pred=prediction,
            coerce="padding"
        )
