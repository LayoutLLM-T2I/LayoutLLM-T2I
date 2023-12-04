import os
import torch as th
import torch


class GroundingNetInput:
    def __init__(self):
        self.set = False

    # @torch.no_grad()
    def prepare(self, batch, text_encoder):
        """
        batch should be the output from dataset.
        Please define here how to process the batch and prepare the
        input only for the ground tokenizer.
        """

        self.set = True
        boxes = batch['boxes']
        masks = batch['masks']
        self.batch, self.max_box, _ = boxes.shape
        self.device = boxes.device
        self.in_dim = 768

        if "text_embeddings" in batch:
            positive_embeddings = batch["text_embeddings"]
        else:
            # labels = [list(category) for category in zip(*batch["labels"])]
            labels = [s.split('|') for s in batch["labels"]]
            box_list = torch.sum(masks, dim=-1).tolist()

            positive_embeddings = torch.zeros((self.batch, self.max_box, self.in_dim)).to(self.device)
            for b in range(self.batch):
                for i in range(box_list[b]):
                    try:
                        positive_embeddings[b, i] = text_encoder.encode_one_token(labels[b][i])
                    except IndexError as e:
                        print(e)
                        print(labels)

        # positive_embeddings = batch["text_embeddings"]

        self.dtype = positive_embeddings.dtype

        return {"boxes": boxes, "masks": masks, "positive_embeddings": positive_embeddings}

    def get_null_input(self, batch=None, device=None, dtype=None):
        """
        Guidance for training (drop) or inference,
        please define the null input for the grounding tokenizer
        """

        assert self.set, "not set yet, cannot call this funcion"
        batch = self.batch if batch is None else batch
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype

        boxes = th.zeros(batch, self.max_box, 4, ).type(dtype).to(device)
        masks = th.zeros(batch, self.max_box).type(dtype).to(device)
        positive_embeddings = th.zeros(batch, self.max_box, self.in_dim).type(dtype).to(device)

        return {"boxes": boxes, "masks": masks, "positive_embeddings": positive_embeddings}
