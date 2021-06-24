# import torch
import tensorflow as tf

# y_true = torch.Tensor([0, 1, 0, 0])
# y_pred = torch.Tensor([-18.6, 0.51, 2.94, -12.8])
# y_true = [0, 1, 0, 0]
# y_pred = [-18.6, 0.51, 2.94, -12.8]
# y_true = torch.Tensor([[0, 1], [0, 0]])
# y_pred = torch.Tensor([[0.6, 0.4], [0.4, 0.6]])
y_true = [[0, 1], [0, 0]]
y_pred = [[0.6, 0.4], [0.4, 0.6]]
# loss = torch.nn.BCELoss(reduction='none')
# losses = loss(y_pred, y_true)
# losses = torch.mean(loss(y_pred, y_true), axis = 0)
bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

print(bce.numpy())

# print(losses)