from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("AlexNet_logs")

for i in range(100):
    writer.add_scalar("123", i, i)

writer.close()