from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

writer.add_image()
writer.add_scalar("y=2x", 0, 0)


writer.close()
