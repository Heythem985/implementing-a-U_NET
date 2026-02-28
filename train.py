import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset import ShapeDataset
from u_net import UNet


def train_one_epoch(model, loader, opt, loss_fn, device):
	model.train()
	total_loss = 0.0
	for imgs, masks in loader:
		imgs = imgs.to(device)
		masks = masks.to(device)

		preds = model(imgs)
		loss = loss_fn(preds, masks)

		opt.zero_grad()
		loss.backward()
		opt.step()

		total_loss += loss.item() * imgs.size(0)

	return total_loss / len(loader.dataset)


def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	dataset = ShapeDataset(num_samples=200, img_size=64)
	loader = DataLoader(dataset, batch_size=8, shuffle=True)

	model = UNet(in_channels=1, out_channels=1).to(device)
	optimizer = optim.Adam(model.parameters(), lr=1e-3)
	loss_fn = nn.BCEWithLogitsLoss()

	epochs = 3
	for ep in range(1, epochs + 1):
		loss = train_one_epoch(model, loader, optimizer, loss_fn, device)
		print(f'Epoch {ep}/{epochs} — loss: {loss:.4f}')

	# save a small checkpoint
	torch.save(model.state_dict(), 'unet_small.pth')
	print('Saved model to unet_small.pth')


if __name__ == '__main__':
	main()

