from discriminator import Discriminator
from generator import Generator
from dataloaders import Colorization, datalaoder
import utils
import torch
import torch.optim as optim
import tqdm as tqdm


train_dir = "/content/data/train_color"
test_dir = "/content/data/test_color"

train_paths_list = utils.get_file_path(train_dir)
test_paths_list = utils.get_file_path(test_dir)

train_dataset = Colorization(train_paths_list, size=256, split="train")
test_dataset = Colorization(test_paths_list, size=256, split="test")

train_dataloader = datalaoder.datalaoders(
    dataset=train_dataset, BATCH_SIZE=32, shuffle=True
)
test_dataloader = datalaoder.datalaoders(
    dataset=test_dataset, BATCH_SIZE=32, shuffle=False
)

device = "cuda" if torch.cuda.is_available() else "cpu"

gen = Generator(in_channels=1).to(device)
dis = Discriminator(in_channels=3).to(device)


bce = torch.nn.BCEWithLogitsLoss()
l1 = torch.nn.L1Loss()
l1lambda = 100

gen_optimizer = optim.Adam(gen.parameters(), lr=0.002, betas=(0.5, 0.999))
dis_optimizer = optim.Adam(dis.parameters(), lr=0.002, betas=(0.5, 0.999))


dis.train()
gen.train()
epochs = 10

for epoch in tqdm(range(epochs)):
    dis_loss = 0
    gen_loss = 0
    # Train Discriminator
    for idx, batch in enumerate(train_dataloader):
        L = batch["L"].to(device)
        ab = batch["ab"].to(device)

        # Train Discriminator
        fake_color = gen(L)

        fake_image = torch.cat([L, fake_color], dim=1)

        D_fake_preds = dis(fake_image.detach())
        loss_D_fake = bce(D_fake_preds, torch.zeros_like(D_fake_preds))
        real_image = torch.cat([L, ab], dim=1)
        D_real_preds = dis(real_image)
        loss_D_real = bce(D_real_preds, torch.ones_like(D_real_preds))

        D_loss = (loss_D_fake + loss_D_real) / 2

        dis_optimizer.zero_grad()
        D_loss.backward()
        dis_optimizer.step()

        dis_loss += D_loss.item()

        # Train Generator
        fake_image = torch.cat([L, fake_color], dim=1)
        fake_preds = dis(fake_image)
        loss_G_GAN = bce(fake_preds, torch.ones_like(fake_preds))
        loss_G_L1 = l1(fake_color, ab) * 100

        G_loss = loss_G_GAN + loss_G_L1

        gen_optimizer.zero_grad()
        G_loss.backward()
        gen_optimizer.step()

        gen_loss += G_loss.item()

    dis_loss /= len(train_dataloader)
    gen_loss /= len(train_dataloader)

    print(
        f"Epoch {epoch+1}/{epochs} Discriminator Loss: {dis_loss:.4f}, Generator Loss: {gen_loss:.4f}"
    )


gen_model_save_path = "/content/drive/MyDrive/Colorize/gen_model.pth"
dis_model_save_path = "/content/drive/MyDrive/Colorize/dis_model.pth"
torch.save(obj=gen.state_dict(), f=gen_model_save_path)
torch.save(obj=dis.state_dict(), f=dis_model_save_path)
