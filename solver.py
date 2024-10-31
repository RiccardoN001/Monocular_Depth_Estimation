import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import DepthDataset
from utils import visualize_img, ssim
from model import Net
import wandb 

class Solver():

    def __init__(self, args):
        # prepare a dataset
        self.args = args

        #Inizializzazione dei parametri di wandb per il logging dei risultati tramite la piattaforma online w&b
        wandb.init(project="depth-estimation", config=args)
        wandb.config.update(args)

        if self.args.is_train:
            self.train_data = DepthDataset(train=DepthDataset.TRAIN,
                                           data_dir=args.data_dir,
                                           transform=None)
            self.val_data = DepthDataset(train=DepthDataset.VAL,
                                         data_dir=args.data_dir,
                                         transform=None)

            self.train_loader = DataLoader(dataset=self.train_data,
                                           batch_size=args.batch_size,
                                           num_workers=4,
                                           shuffle=True, drop_last=True)

            if not os.path.exists(args.ckpt_dir):
                os.makedirs(args.ckpt_dir)
        else:
            self.test_set = DepthDataset(train=DepthDataset.TEST,
                                    data_dir=self.args.data_dir)
            ckpt_file = os.path.join("checkpoint", self.args.ckpt_file)
            self.net.load_state_dict(torch.load(ckpt_file, weights_only=True))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = Net().to(self.device)
        self.loss_fn = torch.nn.MSELoss()

        self.optim = torch.optim.Adam(self.net.parameters(), lr=args.lr)

        self.args = args

    def fit(self):
        args = self.args

        for epoch in range(args.max_epochs):
            self.net.train()
            train_loss = 0.0

            for images, depths in self.train_loader:
                
                images, depths = images.to(self.device), depths.to(self.device)

                # Forward pass
                outputs = self.net(images)
                #print(outputs.shape, depths.shape)
                loss = self.loss_fn(outputs, depths)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                train_loss += loss.item()
                
                wandb.log({"train_loss_batch": loss.item(), "Epoch": epoch +1}) #Logging su wandb di train_loss per ogni batch
                wandb.log({"mean_train_loss": train_loss / len(self.train_loader), "Epoch": epoch +1}) #Logging su wandb di train_loss media per epoca

                print("Epoch [{}/{}] Loss: {:.3f} ".
                      format(epoch + 1, args.max_epochs, loss.item()))

            if (epoch + 1) % args.evaluate_every == 0:
                self.evaluate(DepthDataset.TRAIN, epoch + 1)
                self.evaluate(DepthDataset.VAL, epoch + 1)
                self.save(args.ckpt_dir, args.ckpt_name, epoch + 1)
 

    def evaluate(self, set, epoch):

        args = self.args
        if set == DepthDataset.TRAIN:
            dataset = self.train_data
            suffix = "TRAIN"
        elif set == DepthDataset.VAL:
            dataset = self.val_data
            suffix = "VALIDATION"
        else:
            raise ValueError("Invalid set value")

        loader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=4,
                            shuffle=False, drop_last=False)

        self.net.eval()
        ssim_acc = 0.0
        rmse_acc = 0.0
        with torch.no_grad():
            for i, (images, depth) in enumerate(loader):
                output = self.net(images.to(self.device))
                ssim_acc += ssim(output, depth.to(self.device)).item()
                rmse_acc += torch.sqrt(F.mse_loss(output, depth.to(self.device))).item()
                if i % self.args.visualize_every == 0:
                    visualize_img(images[0].cpu(),
                                  depth[0].cpu(),
                                  output[0].cpu().detach(),
                                  suffix=suffix)
                    wandb.log({
                        f"{suffix} Sample {i}": [
                            wandb.Image(images[0].cpu(), caption="Input Image"),
                            wandb.Image(depth[0].cpu(), caption="True Depth"),
                            wandb.Image(output[0].cpu().detach(), caption="Predicted Depth")
                        ]
                    })
        print("RMSE on", suffix, ":", rmse_acc / len(loader))
        print("SSIM on", suffix, ":", ssim_acc / len(loader))

        wandb.log({f"{suffix} RMSE": rmse_acc/len(loader), f"{suffix} SSIM": ssim_acc/len(loader), "Epoch": epoch})

    def save(self, ckpt_dir, ckpt_name, global_step):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)

        wandb.save(save_path)

    def test(self):

        loader = DataLoader(self.test_set,
                            batch_size=self.args.batch_size,
                            num_workers=4,
                            shuffle=False, drop_last=False)

        ssim_acc = 0.0
        rmse_acc = 0.0
        with torch.no_grad():
            for i, (images, depth) in enumerate(loader):
                output = self.net(images.to(self.device))
                ssim_acc += ssim(output, depth.to(self.device)).item()
                rmse_acc += torch.sqrt(F.mse_loss(output, depth.to(self.device))).item()
                if i % self.args.visualize_every == 0:
                    visualize_img(images[0].cpu(),
                                  depth[0].cpu(),
                                  output[0].cpu().detach(),
                                  suffix="TEST")
        print("RMSE on TEST :", rmse_acc / len(loader))
        print("SSIM on TEST:", ssim_acc / len(loader))
