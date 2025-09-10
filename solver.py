import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import DepthDataset
from utils import visualize_img, ssim
from model import Net
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW
import wandb 

class Solver():

    def __init__(self, args):
        # prepare a dataset
        self.args = args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = Net().to(self.device)

        # Compare gradients of pred and target (dx, dy) instead of absolute values,
        # to penalize differences on edges and local variations.
        def gradient_edge_loss(pred, target):
            # Horizontal gradients (dx)
            dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            dx_true = target[:, :, :, 1:] - target[:, :, :, :-1]

            # Vertical gradients (dy)
            dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            dy_true = target[:, :, 1:, :] - target[:, :, :-1, :]

            # L1 loss on gradients
            loss_dx = F.l1_loss(dx_pred, dx_true)
            loss_dy = F.l1_loss(dy_pred, dy_true)

            return loss_dx + loss_dy

        # The chosen loss for training is a combination of MSE, edge loss and SSIM loss.
        # MSE dominates, SSIM and edge loss refine the results.
        def combined_loss(pred, target):
            mse = F.mse_loss(pred, target)
            edge = gradient_edge_loss(pred, target)
            ssim_loss = 1 - ssim(pred, target)
            return  mse + 0.05 * edge + 0.15 * ssim_loss

        self.loss_fn = combined_loss

        # Since the encoder is pretrained, use a lower learning rate to avoid
        # changing its weights too much. Separate encoder and decoder parameters,
        # selecting those that start with "convnext" for the encoder.
        enc_lr, dec_lr = [], []
        for n, p in self.net.named_parameters():
            if not p.requires_grad:
                continue
            (enc_lr if n.startswith("convnext") else dec_lr).append(p)

        # Use AdamW optimizer with separate parameter groups for encoder and decoder.
        # AdamW has built-in weight decay which helps prevent overfitting.
        # (If using Adam, L2 regularization wouldn't be decoupled; AdamW uses decoupled weight decay.)
        self.optim = AdamW([
            {"params": enc_lr, "lr": 1e-4},   # encoder 
            {"params": dec_lr, "lr": 1e-3},   # decoder
        ])

        self.args = args

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
            
            steps_per_epoch = len(self.train_loader)

            # Use a OneCycleLR scheduler to manage the learning rate during training.
            # It increases the learning rate up to a peak (2e-4 for encoder, 8e-4 for decoder)
            # then decreases it to a very small learning rate (annealing).
            self.scheduler = OneCycleLR(
                self.optim,
                max_lr=[2e-4, 8e-4],   # [encoder, decoder]
                epochs=self.args.max_epochs,
                steps_per_epoch=steps_per_epoch,
                cycle_momentum=False # Disable momentum cycling; typically used with SGD and not needed with AdamW.
            )

            if not os.path.exists(args.ckpt_dir):
                os.makedirs(args.ckpt_dir)
        else:
            self.test_set = DepthDataset(train=DepthDataset.TEST,
                                    data_dir=self.args.data_dir)
            ckpt_file = os.path.join("checkpoint", self.args.ckpt_file)
            self.net.load_state_dict(torch.load(ckpt_file, weights_only=True))

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
                self.scheduler.step()

                train_loss += loss.item()
                
                print("Epoch [{}/{}] Loss: {:.3f} ".
                      format(epoch + 1, args.max_epochs, loss.item()))
            
            #wandb.log({"mean_train_loss": train_loss / len(self.train_loader), "Epoch": epoch + 1}) # Log average train loss per epoch to wandb

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
        print("RMSE on", suffix, ":", rmse_acc / len(loader))
        print("SSIM on", suffix, ":", ssim_acc / len(loader))

        # wandb.log({f"{suffix} RMSE": rmse_acc/len(loader), f"{suffix} SSIM": ssim_acc/len(loader), "Epoch": epoch})

    def save(self, ckpt_dir, ckpt_name, global_step):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)

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
