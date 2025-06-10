import ever as er
import numpy as np
import torch
from tqdm import tqdm

er.registry.register_all()


def register_evaluate_fn(launcher):
    launcher.override_evaluate(evaluate_levircd)


def evaluate_levircd(self, test_dataloader, config=None):
    self.model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    metric_op = er.metric.PixelMetric(2,
                                      self.model_dir,
                                      logger=self.logger)

    with torch.no_grad():
        for img, ret_gt in tqdm(test_dataloader):
            img = img.to(device)

            change = self.model.module(img).sigmoid() > 0.5

            pr_change = change.cpu().numpy().astype(np.uint8)
            gt_change = ret_gt['change']
            gt_change = gt_change.numpy()
            y_true = gt_change.ravel()
            y_pred = pr_change.ravel()

            y_true = np.where(y_true > 0, np.ones_like(y_true), np.zeros_like(y_true))

            metric_op.forward(y_true, y_pred)

    metric_op.summary_all()
    torch.cuda.empty_cache()

try:
    from core import field
except ImportError:
    # If core.field is not directly accessible, you might need to adjust PYTHONPATH
    # or manually define the field name if you know it (e.g., MASK1 = 'mask1_key')
    print("Warning: Could not import 'field' from core.field. Assuming default mask key.")
    class MockField: # Mock class if field module is truly not accessible
        MASK1 = 'mask' # Default key if not found
    field = MockField()


# Rename function for clarity (optional, but good practice)
def evaluate_xview2_building_segmentation(self, test_dataloader, config=None):
    self.model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # PixelMetric(2) is suitable for binary segmentation (building vs. non-building)
    metric_op = er.metric.PixelMetric(2,
                                      self.model_dir,
                                      logger=self.logger)

    with torch.no_grad():
        for img, ret_gt in tqdm(test_dataloader):
            img = img.to(device) # Input is now a 3-channel image from xView2

            # The model was trained with 'semantic' loss. It should output a semantic mask.
            # Assuming self.model.module(img) directly returns the semantic prediction tensor
            # (as suggested by the `train_sup_change.py` evaluation pattern)
            predicted_mask_prob = self.model.module(img).sigmoid() # Get probability for building mask

            # Threshold to get binary mask. Select the first channel (index 0) if multiple channels
            # are output, which is common for binary segmentation (e.g., for background/foreground classes).
            predicted_mask = (predicted_mask_prob[:, 0, :, :] > 0.5).float() # Pick the foreground channel

            pr_mask = predicted_mask.cpu().numpy().astype(np.uint8)
            # Ground truth is now the building mask from xView2
            # The xview2_dataset.py returns `y[field.MASK1] = mask`
            gt_mask = ret_gt[field.MASK1] # Access the mask using field.MASK1
            gt_mask = gt_mask.numpy()

            # Ensure ground truth and prediction are flattened for metric calculation
            y_true = gt_mask.ravel()
            y_pred = pr_mask.ravel()

            # y_true should already be 0 or 1 from the _target.png
            # (No need for `np.where(y_true > 0, ...)` if targets are already binary 0/1)

            metric_op.forward(y_true, y_pred)

    metric_op.summary_all()
    torch.cuda.empty_cache()
if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    SEED = 2333
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    trainer = er.trainer.get_trainer('th_amp_ddp')()
    blob = trainer.run(after_construct_launcher_callbacks=[register_evaluate_fn])
    #blob = trainer.evaluate(after_construct_launcher_callbacks=[register_evaluate_fn])
    # blob = trainer.evaluate(after_construct_launcher_callbacks=[
    #     lambda launcher: launcher.override_evaluate(evaluate_xview2_building_segmentation)
    # ])