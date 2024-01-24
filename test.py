import gc
import torch
import numpy as np
from pathlib import Path
from config import get_config
from data_load import dataset_read
from graph_utils import load_data, make_neighbor_graph

def main():
    # Initialize configuration and environment settings
    args = get_config()
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset and embeddings
    go_SS_embed = torch.load(args.go_SS_embdding)
    go_id, go_embedding = go_SS_embed["go_id"], go_SS_embed["embedding"].to(device)
    train_data, test_data, sp_list = dataset_read(args, go_id)
    print(f"Number of GO terms: {len(go_id)}")
    adj, node_feat = load_data(Path(args.go_HR_dir), go_id, device)

    # Model prediction loop
    for epoch in range(4, 21):
        model_path = Path('Weights/MFO') / f'weights_epoch_{epoch}.pth'
        model = torch.load(model_path).to(device)
        save_path = f'Data/CAFA3/test/MFO/MFO_pred_epoch{epoch}.npy'
        print("Finished dataset loading")

        model.eval()
        all_preds = []
        with torch.no_grad():
            for seq_onehot, seq_embed, target in test_data:
                seq_embed = seq_embed.to(device)
                _, _, pred = model(seq_embed, go_embedding, adj)
                all_preds.append(pred.cpu())

            combined_preds = torch.cat(all_preds, dim=0).numpy()
            np.save(save_path, combined_preds)
            print(f"Epoch {epoch}: Predictions saved")

if __name__ == '__main__':
    main()
