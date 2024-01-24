import gc
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from config import get_config
from data_load import dataset_read
from graph_utils import load_data, make_neighbor_graph
from model import Net

def main():
    args = get_config()
    gc.collect()
    torch.cuda.empty_cache()

    # Device setup
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loading
    go_SS_embed = torch.load(args.go_SS_embdding)
    go_id, go_embedding = go_SS_embed["go_id"], go_SS_embed["embedding"].to(device)
    train_data, val_data, test_data, sp_list = dataset_read(args, go_id)
    print(f"Number of GO terms: {len(go_id)}")
    adj, node_feat = load_data(Path(args.go_HR_dir), go_id, device)
    print("Dataset loaded")

    # Model setup
    model = Net(seq_feature=26, go_feature=1024, nhid=args.nhid, kernel_size=args.kernel_size, dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    # Training loop
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        total_loss, total_triplet_loss = 0.0, 0.0
        for i, (seq_onehot, seq_embed, target) in enumerate(train_data):
            seq_embed, target = seq_embed.to(device), target.type(torch.FloatTensor).to(device)
            optimizer.zero_grad()

            h_semantic, h_structure, pred = model(seq_embed, go_embedding, adj)
            h_semantic_p = make_neighbor_graph(h_semantic, go_id, sp_list, device)
            loss_triplet = compute_triplet_loss(h_semantic, h_semantic_p, h_structure, triplet_loss, args.nneg, device)
            loss_cl = criterion(pred, target)
            loss = 0.5 * loss_triplet + 0.5 * loss_cl

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_triplet_loss += loss_triplet.item()

            if (i + 1) % args.print_every == 0:
                print_epoch_stats(start_time, epoch, i, total_loss, total_triplet_loss, args.print_every)

        save_model_checkpoint(model, optimizer, epoch, loss, args.checkpoint_dir, args.save_model_dir)
        evaluate_on_validation_set(model, val_data, criterion, triplet_loss, args, device, start_time)

def compute_triplet_loss(h_semantic, h_semantic_p, h_structure, triplet_loss, nneg, device):
    loss_mar_sem, loss_mar_str = 0.0, 0.0
    for j in range(nneg):
        indices = torch.randperm(h_semantic.size(0)).to(device)
        h_semantic_n = torch.index_select(h_semantic, dim=0, index=indices)
        loss_mar_sem += triplet_loss(h_semantic, h_semantic_p, h_semantic_n) / nneg
        loss_mar_str += triplet_loss(h_semantic, h_structure, h_semantic_n) / nneg
    return 0.5 * (loss_mar_sem + loss_mar_str)

def print_epoch_stats(start_time, epoch, iteration, total_loss, total_triplet_loss, print_every):
    avg_loss = total_loss / print_every

def print_epoch_stats(start_time, epoch, iteration, total_loss, total_triplet_loss, print_every):
    avg_loss = total_loss / print_every
    avg_triplet_loss = total_triplet_loss / print_every
    elapsed_time = (time.time() - start_time) // 60
    print(f"Time: {elapsed_time}m, Epoch: {epoch + 1}, Iteration: {iteration + 1}, "
          f"Average Loss: {avg_loss:.3f}, Average Triplet Loss: {avg_triplet_loss:.3f}")

def save_model_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, save_model_dir):
    checkpoint_path = f"{checkpoint_dir}/epoch_{epoch + 1}.pth"
    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'model_state_dict': model.state_dict(),
        'loss': loss
    }, checkpoint_path)
    weights_path = f"{save_model_dir}/weights_epoch_{epoch + 1}.pth"
    torch.save(model, weights_path)

def evaluate_on_validation_set(model, val_data, criterion, triplet_loss, args, device, start_time):
    model.eval()
    with torch.no_grad():
        val_total_loss = 0.0
        for i, (seq_onehot, seq_embed, target) in enumerate(val_data):
            seq_embed, target = seq_embed.to(device), target.type(torch.FloatTensor).to(device)
            h_semantic, h_structure, pred = model(seq_embed, go_embedding, adj)
            h_semantic_p = make_neighbor_graph(h_semantic, go_id, sp_list, device)
            loss_triplet = compute_triplet_loss(h_semantic, h_semantic_p, h_structure, triplet_loss, args.nneg, device)
            loss_cl = criterion(pred, target)
            loss = 0.5 * loss_triplet + 0.5 * loss_cl
            val_total_loss += loss.item()

            if (i + 1) % args.print_every == 0:
                avg_loss = val_total_loss / args.print_every
                print(f"Validation - Time: {((time.time() - start_time) // 60)}m, Epoch: {epoch + 1}, "
                      f"Iteration: {i + 1}, Average Loss: {avg_loss:.3f}")

if __name__ == '__main__':
    main()
