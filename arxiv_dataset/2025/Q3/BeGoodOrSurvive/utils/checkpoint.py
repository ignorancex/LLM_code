import os

def save_checkpoint(filename, counter, strong_bnn, bnn_history, ground_truth_label_list,
                    ethical_ground_truths, gen_loss_history, gen_ethical_history, overall_summary):
    torch.save({
        'counter': counter,
        'model_state_dict': strong_bnn.state_dict(),
        'bnn_history': bnn_history,
        'ground_truth_label_list': ground_truth_label_list,
        'ethical_ground_truths': ethical_ground_truths,
        'gen_loss_history': gen_loss_history,
        'gen_ethical_history': gen_ethical_history,
        'overall_summary': overall_summary
    }, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(filename, strong_bnn, device):
    if os.path.exists(filename):
        checkpoint = torch.load(filename, map_location=device)
        counter = checkpoint['counter']
        strong_bnn.load_state_dict(checkpoint['model_state_dict'])
        bnn_history = checkpoint['bnn_history']
        ground_truth_label_list = checkpoint['ground_truth_label_list']
        ethical_ground_truths = checkpoint['ethical_ground_truths']
        gen_loss_history = checkpoint['gen_loss_history']
        gen_ethical_history = checkpoint['gen_ethical_history']
        overall_summary = checkpoint['overall_summary']
        print(f"Checkpoint loaded from {filename}")
        return (counter, strong_bnn, bnn_history, ground_truth_label_list,
                ethical_ground_truths, gen_loss_history, gen_ethical_history, overall_summary)
    else:
        print(f"No checkpoint found at {filename}")
        return None

