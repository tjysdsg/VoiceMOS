import argparse
import csv
import fnmatch
import os
import yaml

import numpy as np
import scipy
import torch
from tqdm import tqdm

from dataset import get_dataloader, get_dataset
from models.MBNet import MBNet
from models.LDNet import SLDNet

import scipy.stats
import matplotlib

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_FRAMES = 1250


def find_files(root_dir, query="*.wav", include_root_dir=True):
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files


def save_results(ep, valid_result, test_result, result_path):
    if os.path.isfile(result_path):
        with open(result_path, "r", newline='') as csvfile:
            rows = list(csv.reader(csvfile))
        data = {row[0]: row[1:] for row in rows}
    else:
        data = {}
    data[str(ep)] = valid_result + test_result
    rows = [[k] + v for k, v in data.items()]
    rows = sorted(rows, key=lambda x: int(x[0]))
    with open(result_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)


def inference(mode, model, ep, dataloader, systems, save_dir, name, dataset_name, return_posterior_scores=False):
    if return_posterior_scores:
        assert mode == "all_listeners"

    ep_scores = []
    predict_mean_scores = []
    post_scores = []
    true_mean_scores = []
    predict_sys_mean_scores = {system: [] for system in systems}
    true_sys_mean_scores = {system: [] for system in systems}

    for i, batch in enumerate(tqdm(dataloader)):
        mag_sgrams_padded, avg_scores, sys_names = batch
        mag_sgrams_padded = mag_sgrams_padded.to(device)

        # avoid OOM caused by long samples
        mag_sgrams_padded = mag_sgrams_padded[:, :MAX_FRAMES]

        # forward
        with torch.no_grad():
            if mode == "mean_net":
                pred_mean_scores = model.only_mean_inference(spectrum=mag_sgrams_padded)
            elif mode == "all_listeners":
                pred_mean_scores, posterior_scores = model.average_inference(spectrum=mag_sgrams_padded,
                                                                             include_meanspk=return_posterior_scores)
                posterior_scores = posterior_scores.cpu().detach().numpy()
                post_scores.extend(posterior_scores.tolist())
            elif mode == "mean_listener":
                pred_mean_scores = model.mean_listener_inference(spectrum=mag_sgrams_padded)
            else:
                raise NotImplementedError

            pred_mean_scores = pred_mean_scores.cpu().detach().numpy()
            avg_scores = avg_scores.cpu().detach().numpy()
            predict_mean_scores.extend(pred_mean_scores.tolist())
            true_mean_scores.extend(avg_scores.tolist())
            for j, sys_name in enumerate(sys_names):
                predict_sys_mean_scores[sys_name].append(pred_mean_scores[j])
                true_sys_mean_scores[sys_name].append(avg_scores[j])

            with torch.cuda.device(device):
                torch.cuda.empty_cache()

    predict_mean_scores = np.array(predict_mean_scores)
    true_mean_scores = np.array(true_mean_scores)
    predict_sys_mean_scores = np.array([np.mean(scores) for scores in predict_sys_mean_scores.values()])
    true_sys_mean_scores = np.array([np.mean(scores) for scores in true_sys_mean_scores.values()])

    # plot utterance-level histrogram
    plt.style.use('seaborn-deep')
    bins = np.linspace(1, 5, 40)
    plt.figure(2)
    plt.hist([true_mean_scores, predict_mean_scores], bins, label=['true_mos', 'predict_mos'])
    plt.legend(loc='upper right')
    plt.xlabel('MOS')
    plt.ylabel('number')
    plt.show()
    plt.savefig(os.path.join(save_dir, dataset_name + "_" + mode + "_" + name, f'{ep}_distribution.png'), dpi=150)
    plt.close()

    # utterance level scores
    MSE = np.mean((true_mean_scores - predict_mean_scores) ** 2)
    LCC = np.corrcoef(true_mean_scores, predict_mean_scores)[0][1]
    SRCC = scipy.stats.spearmanr(true_mean_scores, predict_mean_scores)[0]
    KTAU = scipy.stats.kendalltau(true_mean_scores, predict_mean_scores)[0]
    ep_scores += [MSE, LCC, SRCC, KTAU]
    print("[UTTERANCE] {} MSE: {:.3f}, LCC: {:.3f}, SRCC: {:.3f}, KTAU: {:.3f}".format(name, float(MSE), float(LCC),
                                                                                       float(SRCC), float(KTAU)))

    # plotting utterance-level scatter plot
    M = np.max([np.max(predict_mean_scores), 5])
    plt.figure(3)
    plt.scatter(true_mean_scores, predict_mean_scores, s=15, color='b', marker='o', edgecolors='b', alpha=.20)
    plt.xlim([0.5, M])
    plt.ylim([0.5, M])
    plt.xlabel('True MOS')
    plt.ylabel('Predicted MOS')
    plt.title('Utt level LCC= {:.4f}, SRCC= {:.4f}, MSE= {:.4f}, KTAU= {:.4f}'.format(LCC, SRCC, MSE, KTAU))
    plt.show()
    plt.savefig(os.path.join(save_dir, dataset_name + "_" + mode + "_" + name, f'{ep}_utt_scatter_plot_utt.png'),
                dpi=150)
    plt.close()

    # system level scores
    MSE = np.mean((true_sys_mean_scores - predict_sys_mean_scores) ** 2)
    LCC = np.corrcoef(true_sys_mean_scores, predict_sys_mean_scores)[0][1]
    SRCC = scipy.stats.spearmanr(true_sys_mean_scores, predict_sys_mean_scores)[0]
    KTAU = scipy.stats.kendalltau(true_sys_mean_scores, predict_sys_mean_scores)[0]
    ep_scores += [MSE, LCC, SRCC, KTAU]
    print("[SYSTEM] {} MSE: {:.3f}, LCC: {:.3f}, SRCC: {:.3f}, KTAU: {:.3f}".format(name, float(MSE), float(LCC),
                                                                                    float(SRCC), float(KTAU)))

    # plotting utterance-level scatter plot
    M = np.max([np.max(predict_sys_mean_scores), 5])
    plt.figure(3)
    plt.scatter(true_sys_mean_scores, predict_sys_mean_scores, s=15, color='b', marker='o', edgecolors='b')
    plt.xlim([0.5, M])
    plt.ylim([0.5, M])
    plt.xlabel('True MOS')
    plt.ylabel('Predicted MOS')
    plt.title('Sys level LCC= {:.4f}, SRCC= {:.4f}, MSE= {:.4f}, KTAU= {:.4f}'.format(LCC, SRCC, MSE, KTAU))
    plt.show()
    plt.savefig(os.path.join(save_dir, dataset_name + "_" + mode + "_" + name, f'{ep}_sys_scatter_plot_utt.png'),
                dpi=150)
    plt.close()

    if return_posterior_scores:
        post_scores = np.array(post_scores)
        return ep_scores, post_scores
    else:
        return ep_scores, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="vcc2018")
    parser.add_argument("--data_dir", type=str, default="data/vcc2018")
    parser.add_argument("--exp_dir", type=str, default="exp")
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--ep", type=str, default=None, help="If not specified, evaluate all ckpts.")
    parser.add_argument("--start_ep", type=int, default=0, help="Epoch to start evaluation")
    parser.add_argument("--mode", type=str, required=True, choices=["mean_net", "all_listeners", "mean_listener"],
                        help="Inference mode.")
    args = parser.parse_args()

    # define dir
    save_dir = os.path.join(args.exp_dir, args.tag)
    os.makedirs(os.path.join(save_dir, args.dataset_name + "_" + args.mode + "_valid"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, args.dataset_name + "_" + args.mode + "_test"), exist_ok=True)

    # read config
    if args.config is not None:
        print("[Warning] You would probably use the existing config in the exp folder")
        config_path = args.config
    else:
        config_path = os.path.join(save_dir, "config.yml")
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # read dataset (batch size is always 1 to avoid padding)
    valid_set = get_dataset(args.dataset_name, args.data_dir, "valid")
    test_set = get_dataset(args.dataset_name, args.data_dir, "test")
    valid_loader = get_dataloader(valid_set, batch_size=1, num_workers=1, shuffle=False)
    test_loader = get_dataloader(test_set, batch_size=1, num_workers=1, shuffle=False)
    print("[Info] Number of validation samples: {}".format(len(valid_set)))
    print("[Info] Number of testing samples: {}".format(len(test_set)))

    # define model
    if config["model"] == "MBNet":
        model = MBNet(config).to(device)
    elif config["model"] == "SLDNet":
        model = SLDNet(config).to(device)
    else:
        raise NotImplementedError
    print("[Info] Model parameters: {}".format(model.get_num_params()))

    # either perform inference on one ep (specified by args.ep) or all ep in expdir
    if args.ep is not None:
        all_ckpts = [os.path.join(save_dir, f"model-{args.ep}.pt")]
    else:
        # get all ckpts
        all_ckpts = find_files(save_dir, "model-*.pt")

    # loop through all ckpts
    for model_path in all_ckpts:
        ep = os.path.basename(model_path).split(".")[0].split("-")[1]
        if int(ep) < args.start_ep:
            continue
        print("=================================================")
        print(f"[Info] Evaluating ep {ep}")
        model.load_state_dict(torch.load(model_path), strict=False)
        model.eval()

        # returning posterior score was for analyzing listener embedding, but not useful so just ignore it
        valid_result, valid_posterior_scores = inference(args.mode, model, ep, valid_loader, valid_set.systems,
                                                         save_dir, "valid", args.dataset_name,
                                                         return_posterior_scores=False)
        test_result, test_posterior_scores = inference(args.mode, model, ep, test_loader, test_set.systems,
                                                       save_dir, "test", args.dataset_name,
                                                       return_posterior_scores=False)

        save_results(ep, valid_result, test_result,
                     os.path.join(save_dir, args.dataset_name + "_" + args.mode + ".csv"))


if __name__ == "__main__":
    main()
