import subprocess
import os
import shutil
import argparse
import json
import itertools
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))

MASTER_PORT=10086

def get_exp_name(*args):
    exp_name = f""
    exp_name += f"{args[0]}"
    exp_name += f"_{args[1]}"
    exp_name += f"-bs{args[2]}"
    exp_name += f"-lr{args[3]}"
    exp_name += f"-lnpmodparams{args[4]}-{args[5]}-{args[6]}-{args[7]}"
    exp_name += f"-trainrat{args[8]}"
    exp_name += f"-ep{args[9]}"
    exp_name += f"-pat{args[10]}"
    exp_name += f"-metric{args[11]}"
    exp_name += f"-cagrad{args[12]}"
    exp_name += f"-percentnoise{args[13]}"
    exp_name += f"-labelmargin{args[14]}"
    exp_name += f"-seed{args[15]}_OS"

    return exp_name

def train_comet(hps, root_task_name, metric, config):

    root_save_dir = './train/TEMP/' + config["weights_save_name"]
    root_tmp_save_dir = './train/TEMP/tmp_' + config["weights_save_name"]

    for hp_config in hps:
        (lnp_encoder_attention_heads, 
        lnp_encoder_ffn_embed_dim, 
        lnp_encoder_embed_dim,
        lnp_encoder_layers, 
        warmup, 
        dropout, 
        epoch, 
        lr, 
        batch_size, 
        loss_sample_dropout,
        loss_func, 
        patience, 
        subdataset_patience, 
        epoch_to_freeze_molecule_encoder, 
        cagrad_c,
        percent_noise, 
        contrast_margin_coeff, 
        percent_noise_type,
        train_data_ratio, 
        seed) = hp_config
        
        task_name = root_task_name

        # set up batch_size
        local_batch_size=batch_size
        update_freq=batch_size / local_batch_size

        # compensate max_epoch with loss_sample_dropout
        max_epoch = int(epoch // (1 - loss_sample_dropout))

        # unique experiment name (identifier)
        exp_name = get_exp_name(
            config["exp_name_header"],
            loss_func,
            batch_size,
            lr,
            lnp_encoder_layers,
            lnp_encoder_embed_dim,
            lnp_encoder_ffn_embed_dim,
            lnp_encoder_attention_heads,
            train_data_ratio,
            max_epoch,
            patience,
            metric,
            cagrad_c,
            percent_noise,
            contrast_margin_coeff,
            seed
        )
        # config["exp_name_header"] + f'_{loss_func}-bs{batch_size}-lr{lr}-lnpmodparams{lnp_encoder_layers}-{lnp_encoder_embed_dim}-{lnp_encoder_ffn_embed_dim}-{lnp_encoder_attention_heads}-trainrat{train_data_ratio}-ep{max_epoch}-pat{patience}-metric{metric}-cagrad{cagrad_c}-percentnoise{percent_noise}-labelmargin{contrast_margin_coeff}-seed{seed}_OS'  
        
        print("task_name: ", task_name)
        if config["save_all_model_weights"]:
            save_path = '/save_' + exp_name
            save_dir = os.path.join(root_save_dir, save_path)
            tmp_save_dir = os.path.join(root_tmp_save_dir, save_path)
        else:
            save_dir = root_save_dir
            tmp_save_dir = os.path.join(root_tmp_save_dir, task_name)

        # tensorboard log path
        log_path = 'log_' + exp_name
        logdir=os.path.join("./train/logs/tmp/", log_path) 

        # infer output path
        results_folder = 'infer_' + exp_name
        eval_results_path = os.path.join("./train/" + config["infer_results_save_name"], results_folder) 
        eval_weight_path = os.path.join(save_dir, 'checkpoint_best.pt')

        # Check if this experiment is already done, if so, skip it
        if os.path.exists(eval_results_path):
            print("Infer output dir exists. Skipping experiment: ", eval_results_path)
            continue
        elif os.path.exists(logdir): # if logdir exists, delete it as the previous exp run is not done yet
            print("Tensorboard log dir exists but inference not done. Rerunning exp and removing log dir: ", logdir)
            shutil.rmtree(logdir)

        if os.path.exists(save_dir) and os.path.isdir(save_dir):
            shutil.rmtree(save_dir)

        if os.path.exists(tmp_save_dir) and os.path.isdir(tmp_save_dir):
            shutil.rmtree(tmp_save_dir)

        print("Running training script for: ", logdir)

        data_path = config["data_path"]
        conf_size = config["conf_size"]
        dict_name = config["dict_name"]
        weight_path = config.get("weight_path")
        finetune_from_model_arg = (
            f"--finetune-from-model {data_path + weight_path}"
            if weight_path
            else ""
        )
        only_polar = config["only_polar"]
        task_num = config["task_num"]
        schema_path = config["full_dataset_task_schema_path"]

        subprocess.run(f"python ./unimol/train_np.py {data_path} --task-name {task_name} --user-dir ./unimol --train-subset train --valid-subset valid \
            --conf-size {conf_size} \
            --num-workers 8 --ddp-backend=c10d \
            --dict-name {dict_name} \
            --task mol_np_finetune --loss {loss_func} --arch np_unimol  \
            --classification-head-name {task_name} --num-classes {task_num} \
            --optimizer adam --adam-betas '(0.9, 0.99)' --adam-eps 1e-6 --clip-norm 1.0 \
            --lr-scheduler polynomial_decay --lr {lr} --warmup-ratio {warmup} --max-epoch {max_epoch} --batch-size {local_batch_size} --pooler-dropout {dropout} \
            --loss-sample-dropout {loss_sample_dropout} \
            --update-freq {update_freq} --seed {seed} \
            --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
            --log-interval 100 --log-format simple \
            --validate-interval 1 --keep-last-epochs 10 \
            {finetune_from_model_arg} \
            --best-checkpoint-metric {metric} --patience {patience} \
            --maximize-best-checkpoint-metric \
            --save-dir {save_dir} --tmp-save-dir {tmp_save_dir} --only-polar {only_polar} \
            --tensorboard-logdir {logdir} \
            --full-dataset-task-schema-path {schema_path} \
            --multitask-reg --cagrad-c {cagrad_c} \
            --epoch-to-freeze-molecule-encoder {epoch_to_freeze_molecule_encoder} \
            --concat-datasets \
            --train-data-ratio {train_data_ratio} \
            --lnp-encoder-layers {lnp_encoder_layers} --lnp-encoder-embed-dim {lnp_encoder_embed_dim} --lnp-encoder-ffn-embed-dim {lnp_encoder_ffn_embed_dim} --lnp-encoder-attention-heads {lnp_encoder_attention_heads} \
            --noise-augment-percent --percent-noise {percent_noise} --percent-noise-type {percent_noise_type} \
            --contrast-margin-coeff {contrast_margin_coeff} \
            --include-excipients", 
            shell=True)

        # eval params
        eval_batch_size = 32

        # tensorboard log path
        results_folder = 'infer_' + exp_name
        eval_results_path = os.path.join("./weights/" + config["infer_results_save_name"], results_folder) 
        eval_weight_path = os.path.join(save_dir, 'checkpoint_best.pt')
        
        print("\n\n\nEVAL RESULTS PATH: " + eval_results_path + "\n\n\n")

        subprocess.run(f"python ./unimol/infer_np.py --user-dir ./unimol {data_path} --task-name {task_name} --valid-subset test \
            --num-workers 8 --ddp-backend=c10d --batch-size {eval_batch_size} \
            --task mol_np_finetune --loss {loss_func} --arch np_unimol \
            --classification-head-name {task_name} --num-classes {task_num} \
            --dict-name {dict_name} --conf-size {conf_size} \
            --only-polar {only_polar}  \
            --path {eval_weight_path}  \
            --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
            --log-interval 50 --log-format simple \
            --results-path {eval_results_path} \
            --lnp-encoder-layers {lnp_encoder_layers} --lnp-encoder-embed-dim {lnp_encoder_embed_dim} --lnp-encoder-ffn-embed-dim {lnp_encoder_ffn_embed_dim} --lnp-encoder-attention-heads {lnp_encoder_attention_heads} \
            --full-dataset-task-schema-path {schema_path} \
            --load-full-np-model --concat-datasets \
            --include-excipients",
            shell=True)

        os.replace(eval_weight_path, os.path.join(eval_results_path, "weights.pt"))

    shutil.rmtree("./train/TEMP")

if __name__ == "__main__":

    # Get the config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to the config file')
    args = vars(parser.parse_args())
    config_path = args["config"]
    config = json.load(open(config_path, 'r'))

    n_gpu = config["n_gpu"]
    root_task_name = config["root_task_name"]

    metric="valid_spearmanr_coeff"

    lnp_encoder_attention_heads_list = config["lnp_encoder_attention_heads_list"]
    lnp_encoder_ffn_embed_dim_list = config["lnp_encoder_ffn_embed_dim_list"]
    lnp_encoder_embed_dim_list = config["lnp_encoder_embed_dim_list"]
    lnp_encoder_layers_list = config["lnp_encoder_layers_list"]
    warmups = config["warmups"]
    dropouts = config["dropouts"]
    epoch_list = config["epoch_list"]
    lrs = config["lrs"]
    batch_sizes = config["batch_sizes"]
    loss_sample_dropouts = config["loss_sample_dropouts"]
    loss_funcs = config["loss_funcs"]
    patiences = config["patiences"]
    subdataset_patiences = config["subdataset_patiences"]
    epoch_to_freeze_molecule_encoder_list = config["epoch_to_freeze_molecule_encoder_list"]
    cagrad_cs = config["cagrad_cs"]
    percent_noises = config["percent_noises"]
    contrast_margin_coeffs = config["contrast_margin_coeffs"]
    percent_noise_types = config["percent_noise_types"]
    save_all_model_weights = config["save_all_model_weights"]
    train_data_ratios = config["train_data_ratios"]
    seeds = list(range(1, config["seeds"] + 1))

    hp_configs = itertools.product(
        lnp_encoder_attention_heads_list,
        lnp_encoder_ffn_embed_dim_list,
        lnp_encoder_embed_dim_list,
        lnp_encoder_layers_list,
        warmups,
        dropouts,
        epoch_list,
        lrs,
        batch_sizes,
        loss_sample_dropouts,
        loss_funcs,
        patiences,
        subdataset_patiences,
        epoch_to_freeze_molecule_encoder_list,
        cagrad_cs,
        percent_noises,
        contrast_margin_coeffs,
        percent_noise_types,
        train_data_ratios,
        seeds
    )

    train_comet(hp_configs, root_task_name, metric, config)
