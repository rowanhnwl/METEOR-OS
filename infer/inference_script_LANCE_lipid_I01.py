import subprocess
import os
import shutil

data_path='./'  # replace with your data path
MASTER_PORT=10086
n_gpu=1
dict_name='dict.txt'
task_num=1
dropout=0.1
warmup=0.06
local_batch_size=32
only_polar=0 # -1 all h; 0 no h
conf_size=11
seed=0

full_dataset_task_schema_path = "task_schemas/in_house_lnp_master_schema_NPratio_AOvolratio.json"

lnp_encoder_attention_heads = 8
lnp_encoder_ffn_embed_dim = 256
lnp_encoder_embed_dim = 256
lnp_encoder_layers = 8
loss_func = 'np_finetune_contrastive'


# key eval params
eval_batch_size = 256


deploy_data_dirs = ["in_house_lnp_library"]


# list the dir path of saved model weights here
deploy_model_dirs = [
    "./weights/save_demo_in_house_ED09262023_fig3dDepolyTrain_fold_V0_lnp_np_finetune_contrastive-bs64-lr0.0001-lnpmodparams8-256-256-8-trainrat1-ep200-pat20-metricvalid_spearmanr_coeff-cagrad0.2-percentnoise0.1-labelmargin0.01-seed1_exp19",
    # "./weights/save_demo_in_house_ED09262023_fig3dDepolyTrain_fold_V1_lnp_np_finetune_contrastive-bs64-lr0.0001-lnpmodparams8-256-256-8-trainrat1-ep200-pat20-metricvalid_spearmanr_coeff-cagrad0.2-percentnoise0.1-labelmargin0.01-seed1_exp19",
    # "./weights/save_demo_in_house_ED09262023_fig3dDepolyTrain_fold_V2_lnp_np_finetune_contrastive-bs64-lr0.0001-lnpmodparams8-256-256-8-trainrat1-ep200-pat20-metricvalid_spearmanr_coeff-cagrad0.2-percentnoise0.1-labelmargin0.01-seed1_exp19",
    # "./weights/save_demo_in_house_ED09262023_fig3dDepolyTrain_fold_V3_lnp_np_finetune_contrastive-bs64-lr0.0001-lnpmodparams8-256-256-8-trainrat1-ep200-pat20-metricvalid_spearmanr_coeff-cagrad0.2-percentnoise0.1-labelmargin0.01-seed1_exp19",
    # "./weights/save_demo_in_house_ED09262023_fig3dDepolyTrain_fold_V4_lnp_np_finetune_contrastive-bs64-lr0.0001-lnpmodparams8-256-256-8-trainrat1-ep200-pat20-metricvalid_spearmanr_coeff-cagrad0.2-percentnoise0.1-labelmargin0.01-seed1_exp19",
]

deploy_weight_filename = 'checkpoint_best.pt'

output_append_string = "_infer_results_withclsrep_OS"

# inference results will be outputted at location: dataset_output_dir>model_name>output_filename 

# enumerate through models
for deploy_model_dir in deploy_model_dirs:
    deploy_weight_path = os.path.join(deploy_model_dir, deploy_weight_filename)
    model_name = os.path.basename(deploy_model_dir.rstrip('/'))
    # enumerate through all the deploy data directories
    for deploy_data_dir in deploy_data_dirs:
        for filename in reversed(os.listdir(deploy_data_dir)):
            # Check if the file is a "lmdb" folder
            if filename.endswith('lmdb'):
                dataset_lmdb_path = os.path.join(deploy_data_dir, filename) # part2_in_house_lnp_DC24_luc_22638_22981_lmdb

                # create output directories' and files' paths, e.g. part2_in_house_lnp_DC24_luc_22638_22981_infer_results
                deploy_infer_output_dir = deploy_data_dir + output_append_string
                dataset_output_dir = os.path.join(deploy_infer_output_dir, filename.replace("lmdb", "infer_results"))

                full_output_dir = os.path.join(dataset_output_dir, model_name)

                # make infer_results output dir
                os.makedirs(full_output_dir, exist_ok=True)

                # check inference is already done by checking if the output file already exists
                infer_alr_done = False
                for output_filename in os.listdir(full_output_dir):
                    if output_filename.endswith(".out.pkl"):
                        infer_alr_done = True
                        print("inference already done, output_filename: ", output_filename)
                        break
                if infer_alr_done:
                    continue

                # set up key folder paths for inference
                task_name = dataset_lmdb_path
                eval_weight_path = deploy_weight_path
                eval_results_path = full_output_dir

                subprocess.run(f"python ../unimol/infer_np.py --user-dir ../unimol {data_path} --task-name {task_name} --valid-subset infer \
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
                    --full-dataset-task-schema-path {full_dataset_task_schema_path} \
                    --load-full-np-model --concat-datasets \
                    --output-cls-rep",
                    shell=True)

