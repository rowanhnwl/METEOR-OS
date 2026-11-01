import subprocess
import os
import shutil
import argparse
import json

from pyprojroot import here
import sys
sys.path.insert(0, str(here()))

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
config = json.load(open(vars(parser.parse_args())["config"], "r"))

data_path = config["data_path"]
dict_name = os.path.join("infer/misc", config["dict_name"])
task_num = config["task_num"]
local_batch_size = config["local_batch_size"]
only_polar = config["only_polar"]
conf_size = config["conf_size"]
full_dataset_task_schema_path = config["schema_path"]
lnp_encoder_attention_heads = config["lnp_encoder_attention_heads"]
lnp_encoder_ffn_embed_dim = config["lnp_encoder_ffn_embed_dim"]
lnp_encoder_embed_dim = config["lnp_encoder_embed_dim"]
lnp_encoder_layers = config["lnp_encoder_layers"]
loss_func = config["loss_func"]
eval_batch_size = config["eval_batch_size"]
vlib_path = config["vlib_path"]
deploy_model_master_dir = config["deploy_model_master_dir"]
deploy_weight_filename = config["deploy_weight_filename"]
output_dir = config["output_dir"]
output_append_string = config["output_append_string"]
include_excipients = config["include_excipients"]

deploy_data_dirs = [os.path.join(vlib_path, dir_name) for dir_name in os.listdir(vlib_path)]
deploy_model_dirs = [os.path.join(deploy_model_master_dir, dir_name) for dir_name in os.listdir(deploy_model_master_dir)]

# enumerate through models
for deploy_model_dir in deploy_model_dirs:
    deploy_weight_path = os.path.join(deploy_model_dir, deploy_weight_filename)
    model_name = os.path.basename(deploy_model_dir.rstrip('/'))

    # enumerate through all the deploy data directories
    for deploy_data_dir in deploy_data_dirs:
        for filename in reversed(os.listdir(deploy_data_dir)):
            # Check if the file is a "lmdb" folder
            if filename.endswith('lmdb'):
                dataset_lmdb_path = os.path.join(deploy_data_dir, filename) 

                # create output directories' and files' paths
                deploy_infer_output_dir = os.path.join(output_dir, os.path.basename(deploy_data_dir) + output_append_string)
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
                task_name = os.path.dirname(dataset_lmdb_path)
                eval_weight_path = deploy_weight_path
                eval_results_path = full_output_dir

                infer_args = f"python unimol/infer_np.py --user-dir unimol {data_path} --task-name {task_name} --valid-subset infer \
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
                    "
                
                if include_excipients: # If this is flagged, the schema should include the proper dictionary values for component_types
                    infer_args += "--include-excipients"

                subprocess.run(infer_args,
                    shell=True)