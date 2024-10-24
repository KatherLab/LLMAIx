import json
import os
import pandas as pd
import zipfile
import re
from io import StringIO
import ast

def load_csvs_from_zip(zip_path):
    """
    Load CSV files from a ZIP archive into pandas DataFrames.
    Returns two DataFrames: one for LLM output and one for metrics.
    
    Parameters:
    zip_path (str): Path to the ZIP file
    
    Returns:
    tuple: (llm_output_df, metrics_df)
    """
    llm_output_df = None
    metrics_df = None
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # List all files in the ZIP
        file_list = zip_ref.namelist()
        
        # Find the relevant CSV files
        llm_file = next((f for f in file_list if f.startswith('llm-output-')), None)
        metrics_file = next((f for f in file_list if f.startswith('metrics_')), None)
        
        # Load LLM output CSV
        if llm_file:
            with zip_ref.open(llm_file) as f:
                content = f.read().decode('utf-8')
                llm_output_df = pd.read_csv(StringIO(content))
                
        # Load metrics CSV
        if metrics_file:
            with zip_ref.open(metrics_file) as f:
                content = f.read().decode('utf-8')
                metrics_df = pd.read_csv(StringIO(content))
    
    return llm_output_df, metrics_df

# -------------------------------------------------------------------------------------------------------------- #
# Edit this part to your needs
# Generation Metadata
model_name_experiment = "LLaMA 3.1 8B Instruct Q5_K_M"
model_name_huggingface = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
model_quant = "q5km"
model_params = 3 # in BILLION parameters

# Add all experiments for one model. Can contain multiple metrics per experiment
experiments = [
    {
        "filepath": "/Users/fabianwolf/Downloads/metrics_LLaMA3.18BInstructQ5-K-M_202410141018_3j5i_Kvh0NQ.zip",
        "name": "exp1",
        "metrics": [
            {
                "name": "accuracy", # how the metric should be called in the leaderboard
                "csv_key": "accuracy", # how the metric is called in the metrics csv
                "labels": "all" # implement later TODO
            }
        ]
    }
]


output_dir = "/Users/fabianwolf/Download/leaderboard_output"

# -------------------------------------------------------------------------------------------------------------- #


os.makedirs(output_dir, exist_ok=True)


# Generate the following json:

# {
#     "config": {
#         "model_dtype": "q8", 
#         "model_params": 3,
#         "model_name": "bartowski/Meta-Llama-3.1-3B-Instruct-GGUF",
#         "model_sha": "",
#         "model_type": "LLM"
#     },
#     "results": {
#         "exp1": {
#             "acc": 0.9
#         },
#         "exp2": {
#             "acc_norm": 0.95
#         }
#     }
# }

leaderboard_json = {
    "config": {
        "model_dtype": model_quant,
        "model_params": model_params,
        "model_name": model_name_huggingface,
        "model_sha": "",
        "model_type": "LLM"
    },
    "results": {
    }
}

for experiment in experiments:
    llm_df, metrics_df = load_csvs_from_zip(experiment["filepath"])

    metadata = ast.literal_eval(llm_df.iloc[0]['metadata'])
    assert metadata['llm_processing']['model_name'] == model_name_experiment

    metric_dict = {}

    for metric in experiment["metrics"]:
        csv_key = metric["csv_key"]
        # get row where id is macro_scores
        row = metrics_df[metrics_df["id"] == "macro_scores"].iloc[0]
        # get all column names which have the csv_key as suffix in their name after a $ sign, take all values of those in the row and average them
        metric_columns = [col for col in row.index if csv_key in col and "$" in col]

        numeric_values = pd.to_numeric(row[metric_columns], errors='coerce')
        metric_dict[metric["name"]] = numeric_values.mean()

        # also add all individual values of the metric_columns to the metric_dict as metric['name']_namebeforethedollarsign
        for col in metric_columns:
            metric_dict[metric["name"] + "_" +  col.split("$")[0]] = float(row[col])
        

    leaderboard_json["results"][experiment["name"]] = metric_dict

# save json to file 
with open(os.path.join(output_dir, f"leaderboard_{model_name_huggingface.replace('/', '_')}.json"), "w") as f:
    json.dump(leaderboard_json, f)