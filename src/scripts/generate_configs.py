import json
import os
import itertools
from typing import Dict, List, Any, Tuple
import argparse


def load_base_config(config_path: str) -> Dict[str, Any]:
    """Load a base configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def generate_sft_configs(
        base_config: Dict[str, Any],
        train_files: List[str],
        eval_files: List[str],
        learning_rates: List[float],
        num_epochs: List[int],
        batch_sizes: List[int],
        schedules: List[str],
        warmup_ratios: List[float],
        weight_decays: List[float],
        output_dir: str
) -> List[str]:
    """Generate SFT configurations with different hyperparameters."""

    configs_generated = []
    # Extract model name for naming
    model_key = base_config["model_name"].split("/")[-1].lower().replace("-", "").replace(".", "_")

    # Use simple version counters for config names
    version_counter = 1
    for train_file, eval_file, lr, epochs, batch_size, schedule, warmup_ratio, weight_decay in itertools.product(
            train_files, eval_files, learning_rates, num_epochs, batch_sizes, schedules, warmup_ratios, weight_decays):
        config = base_config.copy()

        # Update hyperparameters
        config["train_file"] = train_file
        config["eval_file"] = eval_file
        config["trainer"]["learning_rate"] = lr
        config["trainer"]["num_train_epochs"] = epochs
        config["trainer"]["per_device_train_batch_size"] = batch_size
        config["trainer"]["per_device_eval_batch_size"] = max(1, batch_size // 2)
        config["trainer"]["lr_scheduler_type"] = schedule
        config["trainer"]["warmup_ratio"] = warmup_ratio
        config["trainer"]["weight_decay"] = weight_decay

        # Generate simple versioned config name with skommarkhos prefix
        config_name = f"skommarkhos_{model_key}_sft_v{version_counter}"
        version_counter += 1

        # Update output directory and hub model id
        config["output_dir"] = f"/workspace/models/{config_name}"
        config["trainer"]["hub_model_id"] = f"igorktech/{config_name.replace('_', '-')}"

        # Save config
        config_path = os.path.join(output_dir, f"{config_name}.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        configs_generated.append(config_path)
        print(f"Generated: {config_path}")

    return configs_generated


def generate_arpo_configs(
        base_config: Dict[str, Any],
        file_pairs: List[Tuple[str, str]],
        learning_rates: List[float],
        num_epochs: List[int],
        batch_sizes: List[int],
        loss_types: List[str],
        cpo_alphas: List[float],
        relax_cofficients_1: List[float],
        relax_cofficients_2: List[float],
        output_dir: str,
        sft_model_paths: List[str] = None
) -> List[str]:
    """Generate ARPO configurations with different hyperparameters."""

    configs_generated = []

    # Extract model name for naming
    model_key = base_config["model_name"].split("/")[-1].lower().replace("-", "").replace(".", "_")

    # If SFT model paths provided, use them as base models
    base_models = sft_model_paths if sft_model_paths else [base_config["model_name"]]

    # Use simple version counters for ARPO config names
    version_counter = 1

    for file_pair, base_model, lr, epochs, batch_size, loss_type, cpo_alpha, relax_cofficient_1, relax_cofficient_2 in itertools.product(
            file_pairs, base_models, learning_rates, num_epochs, batch_sizes, loss_types, cpo_alphas,
            relax_cofficients_1, relax_cofficients_2
    ):
        train_file, eval_file = file_pair
        config = base_config.copy()

        # Update model name if using SFT checkpoint
        if base_model != base_config["model_name"]:
            config["model_name"] = base_model
            model_suffix = "sft"
        else:
            model_suffix = "base"

        # Update hyperparameters
        config["train_file"] = train_file
        config["eval_file"] = eval_file
        config["trainer"]["learning_rate"] = lr
        config["trainer"]["num_train_epochs"] = epochs
        config["trainer"]["per_device_train_batch_size"] = batch_size
        config["trainer"]["per_device_eval_batch_size"] = max(1, batch_size // 4)
        config["trainer"]["loss_type"] = loss_type
        config["trainer"]["cpo_alpha"] = cpo_alpha
        config["trainer"]["relax_cofficient_1"] = relax_cofficient_1
        config["trainer"]["relax_cofficient_2"] = relax_cofficient_2

        # Generate short versioned name with skommarkhos prefix
        config_name = f"skommarkhos_{model_key}_{model_suffix}_arpo_a{version_counter}"
        version_counter += 1

        # Update output directory and hub model id
        config["output_dir"] = f"/workspace/models/{config_name}"
        config["trainer"]["hub_model_id"] = f"igorktech/{config_name.replace('_', '-')}"

        # Save config
        config_path = os.path.join(output_dir, f"{config_name}.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        configs_generated.append(config_path)
        print(f"Generated: {config_path}")

    return configs_generated


def main():
    parser = argparse.ArgumentParser(description="Generate training configurations")
    parser.add_argument("--base-sft-config", required=True, help="Base SFT configuration file")
    parser.add_argument("--base-arpo-config", required=True, help="Base ARPO configuration file")
    parser.add_argument("--output-dir", default="configs/generated", help="Output directory for configs")
    parser.add_argument("--sft-only", action="store_true", help="Generate only SFT configs")
    parser.add_argument("--arpo-only", action="store_true", help="Generate only ARPO configs")
    parser.add_argument("--sft-models", nargs="*", help="SFT model paths for ARPO training")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Hyperparameter ranges
    sft_train_files = ["data/task2/processed/train_sft.json",
                       "data/task2/processed/train_sft_extended.json"]
    sft_eval_files = ["data/task2/processed/validation_sft.json"]
    sft_learning_rates = [5e-5, 1e-4]
    sft_num_epochs = [1, 2]
    sft_batch_sizes = [16]
    sft_schedules = ["inverse_sqrt"]
    sft_warmup_ratios = [0.01]
    sft_weight_decays = [0.01]

    # ARPO hyperparameters
    arpo_file_pairs = [("data/task2/processed/train_arpo_X-ALMA-13B-Group4.json",
                        "data/task2/processed/validation_arpo_X-ALMA-13B-Group4.json"),
                       ("data/task2/processed/train_arpo_X-ALMA-13B-Group4_extended.json",
                        "data/task2/processed/validation_arpo_X-ALMA-13B-Group4.json"),
                       ("data/task2/processed/train_arpo_gpt-4o-mini.json",
                        "data/task2/processed/validation_arpo_gpt-4o-mini.json"),
                       ("data/task2/processed/train_arpo_gpt-4o-mini_extended.json",
                        "data/task2/processed/validation_arpo_gpt-4o-mini.json")
                       ]
    arpo_learning_rates = [5e-7]
    arpo_num_epochs = [1]
    arpo_batch_sizes = [8]
    arpo_loss_types = ["sigmoid"]
    arpo_cpo_alphas = [1.0]
    arpo_relax_cofficients_1 = [0.9]
    arpo_relax_cofficients_2 = [0.4, 1.0, 1.5]
    sft_configs = []
    arpo_configs = []

    # Generate SFT configs
    if not args.arpo_only:
        print("Generating SFT configurations...")
        base_sft_config = load_base_config(args.base_sft_config)
        sft_configs = generate_sft_configs(
            base_sft_config,
            sft_train_files,
            sft_eval_files,
            sft_learning_rates,
            sft_num_epochs,
            sft_batch_sizes,
            sft_schedules,
            sft_warmup_ratios,
            sft_weight_decays,
            args.output_dir
        )
        print(f"Generated {len(sft_configs)} SFT configurations")

    # Generate ARPO configs
    if not args.sft_only:
        print("\nGenerating ARPO configurations...")
        base_arpo_config = load_base_config(args.base_arpo_config)
        arpo_configs = generate_arpo_configs(
            base_arpo_config,
            arpo_file_pairs,
            arpo_learning_rates,
            arpo_num_epochs,
            arpo_batch_sizes,
            arpo_loss_types,
            arpo_cpo_alphas,
            arpo_relax_cofficients_1,
            arpo_relax_cofficients_2,
            args.output_dir,
            args.sft_models
        )
        print(f"Generated {len(arpo_configs)} ARPO configurations")

    # Save config lists for bash scripts
    if sft_configs:
        with open(os.path.join(args.output_dir, "sft_configs.txt"), 'w') as f:
            f.write('\n'.join(sft_configs))

    if arpo_configs:
        with open(os.path.join(args.output_dir, "arpo_configs.txt"), 'w') as f:
            f.write('\n'.join(arpo_configs))

    print(f"\nTotal configurations generated:")
    print(f"  SFT: {len(sft_configs)}")
    print(f"  ARPO: {len(arpo_configs)}")
    print(f"  Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
