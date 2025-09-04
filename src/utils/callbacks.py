from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from datasets import Dataset
from accelerate.utils import gather_object
from trl.trainer.callbacks import _generate_completions
from utils.metrics import compute_translation_metrics

class NMTCallback(TrainerCallback):
    def __init__(
        self,
        trainer,
        eval_dataset: Dataset = None,
        generation_config=None,
        eval_steps: int = 50,
        comet_metric=None,
        language_pairs=None,
    ):
        super().__init__()
        self.trainer           = trainer
        if eval_dataset is None:
            self.eval_dataset  = trainer.eval_dataset
        else:
            self.eval_dataset  = eval_dataset
        # if "prompt" not in self.eval_dataset.column_names or "instruction" not in self.eval_dataset.column_names:
        #     raise ValueError("The evaluation dataset must contain a 'prompt' or 'instruction' column.")
        self.tokenizer         = trainer.tokenizer
        self.generation_config = generation_config
        self.eval_steps        = eval_steps
        self.comet_metric      = comet_metric
        self.language_pairs    = language_pairs

    def _run_eval(self, args: TrainingArguments, state: TrainerState):
        # tokenizer = kwargs["processing_class"]
        self.tokenizer.padding_side = "left"
        accelerator = self.trainer.accelerator
        model = getattr(self.trainer, "ref_model", None)
        # At this point, there are two cases where `ref_model` is None:
        # 1. The method doesn't require a reference model.
        # 2. The method uses a reference model, but `ref_model` is set to None.
        #    This occurs when using PEFT, where the reference model can be obtained by simply disabling the model's adapter.
        #    In theory, we should disable the adapter here, but since it's zero-initialized at the start of training,
        #    the model behaves identically with or without the adapter.
        #    Therefore, there's no need to explicitly disable it at this point.
        if model is None:
            model = self.trainer.model_wrapped

        prompt_column = "prompt" if "prompt" in self.eval_dataset.column_names else "instruction"
        prompts = self.eval_dataset[prompt_column]
        # Use accelerator to shard prompts across GPUs
        with accelerator.split_between_processes(prompts) as shard_prompts:
            raw_preds = _generate_completions(
                prompts=shard_prompts,
                model=model,
                tokenizer=self.tokenizer,
                accelerator=accelerator,
                generation_config=self.generation_config,
                batch_size=args.per_device_eval_batch_size,
            )

        # Gather predictions from all processes
        predictions = gather_object(raw_preds)

        target_key = (
            "completion" if "completion" in self.eval_dataset.column_names
            else "target" if "target" in self.eval_dataset.column_names
            else "chosen"
        )
        references = [[ref.replace(self.tokenizer.eos_token, "")] for ref in self.eval_dataset[target_key]]

        # bleu = bleu_calc.corpus_score(predictions, references).score
        # chrf = chrf_calc.corpus_score(predictions, references).score

        batch = self.tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt"
        )

        # Batch-decode, skipping all special tokens
        sources = self.tokenizer.batch_decode(
            batch["input_ids"],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        metrics = compute_translation_metrics(
            preds=predictions,
            refs=references,
            language_pairs=self.language_pairs,
            prompts=sources,
            comet_metric=self.comet_metric,
        )
        # Logging
        if self.trainer.accelerator.is_main_process:
            # Log to HF and optionally W&B
            # Log with eval_ prefix
            self.trainer.log({f'eval_{k}': v for k, v in metrics.items()})
            if "wandb" in args.report_to:
                    import wandb
                    if wandb.run is not None:
                        table = wandb.Table(columns=["prompt","pred","ref"])
                        for p,pr,rf in zip(self.eval_dataset[prompt_column], predictions, [r[0] for r in references]):
                            table.add_data(p, pr, rf)
                        wandb.log({"eval/translations": table})

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % self.eval_steps != 0:
            return control

        self._run_eval(args, state)
        return control

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self._run_eval(args, state)
        return control


