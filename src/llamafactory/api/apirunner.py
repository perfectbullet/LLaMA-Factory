import os
from copy import deepcopy
from subprocess import Popen, TimeoutExpired
from typing import TYPE_CHECKING, Any, Dict, Generator, Optional

from transformers.trainer import TRAINING_ARGS_NAME

from ..extras.constants import LLAMABOARD_CONFIG, PEFT_METHODS, TRAINING_STAGES
from ..extras.misc import is_gpu_or_npu_available, torch_gc

from ..webui.common import DEFAULT_CACHE_DIR, DEFAULT_CONFIG_DIR, QUANTIZATION_BITS, get_save_dir, load_config
from ..webui.locales import ALERTS, LOCALES
from ..webui.utils import abort_process, gen_cmd, get_eval_results, get_trainer_info, load_args, save_args, save_cmd, \
    get_trainer_info_api


class ApiRunner:
    def __init__(self, demo_mode: bool = False) -> None:
        self.demo_mode = demo_mode
        """ Resume """
        self.trainer: Optional[Popen] = None
        self.do_train = True
        self.running_data: Dict[str, Any] = None
        """ State """
        self.aborted = False
        self.running = False

    def set_abort(self) -> None:
        self.aborted = True
        if self.trainer is not None:
            abort_process(self.trainer.pid)

    def _initialize(self, data: Dict[str, Any], do_train: bool, from_preview: bool) -> str:

        lang, model_name, model_path = data.get("lang"), data.get("model_name"), data.get("model_path")

        dataset = data.get("dataset") if do_train else data.get("dataset")

        if self.running:
            return ALERTS["err_conflict"][lang]

        if not model_name:
            return ALERTS["err_no_model"][lang]

        if not model_path:
            return ALERTS["err_no_path"][lang]

        if not dataset:
            return ALERTS["err_no_dataset"][lang]

        if not from_preview and self.demo_mode:
            return ALERTS["err_demo"][lang]

        if do_train:
            if not data.get("output_dir"):
                return ALERTS["err_no_output_dir"][lang]

            stage = data.get("stage")
            if stage == "ppo" and not data.get("reward_model"):
                return ALERTS["err_no_reward_model"][lang]
        else:
            if not data.get("eval_output_dir"):
                return ALERTS["err_no_output_dir"][lang]
        return ""

    def _finalize(self, lang: str, finish_info: str) -> str:
        finish_info = ALERTS["info_aborted"][lang] if self.aborted else finish_info
        self.trainer = None
        self.aborted = False
        self.running = False
        self.running_data = None
        torch_gc()
        return finish_info

    def _parse_train_args(self, data: Dict[str, Any]) -> Dict[str, Any]:

        model_name, finetuning_type = data.get("model_name"), data.get("finetuning_type")
        user_config = load_config()

        if data.get("quantization_bit") in QUANTIZATION_BITS:
            quantization_bit = int(data.get("quantization_bit"))
        else:
            quantization_bit = None

        args = dict(
            stage=data.get("stage"),
            do_train=True,
            model_name_or_path=data.get("model_path"),
            cache_dir=user_config.get("cache_dir", None),
            preprocessing_num_workers=16,
            finetuning_type=finetuning_type,
            quantization_bit=quantization_bit,
            quantization_method=data.get("quantization_method"),
            template=data.get("template"),
            rope_scaling=data.get("rope_scaling") if data.get("rope_scaling") in ["linear", "dynamic"] else None,
            flash_attn="fa2" if data.get("booster") == "flashattn2" else "auto",
            use_unsloth=(data.get("booster") == "unsloth"),
            visual_inputs=data.get("visual_inputs"),
            dataset_dir=data.get("dataset_dir"),
            dataset=",".join(data.get("dataset")),
            cutoff_len=data.get("cutoff_len"),
            learning_rate=float(data.get("learning_rate")),
            num_train_epochs=float(data.get("num_train_epochs")),
            max_samples=int(data.get("max_samples")),
            per_device_train_batch_size=data.get("batch_size"),
            gradient_accumulation_steps=data.get("gradient_accumulation_steps"),
            lr_scheduler_type=data.get("lr_scheduler_type"),
            max_grad_norm=float(data.get("max_grad_norm")),
            logging_steps=data.get("logging_steps"),
            save_steps=data.get("save_steps"),
            warmup_steps=data.get("warmup_steps"),
            neftune_noise_alpha=data.get("neftune_alpha") or None,
            optim=data.get("optim"),
            resize_vocab=data.get("resize_vocab"),
            packing=data.get("packing"),
            upcast_layernorm=data.get("upcast_layernorm"),
            use_llama_pro=data.get("use_llama_pro"),
            shift_attn=data.get("shift_attn"),
            report_to="all" if data.get("report_to") else "none",
            use_galore=data.get("use_galore"),
            use_badam=data.get("use_badam"),
            output_dir=get_save_dir(model_name, finetuning_type, data.get("output_dir")),
            fp16=(data.get("compute_type") == "fp16"),
            bf16=(data.get("compute_type") == "bf16"),
            pure_bf16=(data.get("compute_type") == "pure_bf16"),
            plot_loss=True,
            ddp_timeout=180000000,
            include_num_input_tokens_seen=True,
        )

        # checkpoints
        if data.get("checkpoint_path"):
            if finetuning_type in PEFT_METHODS:  # list
                args["adapter_name_or_path"] = ",".join(
                    [get_save_dir(model_name, finetuning_type, adapter) for adapter in data.get("checkpoint_path")]
                )
            else:  # str
                args["model_name_or_path"] = get_save_dir(model_name, finetuning_type, data.get("checkpoint_path"))

        # freeze config
        if args["finetuning_type"] == "freeze":
            args["freeze_trainable_layers"] = data.get("train.freeze_trainable_layers")
            args["freeze_trainable_modules"] = data.get("train.freeze_trainable_modules")
            args["freeze_extra_modules"] = data.get("train.freeze_extra_modules") or None

        # lora config
        if args["finetuning_type"] == "lora":
            args["lora_rank"] = data.get("train.lora_rank")
            args["lora_alpha"] = data.get("train.lora_alpha")
            args["lora_dropout"] = data.get("train.lora_dropout")
            args["loraplus_lr_ratio"] = data.get("train.loraplus_lr_ratio") or None
            args["create_new_adapter"] = data.get("train.create_new_adapter")
            args["use_rslora"] = data.get("train.use_rslora")
            args["use_dora"] = data.get("train.use_dora")
            args["pissa_init"] = data.get("train.use_pissa")
            args["pissa_convert"] = data.get("train.use_pissa")
            args["lora_target"] = data.get("train.lora_target") or "all"
            args["additional_target"] = data.get("train.additional_target") or None

            if args["use_llama_pro"]:
                args["num_layer_trainable"] = data.get("train.num_layer_trainable")

        # rlhf config
        if args["stage"] == "ppo":
            if finetuning_type in PEFT_METHODS:
                args["reward_model"] = ",".join(
                    [get_save_dir(model_name, finetuning_type, adapter) for adapter in data.get("train.reward_model")]
                )
            else:
                args["reward_model"] = get_save_dir(model_name, finetuning_type, data.get("train.reward_model"))

            args["reward_model_type"] = "lora" if finetuning_type == "lora" else "full"
            args["ppo_score_norm"] = data.get("train.ppo_score_norm")
            args["ppo_whiten_rewards"] = data.get("train.ppo_whiten_rewards")
            args["top_k"] = 0
            args["top_p"] = 0.9
        elif args["stage"] in ["dpo", "kto"]:
            args["pref_beta"] = data.get("train.pref_beta")
            args["pref_ftx"] = data.get("train.pref_ftx")
            args["pref_loss"] = data.get("train.pref_loss")

        # galore config
        if args["use_galore"]:
            args["galore_rank"] = data.get("train.galore_rank")
            args["galore_update_interval"] = data.get("train.galore_update_interval")
            args["galore_scale"] = data.get("train.galore_scale")
            args["galore_target"] = data.get("train.galore_target")

        # badam config
        if args["use_badam"]:
            args["badam_mode"] = data.get("train.badam_mode")
            args["badam_switch_mode"] = data.get("train.badam_switch_mode")
            args["badam_switch_interval"] = data.get("train.badam_switch_interval")
            args["badam_update_ratio"] = data.get("train.badam_update_ratio")

        # eval config
        if data.get("train.val_size") > 1e-6 and args["stage"] != "ppo":
            args["val_size"] = data.get("train.val_size")
            args["eval_strategy"] = "steps"
            args["eval_steps"] = args["save_steps"]
            args["per_device_eval_batch_size"] = args["per_device_train_batch_size"]

        # ds config
        if data.get("train.ds_stage") != "none":
            ds_stage = data.get("train.ds_stage")
            ds_offload = "offload_" if data.get("train.ds_offload") else ""
            args["deepspeed"] = os.path.join(DEFAULT_CACHE_DIR, "ds_z{}_{}config.json".format(ds_stage, ds_offload))

        return args

    def _parse_eval_args(self, data: Dict["Component", Any]) -> Dict[str, Any]:
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
        model_name, finetuning_type = data.get("model_name"), data.get("finetuning_type")
        user_config = load_config()

        if data.get("quantization_bit") in QUANTIZATION_BITS:
            quantization_bit = int(data.get("quantization_bit"))
        else:
            quantization_bit = None

        args = dict(
            stage="sft",
            model_name_or_path=data.get("model_path"),
            cache_dir=user_config.data.get("cache_dir", None),
            preprocessing_num_workers=16,
            finetuning_type=finetuning_type,
            quantization_bit=quantization_bit,
            quantization_method=data.get("quantization_method"),
            template=data.get("template"),
            rope_scaling=data.get("rope_scaling") if data.get("rope_scaling") in ["linear", "dynamic"] else None,
            flash_attn="fa2" if data.get("booster") == "flashattn2" else "auto",
            use_unsloth=(data.get("booster") == "unsloth"),
            visual_inputs=data.get("visual_inputs"),
            dataset_dir=data.get("eval.dataset_dir"),
            dataset=",".join(data.get("eval.dataset")),
            cutoff_len=data.get("eval.cutoff_len"),
            max_samples=int(data.get("eval.max_samples")),
            per_device_eval_batch_size=data.get("eval.batch_size"),
            predict_with_generate=True,
            max_new_tokens=data.get("eval.max_new_tokens"),
            top_p=data.get("eval.top_p"),
            temperature=data.get("eval.temperature"),
            output_dir=get_save_dir(model_name, finetuning_type, data.get("eval.output_dir")),
        )

        if data.get("eval.predict"):
            args["do_predict"] = True
        else:
            args["do_eval"] = True

        if data.get("checkpoint_path"):
            if finetuning_type in PEFT_METHODS:  # list
                args["adapter_name_or_path"] = ",".join(
                    [get_save_dir(model_name, finetuning_type, adapter) for adapter in data.get("checkpoint_path")]
                )
            else:  # str
                args["model_name_or_path"] = get_save_dir(model_name, finetuning_type, data.get("checkpoint_path"))

        return args

    def _preview(self, data: Dict["Component", Any], do_train: bool) -> Generator[Dict["Component", str], None, None]:
        output_box = self.manager.get_elem_by_id("{}.output_box".format("train" if do_train else "eval"))
        error = self._initialize(data, do_train, from_preview=True)
        if error:
            yield {output_box: error}
        else:
            args = self._parse_train_args(data) if do_train else self._parse_eval_args(data)
            yield {output_box: gen_cmd(args)}

    def _launch(self, data: Dict["Component", Any], do_train: bool) -> Generator[Dict["Component", Any], None, None]:
        output_box = self.manager.get_elem_by_id("{}.output_box".format("train" if do_train else "eval"))
        error = self._initialize(data, do_train, from_preview=False)
        if error:
            yield {output_box: error}
        else:
            #
            self.do_train, self.running_data = do_train, data
            args = self._parse_train_args(data) if do_train else self._parse_eval_args(data)

            os.makedirs(args["output_dir"], exist_ok=True)
            save_args(os.path.join(args["output_dir"], LLAMABOARD_CONFIG), self._form_config_dict(data))

            env = deepcopy(os.environ)
            env["LLAMABOARD_ENABLED"] = "1"
            env["LLAMABOARD_WORKDIR"] = args["output_dir"]
            if args.get("deepspeed", None) is not None:
                env["FORCE_TORCHRUN"] = "1"
            train_cmd = "llamafactory-cli train {}".format(save_cmd(args))
            self.trainer = Popen(train_cmd, env=env, shell=True)
            yield from self.monitor()

    def _form_config_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        config_dict = {}
        skip_ids = ["lang", "model_path", "output_dir", "config_path"]
        for elem, value in data.items():
            elem_id = self.running_data.get(elem)
            if elem_id not in skip_ids:
                config_dict[elem_id] = value

        return config_dict

    def preview_train(self, data):
        yield from self._preview(data, do_train=True)

    def preview_eval(self, data):
        yield from self._preview(data, do_train=False)

    def run_train(self, data):
        # 开始训练入口哦
        yield from self._launch(data, do_train=True)

    def run_eval(self, data):
        yield from self._launch(data, do_train=False)

    def monitor(self):
        self.aborted = False
        self.running = True
        lang, model_name, finetuning_type = self.running_data.get("lang"), self.running_data.get("model_name"), self.running_data.get("finetuning_type")
        output_dir = self.running_data.get("output_dir")
        output_path = get_save_dir(model_name, finetuning_type, output_dir)
        # 没有gradio如何使用呢？
        # output_box = self.manager.get_elem_by_id("{}.output_box".format("train" if self.do_train else "eval"))
        # progress_bar = self.manager.get_elem_by_id("{}.progress_bar".format("train" if self.do_train else "eval"))
        # loss_viewer = self.manager.get_elem_by_id("train.loss_viewer") if self.do_train else None
        while self.trainer is not None:
            if self.aborted:
                yield {
                    'output_box': ALERTS["info_aborting"][lang],
                    'progress_bar': '',
                }
            else:
                running_log, running_progress, running_loss = get_trainer_info_api(output_path, self.do_train)
                return_dict = {
                    'output_box': running_log,
                    'progress_bar': running_progress,
                }
                if running_loss is not None:
                    return_dict['running_loss'] = running_loss
                yield return_dict
            try:
                self.trainer.wait(2)
                self.trainer = None
            except TimeoutExpired:
                continue
        if self.do_train:
            if os.path.exists(os.path.join(output_path, TRAINING_ARGS_NAME)):
                finish_info = ALERTS["info_finished"][lang]
            else:
                finish_info = ALERTS["err_failed"][lang]
        else:
            if os.path.exists(os.path.join(output_path, "all_results.json")):
                all_results_path = os.path.join(output_path, "all_results.json")
                finish_info = get_eval_results(
                    all_results_path
                )
            else:
                finish_info = ALERTS["err_failed"][lang]
        return_dict = {
            'output_box': self._finalize(lang, finish_info),
            'progress_bar': '',
        }
        yield return_dict

    def save_args(self, data):
        output_box = self.running_data.get("output_box")
        error = self._initialize(data, do_train=True, from_preview=True)
        if error:
            return {output_box: error}
        lang = data[self.running_data.get("lang")]
        config_path = data[self.running_data.get("config_path")]
        os.makedirs(DEFAULT_CONFIG_DIR, exist_ok=True)
        save_path = os.path.join(DEFAULT_CONFIG_DIR, config_path)
        save_args(save_path, self._form_config_dict(data))
        return {output_box: ALERTS["info_config_saved"][lang] + save_path}

    def load_args(self, lang: str, config_path: str):
        output_box = self.running_data.get("output_box")
        config_dict = load_args(os.path.join(DEFAULT_CONFIG_DIR, config_path))
        if config_dict is None:
            return {output_box: ALERTS["err_config_not_found"][lang]}
        output_dict: Dict[str, Any] = {output_box: ALERTS["info_config_loaded"][lang]}
        for elem_id, value in config_dict.items():
            output_dict[self.running_data.get(elem_id)] = value
        return output_dict

    def check_output_dir(self, lang: str, model_name: str, finetuning_type: str, output_dir: str):
        output_box = self.running_data.get("output_box")
        output_dict: Dict[str, Any] = {output_box: LOCALES["output_box"][lang]["value"]}
        if model_name and output_dir and os.path.isdir(get_save_dir(model_name, finetuning_type, output_dir)):

            output_dict[output_box] = ALERTS["warn_output_dir_exists"][lang]

            output_dir = get_save_dir(model_name, finetuning_type, output_dir)
            config_dict = load_args(os.path.join(output_dir, LLAMABOARD_CONFIG))  # load llamaboard config
            for elem_id, value in config_dict.items():
                output_dict[self.running_data.get(elem_id)] = value

        return output_dict
