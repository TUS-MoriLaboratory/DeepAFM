
from typing import Optional, Dict, Type
from dataclasses import dataclass, field, asdict, fields

import yaml, argparse, os, datetime

from configs.nn_config import TrainConfig
from configs.nn_config import BaseModelConfig, ViTClassifierConfig, ViTAutoEncoderConfig, ViTMultiTaskAutoEncoderConfig
from configs.evaluation_config import UMAPConfig, StructureDependencyConfig
from configs.data_config import DataConfig
from configs.system_config import SystemConfig
from configs.data_dir_config import DataDirConfig

# AFM image generation configs
from afm_image_generation.configs.afm_image_parameter import AFMImageFixedParams, AFMImageRandomRange, AFMDataSamplingParams, AFMBrownianMotionParams, AFMScanParams
from afm_image_generation.configs.afm_job_config import AFMGenerationJobConfig

# Model config mapping
MODEL_CONFIG_MAP: Dict[str, Type[BaseModelConfig]] = {
    "vit_classifier": ViTClassifierConfig,
    "vit_autoencoder": ViTAutoEncoderConfig,
    "vit_multitask_ae": ViTMultiTaskAutoEncoderConfig,
}

@dataclass
class AFMConfig:
    fixed: AFMImageFixedParams = field(default_factory=AFMImageFixedParams)
    random: AFMImageRandomRange = field(default_factory=AFMImageRandomRange)
    sampling: AFMDataSamplingParams = field(default_factory=AFMDataSamplingParams)
    brownian: AFMBrownianMotionParams = field(default_factory=AFMBrownianMotionParams)
    scan: AFMScanParams = field(default_factory=AFMScanParams)
    job: AFMGenerationJobConfig = field(default_factory=AFMGenerationJobConfig)

@dataclass
class ExperimentConfig:
    exp_id: Optional[str] = None
    run_name: Optional[str] = None
    description: Optional[str] = None
    protein_of_project: str = "SecYAEG_NDs"
    scan_direction: str = "y"  # "x" or "y"

    afm: AFMConfig = field(default_factory=AFMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    data_dir: Optional[DataDirConfig] = None
    system: SystemConfig = field(default_factory=SystemConfig)
    model: BaseModelConfig = field(default_factory=ViTClassifierConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    umap: UMAPConfig = field(default_factory=UMAPConfig)
    struct_dep: StructureDependencyConfig = field(default_factory=StructureDependencyConfig)

    def __post_init__(self):

        # Set experiment ID
        # exp_id for pytest
        test_exp_id = os.environ.get("TEST_EXP_ID")
        if test_exp_id:
            self.exp_id = test_exp_id

        # if id is not set, create one based on timestamp
        if self.exp_id is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.exp_id = f"exp_{timestamp}"

        if self.run_name is None:
            self.run_name = self.exp_id
        
        # update run_name in system
        self.system.run_name = self.exp_id
        self.system.scan_direction = self.scan_direction

        if self.data_dir is None:
            self.data_dir = DataDirConfig(
                root_dir=self.system.project_root,
                exp_id=self.exp_id,
                run_name=self.run_name,
                protein_of_project=self.protein_of_project,
                scan_direction=self.scan_direction,
            )

        # Set AFM fixed params paths
        self.afm.fixed.pdb_path = f"{self.data_dir.md_dir}/initial.pdb"
        self.afm.fixed.dcd_path = f"{self.data_dir.md_dir}/rst_total.dcd"

        # Set scan direction
        self.afm.scan.scan_direction = self.scan_direction

        # Set AFM job save directory
        self.afm.job.save_dir = f"{self.data_dir.afm_data_dir}"
    # --- Conversion utilities ---
    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):

        def smart_init(data_cls, data_dict):
            # get fields that the data class __init__ can accept
            valid_keys = {f.name for f in fields(data_cls) if f.init}
            filtered_dict = {k: v for k, v in data_dict.items() if k in valid_keys}
            return data_cls(**filtered_dict)

        model_data = d.get("model", {})
        model_name = model_data.get("name", "vit_classifier")
        model_cls = MODEL_CONFIG_MAP.get(model_name, ViTClassifierConfig)

        afm_data = d.get("afm", {})
        rigid_data = d.get("rigid_body", {})

        system_dict = d.get("system", {})
        system_config = smart_init(SystemConfig, system_dict)
        
        data_dict = d.get("data_dir", {}).copy() 
        
        if "run_name" not in data_dict:
            data_dict["run_name"] = system_config.run_name

        return cls(
            exp_id=d.get("exp_id"),
            description=d.get("description"),
            protein_of_project=d.get("protein_of_project", "SecYAEG_NDs"),

            afm=AFMConfig(
                fixed=smart_init(AFMImageFixedParams, afm_data.get("fixed", {})),
                random=smart_init(AFMImageRandomRange, afm_data.get("random", {})),
                sampling=smart_init(AFMDataSamplingParams, afm_data.get("sampling", {})),
                brownian=smart_init(AFMBrownianMotionParams, afm_data.get("brownian", {})),
                scan=smart_init(AFMScanParams, afm_data.get("scan", {})),
                job=smart_init(AFMGenerationJobConfig, afm_data.get("job", {})),
            ),
            
            data=smart_init(DataConfig, d.get("data", {})),
            data_dir=smart_init(DataDirConfig, data_dict), 
            system=smart_init(SystemConfig, d.get("system", {})),
            model=smart_init(model_cls, model_data),
            train=smart_init(TrainConfig, d.get("train", {})),
            umap=smart_init(UMAPConfig, d.get("umap", {})),
            struct_dep=smart_init(StructureDependencyConfig, d.get("structure_dependency", {})),
        )

    # --- YAML I/O ---
    def save_yaml(self, path: str = None):
        if path is None:
            path = f"runs/{self.system.run_name}/config.yaml"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def load_yaml(cls, path: str, override_root_dir: str = None):
        with open(path) as f:
            d = yaml.safe_load(f)

        cfg = cls.from_dict(d)

        if override_root_dir is not None:
            print(f"[Config] Overriding root_dir: {cfg.system.project_root} -> {override_root_dir}")
            cfg.system.project_root = override_root_dir
            
            cfg.data_dir = DataDirConfig(
                root_dir=override_root_dir,
                run_name=cfg.system.run_name,
                exp_id=cfg.exp_id,
                protein_of_project=cfg.protein_of_project,
            )

            if hasattr(cfg, "unsupervised"):
                cfg.unsupervised.protein_of_project = cfg.protein_of_project
                cfg.unsupervised.update_paths()

        return cfg
