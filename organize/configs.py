from typing import List, Optional, Dict, Callable
from pathlib import Path
import xarray as xr
from dataclasses import field

from exceptiongroup import suppress
from pydantic import Field, BaseModel

from rf_torch.parameters import Noise_Params
from polarspike.Overview import Recording
from location.x_array import x_y_and_scale
from rf_torch.parameters import Loadable_Model
from pydantic import ConfigDict
from typing import ClassVar, Annotated, Any
from pydantic import BeforeValidator, PlainSerializer
from numpydantic import NDArray
import inspect
from typing import Type, Any, get_type_hints
from functools import partial
import json
from rf_analysis import calculate_quality
import warnings
from combined_analysis.reduce_all import sta_2d_cov_collapse, circular_reduction
import numpy as np
from stats.summary import calculate_stats


# %%
def validate_xr(v: Any) -> xr.DataArray:
    if isinstance(v, dict):
        return xr.DataArray.from_dict(v)
    return v


def serialize_xr(v: xr.DataArray) -> Dict:
    return v.to_dict()


# Custom type for xarray validation and serialization in Pydantic
XrDataArray = Annotated[
    xr.DataArray,
    BeforeValidator(validate_xr),  # 1. Turn Dict -> Array on load
    PlainSerializer(serialize_xr, return_type=dict),  # 2. Turn Array -> Dict on save
]


# %% Classes


class Recording_Config(Loadable_Model):
    """
    Recording object for sta-analysis session.
    Parameters:
    ----------
    output_folder (str): Folder to save recording configuration.
    JSON_FILENAME (ClassVar[str]): Filename for saving/loading configuration.
    model_config (ConfigDict): Pydantic configuration dictionary.
    root_path (Path): Root directory of the recording session.
    channel_names (List[str]): Names of the stimulus channels.
    channel_colours (List[str]): Colours associated with each channel.
    nr_channels (int): Number of stimulus channels.
    channel_configs (Dict[str, Noise_Stimulus_Config]): Configurations for each noise stimulus channel.
    nr_cells (int): Number of cells in the recording (derived from overview).
    """

    output_folder: str | Path = Field(default="")
    JSON_FILENAME: ClassVar[str] = "recording_config.json"
    model_config = ConfigDict(frozen=True)
    root_path: Path
    channel_names: List[str] = field(default_factory=list)
    channel_colours: List[str] = field(default_factory=list)
    nr_channels: int = field(default=0)
    channel_configs: Dict[str, "Noise_Stimulus_Config"] = field(default_factory=dict)
    nr_cells: int = field(default=0)

    def model_post_init(self, __context):
        object.__setattr__(self, "output_folder", self.root_path / self.output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        object.__setattr__(self, "nr_cells", self.overview.nr_cells)

    @property
    def overview(self) -> Recording:
        """
        Loads the overview recording from the specified root path.
        Returns
        -------
        Recording
            The loaded overview recording object.
        """
        return Recording.load(self.root_path / "overview")

    def add_channel(self, stimulus_id: int, name: str, colour: str):
        """
        Parameters
        ----------
        stimulus_id (str): Identifier for the noise stimulus.
        name (str): Name of the stimulus.
        colour (str): Colour associated with the stimulus.

        """
        # get stimulus_name from overview

        self.channel_names.append(name)
        self.channel_colours.append(colour)
        stimulus_name = self.overview.stimulus_df.query(
            f"stimulus_index=={stimulus_id}"
        )["stimulus_name"].values[0]
        channel_path = self.root_path / f"{stimulus_name}_idx_{stimulus_id}"
        # check all folders with ending cell_idx inside channel_path
        self.channel_configs[name] = Noise_Stimulus_Config(
            stimulus_name=stimulus_name, root_path=channel_path
        )
        object.__setattr__(self, "nr_channels", len(self.channel_names))


class Noise_Stimulus_Config(Loadable_Model):
    """
    Configuration for a noise stimulus used in spike-triggered average (STA) analysis.
    Parameters
    ----------
    stimulus_name (str): Name of the noise stimulus.
    root_path (Path): Root directory in which the cell folders are stored.
    image_shape (xr.DataArray): Shape of the stimulus image (x, y).
    post_spike_bins (int): Number of time bins after a spike to consider.
    dt_ms (float): Time resolution of the sta in milliseconds.
    stimulus_repeats (int): Number of repeats of the stimulus.
    nr_triggers (int): Number of triggers per repeat.
    pixel_size (float): Size of each pixel in micrometers.
    good_cells (Optional[np.ndarray]): Array of indices of good cells to include in analysis.
    sta_file (str): Filename of the STA data file.
    """

    JSON_FILENAME: ClassVar[str] = "noise_stimulus_config.json"
    model_config = ConfigDict(arbitrary_types_allowed=True)
    stimulus_name: str
    root_path: Path
    # Provide defaults for Pydantic so they are not required at init, and fill in __post_init__
    output_folder: Path | None = Field(default=None)
    image_shape: XrDataArray = Field(default=None)
    post_spike_bins: int | None = Field(default=None)
    dt_ms: float | None = Field(default=None)
    total_sta_len: int | None = Field(default=None)
    stimulus_repeats: int | None = Field(default=None)
    nr_triggers: int | None = Field(default=None)
    pixel_size: float | None = Field(default=None)
    good_cells: Optional[NDArray] = Field(default=None)
    sta_file: str = Field(default="kernel.npy")

    def model_post_init(self, __context):
        # try getting sta_shape

        noise_params = Noise_Params.load_from_root_json(self.root_path)
        object.__setattr__(
            self, "image_shape", x_y_and_scale(noise_params.x_len, noise_params.y_len)
        )
        object.__setattr__(self, "post_spike_bins", noise_params.post_spike_len)
        object.__setattr__(self, "dt_ms", noise_params.target_ms)
        object.__setattr__(self, "stimulus_repeats", noise_params.stim_repeats)
        object.__setattr__(self, "nr_triggers", noise_params.trigger_per_repeat)
        object.__setattr__(self, "pixel_size", noise_params.noise_pixel_size)
        object.__setattr__(self, "total_sta_len", noise_params.total_sta_len)


class Collapse_2d_Config(Loadable_Model):
    """
    Configuration for circular reduction of spike-triggered averages (STAs).
    Parameters
    ----------
    recording_config (Recording_Config): Configuration of the recording session.
    cut_size_um (xr.DataArray): Size of the cutout region in micrometers
        around each cell for STA extraction.
    degree_bins (int): Number of angular bins for circular reduction.
    max_cut_size_px (xr.DataArray): Maximum cut size in pixels across channels (Calculated from cut_size_um).
    max_half_cut_size_px (xr.DataArray): Half of the maximum cut size in pixels (Calculated from max_cut_size_px).
    extended_cut_size_px (xr.DataArray): Extended cut size in pixels for padding (Calculated from cut_size_px and max_half_cut_size_px).
    max_radius_px (Optional[int]): Maximum radius in pixels for circular reduction. Calculated if not provided.
    sta_to_2d_method (Callable): Method to convert STA to 2D representation. Defaults to variance over time.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    recording_config: Recording_Config
    cut_size_um: XrDataArray
    threshold: float = Field(default=20.0)
    max_radius_px: Dict[str, XrDataArray] = Field(default_factory=dict)
    sta_to_2d_method: str = Field(
        default="covariance", description="Method to convert STA to 2D representation"
    )
    cut_size_px: Dict[str, XrDataArray] = Field(default_factory=dict)
    half_cut_size_px: Dict[str, XrDataArray] = Field(default_factory=dict)
    extended_cut_size_px: Dict[str, XrDataArray] = Field(default_factory=dict)
    border_buffer_px: Dict[str, XrDataArray] = Field(default_factory=dict)
    JSON_FILENAME: ClassVar[str] = "collapse_2d_config.json"
    output_folder: Path | None = Field(default=None)

    def model_post_init(self, __context):
        max_cut_size_px: Dict[str, XrDataArray] = {}
        max_half_cut_size_px: Dict[str, XrDataArray] = {}
        extended_cut_size_px: Dict[str, XrDataArray] = {}
        border_buffer_px: Dict[str, XrDataArray] = {}
        max_radius_px: Dict[str, int] = {}

        for channel_name in self.recording_config.channel_configs:
            cut_size_px = (
                (
                        self.cut_size_um
                        / self.recording_config.channel_configs[channel_name].pixel_size
                )
                .round()
                .astype(int)
            )

            half_cut = (cut_size_px / 2).astype(int)

            max_cut_size_px[channel_name] = cut_size_px
            max_half_cut_size_px[channel_name] = half_cut
            extended_cut_size_px[channel_name] = (
                    self.recording_config.channel_configs[channel_name].image_shape
                    + cut_size_px
            )
            border_buffer_px[channel_name] = (
                                                     extended_cut_size_px[channel_name]
                                                     - self.recording_config.channel_configs[channel_name].image_shape
                                             ) // 2
            max_radius_px[channel_name] = (
                    np.sqrt(cut_size_px.max() ** 2 + cut_size_px.max() ** 2) // 2
            )

        object.__setattr__(self, "cut_size_px", max_cut_size_px)
        object.__setattr__(self, "half_cut_size_px", max_half_cut_size_px)
        object.__setattr__(self, "extended_cut_size_px", extended_cut_size_px)
        object.__setattr__(self, "border_buffer_px", border_buffer_px)
        object.__setattr__(self, "max_radius_px", max_radius_px)


class Circular_Reduction_Config(Loadable_Model):
    """
    Configuration for circular reduction of spike-triggered averages (STAs).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    JSON_FILENAME: ClassVar[str] = "circular_reduction_config.json"
    output_folder: Path | None = Field(default=None)
    degree_bins: int = Field(default=10)


class Analysis_Pipeline:
    """
    Wrapper that collects analysis configuration objects and executes
    analysis functions by injecting dependencies based on type hints.
    """

    CONFIG_REGISTRY = {
        "Recording_Config": Recording_Config,
        "Noise_Stimulus_Config": Noise_Stimulus_Config,
        "Collapse_2d_Config": Collapse_2d_Config,
        "Circular_Reduction_Config": Circular_Reduction_Config,
        # Add any other config classes you create
    }

    CALLABLES_REGISTRY = {
        # Add any analysis functions you create here for loading
        "calculate_rf_quality": calculate_quality.calculate_rf_quality,
        "sta_2d_cov_collapse": sta_2d_cov_collapse,
        "circular_reduction": circular_reduction,
        "calculate_stats": calculate_stats,
    }

    def __init__(
            self,
            analysis_folder: str,
            recording_config: Recording_Config,
            other_configs: List[Any] = None,
            tasks: Optional[List[Callable[..., Any]]] = None,
    ):
        if other_configs is None:
            other_configs = []
        self.configs: Dict[Type[Any], Any] = {type(recording_config): recording_config}
        # We store configs in a dictionary mapping the Class Type to the Instance
        for config in other_configs:
            self.configs[type(config)] = config
        self.config_filenames = [
            config.JSON_FILENAME for config in self.configs.values()
        ]

        self.scheduled_functions: List[Callable[..., Any]] = []
        self._scheduled_function_names: List[str] = []
        # 1. Store the analysis folder name
        self.analysis_folder = analysis_folder
        # 2. Get the root output folder from the primary config
        self.root_output_folder = recording_config.output_folder
        # 3. Define the full, immutable pipeline output path
        self.pipeline_output_path = self.root_output_folder / analysis_folder
        # Create the analysis folder if it doesn't exist
        for task in tasks or []:
            self.schedule([task])
        self._finished_tasks: List[str] = []
        try:
            self.pipeline_output_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            warnings.warn(
                "Analysis folder already exists. Using existing analysis folder."
            )
            try:
                self.load_existing()
            except FileNotFoundError:
                warnings.warn("No existing pipeline state found. Starting fresh.")

    def add_config(self, config: Loadable_Model) -> None:
        """
        Adds or replaces a configuration object in the pipeline.
        """
        config_type = type(config)
        self.configs[config_type] = config
        if config.JSON_FILENAME not in self.config_filenames:
            self.config_filenames.append(config.JSON_FILENAME)

    def schedule(self, funcs: List[Callable[..., Any]]) -> None:
        """
        Schedules analysis functions by injecting dependencies
        based on type hints.
        """
        for func in funcs:
            sig = inspect.signature(func)
            kwargs = {}
            print(f"\n-> Scheduling: **{func.__name__}**")
            resolved_types = get_type_hints(func, globalns=globals())

            for name, param in sig.parameters.items():
                param_type = resolved_types.get(name, Any)
                # Skip 'self' or 'cls' for method inspection (if needed, though Pydantic often handles this)
                if name in ("self", "cls", "analysis_folder"):
                    continue

                # 1. Dependency Injection (Config Found)
                if param_type in self.configs:
                    kwargs[name] = self.configs[param_type]
                    print(
                        f"   [INJECTED] Parameter '{name}' with config type {param_type.__name__}"
                    )

                # 2. Handle Default Values (Parameter is Optional)
                elif param.default is not param.empty:
                    # If it has a default, we rely on Python to use it.
                    print(
                        f"   [OPTIONAL] Parameter '{name}' is optional (default will be used)."
                    )
                    continue

                # 3. Required Argument Missing (Failure Case)
                else:
                    # This argument is required, but it's not a known config type,
                    # and it wasn't provided (since 'schedule' doesn't take kwargs).
                    raise ValueError(
                        f"💥 Pipeline Error: Function **{func.__name__}** requires parameter '{name}' "
                        f"of type **{param_type.__name__}**, but no matching configuration "
                        "was loaded in the pipeline and the parameter has no default value."
                    )
            if func.__name__ in self._scheduled_function_names:
                warnings.warn(
                    f"Warning: Function '{func.__name__}' is already scheduled. Skipping duplicate scheduling."
                )
                continue
            # save the function name
            self._scheduled_function_names.append(func.__name__)
            # add analysis_folder to kwargs
            kwargs["analysis_folder"] = self.analysis_folder

            # We need to carry over any pipeline dependencies
            scheduled_func = partial(func, **kwargs)
            if hasattr(func, "_pipeline_dependencies"):
                setattr(
                    scheduled_func,
                    "_pipeline_dependencies",
                    getattr(func, "_pipeline_dependencies"),
                )
            self.scheduled_functions.append(scheduled_func)

    def save(self):
        """
        Saves all configuration objects to their respective JSON files.
        """
        for config in self.configs.values():
            config.save_json(self.root_output_folder / self.analysis_folder)

        config_types = [type(config).__name__ for config in self.configs.values()]

        with open(
                self.root_output_folder / self.analysis_folder / "tasks_configs.json", "w"
        ) as f:
            json.dump(
                {
                    "tasks": self._scheduled_function_names,
                    "finished_tasks": self._finished_tasks,
                    "configs": self.config_filenames,
                    "config_types": config_types,
                },
                f,
                indent=4,
            )

        print(f"Pipeline state saved to {self.configs[Recording_Config].output_folder}")

    def load_existing(self):
        task_json_path = (
                self.root_output_folder / self.analysis_folder / "tasks_configs.json"
        )
        configs = []
        with open(task_json_path, "r") as f:
            task_dict = json.load(f)
        analysis_folder = task_json_path.parent.name
        for config_types in task_dict["config_types"]:
            configs.append(
                Analysis_Pipeline.CONFIG_REGISTRY[config_types].load_from_root_json(
                    task_json_path.parent
                )
            )
        tasks = []
        for task_name in task_dict["tasks"]:
            try:
                tasks.append(Analysis_Pipeline.CALLABLES_REGISTRY[task_name])
            except KeyError:
                raise KeyError(
                    f"💥 Pipeline Error: Task function '{task_name}' not found in CALLABLES_REGISTRY."
                )
        # update self
        self.analysis_folder = analysis_folder
        self.configs = {type(configs[0]): configs[0]}
        for config in configs[1:]:
            self.configs[type(config)] = config
        self.config_filenames = [
            config.JSON_FILENAME for config in self.configs.values()
        ]
        self.scheduled_functions = []
        self._scheduled_function_names = []
        for task in tasks:
            self.schedule([task])
        self._finished_tasks = task_dict.get("finished_tasks", [])
        print(
            f"Loaded Recording with following channels {self.configs[Recording_Config].channel_names}:"
        )
        print(f"Finished_tasks: {self._finished_tasks}")

    @classmethod
    def load(cls, task_json_path: Path) -> "Analysis_Pipeline":
        # check which *_config.json files are in analysis_folder
        configs = []
        with open(task_json_path, "r") as f:
            task_dict = json.load(f)
        analysis_folder = task_json_path.parent.name
        for config_types in task_dict["config_types"]:
            configs.append(
                Analysis_Pipeline.CONFIG_REGISTRY[config_types].load_from_root_json(
                    task_json_path.parent
                )
            )
        tasks = []
        for task_name in task_dict["tasks"]:
            try:
                tasks.append(Analysis_Pipeline.CALLABLES_REGISTRY[task_name])
            except KeyError:
                raise KeyError(
                    f"💥 Pipeline Error: Task function '{task_name}' not found in CALLABLES_REGISTRY."
                )

        pipeline = cls(
            analysis_folder=analysis_folder,
            recording_config=configs[0],
            other_configs=configs[1:],
            tasks=tasks,
        )
        pipeline._finished_tasks = task_dict.get("finished_tasks", [])
        print(
            f"Loaded Recording with following channels {pipeline.configs[Recording_Config].channel_names}:"
        )
        print(f"Finished_tasks: {pipeline._finished_tasks}")
        return pipeline

    def run(self, task_names: Optional[List[str]] = None):
        """
        Executes all scheduled analysis functions in order, optionally filtering by name.
        """
        task_filter = set(task_names) if task_names is not None else None

        for func in self.scheduled_functions:
            func_name = getattr(getattr(func, "func", func), "__name__", repr(func))
            try:
                requirements = getattr(func, "_pipeline_dependencies, []")
            except AttributeError:
                requirements = []
            missing_dependencies = [
                task for task in requirements if task not in self._finished_tasks
            ]
            if missing_dependencies:
                raise RuntimeError(
                    f"Cannot run task '{func_name}' because it is missing dependencies: {missing_dependencies}"
                )
            if task_filter is not None and func_name not in task_filter:
                print(f"\n=== Skipping (not requested): {func_name} ===")
                continue

            if func_name in self._finished_tasks:
                print(f"\n=== Skipping (already done): {func_name} ===")
                continue

            print(f"\n=== Running: {func_name} ===")
            func()
            self._finished_tasks.append(func_name)
            # save after each task
            self.save()
