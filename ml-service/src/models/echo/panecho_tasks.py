"""Task definitions for the pretrained PanEcho model.

PanEcho (Holste et al., JAMA 2025) has 40 output heads covering 39 reporting
tasks from one video clip. The table below mirrors the task metadata that
ships with the official release (``content/tasks.pkl``), stored as plain
Python so the service never has to unpickle remote files at start-up.

Each output head in the checkpoint is named ``<task_name>_head``. The order of
this table is the order of the heads in the original model.

Only ``EF`` is validated against EchoNet-Dynamic ground truth in this project.
All other tasks are exposed as *auxiliary* outputs and must not be fed into the
disease registry until they are validated on a labelled dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field

REGRESSION = "regression"
BINARY = "binary_classification"
MULTICLASS = "multi-class_classification"


@dataclass(frozen=True)
class PanEchoTask:
    name: str
    task_type: str
    class_names: tuple[str, ...] = field(default_factory=tuple)
    # Training-set mean used by PanEcho to initialise the head bias.
    # Only matters when a head is created without pretrained weights.
    mean: float = 0.0
    unit: str | None = None
    description: str = ""
    # For binary tasks: the label that the post-activation probability refers to.
    positive_label: str | None = None
    # PanEcho stores some binary heads with the "positive" class at index 0.
    # For those, probability(positive) = 1 - sigmoid(logit).
    invert_sigmoid: bool = False

    @property
    def num_outputs(self) -> int:
        return len(self.class_names) if self.task_type == MULTICLASS else 1


# fmt: off
PANECHO_TASKS: tuple[PanEchoTask, ...] = (
    PanEchoTask("pericardial-effusion", BINARY, ("mild_mod_severe", "none_trace"), 0.961859643488036, None, "Pericardial effusion", "mild_mod_severe", True),
    PanEchoTask("EF", REGRESSION, (), 59.002175258801685, "%", "Left ventricular ejection fraction"),
    PanEchoTask("GLS", REGRESSION, (), 18.556993691537958, "%", "Global longitudinal strain (actual value is -1 x output)"),
    PanEchoTask("LVEDV", REGRESSION, (), 115.5111406156421, "mL", "LV end-diastolic volume"),
    PanEchoTask("LVESV", REGRESSION, (), 49.583417494769854, "mL", "LV end-systolic volume"),
    PanEchoTask("LVSV", REGRESSION, (), 65.83874621632124, "mL", "LV stroke volume"),
    PanEchoTask("LVSize", MULTICLASS, ("Mildly Increased", "Moderately|Severely Increased", "Normal"), 1.7461797378240904, None, "LV size"),
    PanEchoTask("LVWallThickness-increased-any", BINARY, ("Increased", "Normal"), 0.5219670379275719, None, "Any LV hypertrophy (increased wall thickness)", "Increased", True),
    PanEchoTask("LVWallThickness-increased-modsev", BINARY, ("Moderately|severely increased", "Normal or mildly increased"), 0.9085002140512771, None, "Moderate or greater LV hypertrophy", "Moderately|severely increased", True),
    PanEchoTask("LVSystolicFunction", MULTICLASS, ("Mildly Decreased", "Moderately|Severely Decreased", "Normal|Hyperdynamic"), 1.7900815714390226, None, "LV systolic function"),
    PanEchoTask("LVWallMotionAbnormalities", BINARY, ("None", "Present"), 0.21668860816223634, None, "LV wall motion abnormalities", "Present"),
    PanEchoTask("IVSd", REGRESSION, (), 1.0687505374363675, "cm", "Interventricular septum thickness at diastole"),
    PanEchoTask("LVPWd", REGRESSION, (), 1.0329714537357577, "cm", "LV posterior wall thickness at diastole"),
    PanEchoTask("LVIDs", REGRESSION, (), 3.1036937225490657, "cm", "LV internal diameter at systole"),
    PanEchoTask("LVIDd", REGRESSION, (), 4.662377451202295, "cm", "LV internal diameter at diastole"),
    PanEchoTask("LVOTDiam", REGRESSION, (), 2.0567495134373415, "cm", "LV outflow tract diameter"),
    PanEchoTask("LVDiastolicFunction", MULTICLASS, ("Mild|Indeterminate", "Moderate|Severe", "Normal"), 1.2414207550193448, None, "LV diastolic function"),
    PanEchoTask("E|EAvg", REGRESSION, (), 10.804208154645387, None, "E/e' ratio"),
    PanEchoTask("RVSP", REGRESSION, (), 30.469686248382423, "mmHg", "RV systolic pressure"),
    PanEchoTask("RVSize", MULTICLASS, ("Mildly Increased", "Moderately|Severely Increased", "Normal"), 1.7230919397830087, None, "RV size"),
    PanEchoTask("RVSystolicFunction", BINARY, ("Decreased", "Normal"), 0.9085695482536549, None, "RV systolic function", "Decreased", True),
    PanEchoTask("RVIDd", REGRESSION, (), 3.162131545983516, "cm", "RV internal diameter at diastole"),
    PanEchoTask("TAPSE", REGRESSION, (), 2.1422161938799253, "cm", "Tricuspid annular plane systolic excursion"),
    PanEchoTask("RVSVel", REGRESSION, (), 12.794537628299329, "cm/s", "RV S' velocity"),
    PanEchoTask("LASize", MULTICLASS, ("Mildly Dilated", "Moderately|Severely Dilated", "Normal"), 1.4749212460623031, None, "Left atrial size"),
    PanEchoTask("LAIDs2D", REGRESSION, (), 4.127667695482703, "cm", "LA internal diameter at systole"),
    PanEchoTask("LAVol", REGRESSION, (), 59.82227248910341, "mL", "LA volume"),
    PanEchoTask("RASize", BINARY, ("Dilated", "Normal"), 0.7001227886670608, None, "Right atrial size", "Dilated", True),
    PanEchoTask("RADimensionM-L(cm)", REGRESSION, (), 3.8826537010431927, "cm", "RA major dimension"),
    PanEchoTask("AVStructure", BINARY, ("Bicuspid", "Normal"), 0.9848413783468221, None, "Bicuspid aortic valve", "Bicuspid", True),
    PanEchoTask("AVStenosis", MULTICLASS, ("Mild|Moderate", "None", "Severe"), 0.9203847129284614, None, "Aortic valve stenosis"),
    PanEchoTask("AVPkVel(m|s)", REGRESSION, (), 1.712958363035933, "m/s", "Aortic valve peak velocity"),
    PanEchoTask("AVRegurg", MULTICLASS, ("Mild", "Moderate|Severe", "None|Trace"), 1.5436156398788055, None, "Aortic valve regurgitation"),
    PanEchoTask("LVOT20mmHg", BINARY, ("0.0", "1.0"), 0.009691400136442072, None, "Elevated LVOT pressure gradient", "1.0"),
    PanEchoTask("MVStenosis", BINARY, ("Mild|Moderate|Severe", "None"), 0.9743735514059562, None, "Mitral valve stenosis", "Mild|Moderate|Severe", True),
    PanEchoTask("MVRegurgitation", MULTICLASS, ("Mild", "Moderate|Severe", "None|Trace"), 1.1843198385309197, None, "Mitral valve regurgitation"),
    PanEchoTask("TVRegurgitation", MULTICLASS, ("Mild", "Moderate|Severe", "None|Trace"), 1.1358925652692495, None, "Tricuspid valve regurgitation"),
    PanEchoTask("TVPkGrad", REGRESSION, (), 26.13051951375867, "mmHg", "Tricuspid valve peak gradient"),
    PanEchoTask("RAP-8-or-higher", BINARY, ("0.0", "1.0"), 0.1608393621308361, None, "Elevated RA pressure (>= 8 mmHg)", "1.0"),
    PanEchoTask("AORoot", REGRESSION, (), 3.386209349294387, "cm", "Aortic root diameter"),
)
# fmt: on

TASKS_BY_NAME: dict[str, PanEchoTask] = {task.name: task for task in PANECHO_TASKS}
ALL_TASK_NAMES: tuple[str, ...] = tuple(TASKS_BY_NAME)

# Tasks that line up with EchoNet-Dynamic FileList.csv columns.
ECHONET_TASK_COLUMNS: dict[str, str] = {
    "EF": "EF",
    "LVEDV": "EDV",
    "LVESV": "ESV",
}


def resolve_tasks(tasks: str | list[str] | tuple[str, ...] = "all") -> tuple[PanEchoTask, ...]:
    """Return task objects for ``'all'`` or a list of task names (original order kept)."""
    if tasks == "all":
        return PANECHO_TASKS
    if isinstance(tasks, str):
        tasks = [tasks]
    unknown = [name for name in tasks if name not in TASKS_BY_NAME]
    if unknown:
        raise KeyError(f"Unknown PanEcho task(s): {unknown}. Valid names: {list(ALL_TASK_NAMES)}")
    wanted = set(tasks)
    return tuple(task for task in PANECHO_TASKS if task.name in wanted)


__all__ = [
    "PanEchoTask",
    "PANECHO_TASKS",
    "TASKS_BY_NAME",
    "ALL_TASK_NAMES",
    "ECHONET_TASK_COLUMNS",
    "REGRESSION",
    "BINARY",
    "MULTICLASS",
    "resolve_tasks",
]
