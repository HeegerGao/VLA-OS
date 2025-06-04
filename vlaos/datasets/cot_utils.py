import enum


class CotTag(enum.Enum):
    TASK = "TASK:"
    PLAN = "PLAN:"
    VISIBLE_OBJECTS = "VISIBLE OBJECTS:"
    SUBTASK_REASONING = "SUBTASK REASONING:"
    SUBTASK = "SUBTASK:"
    MOVE_REASONING = "MOVE REASONING:"
    MOVE = "MOVE:"
    GRIPPER_POSITION = "GRIPPER POSITION:"
    ACTION = "ACTION:"
    # visual planning
    VISUAL_BBOX = "VISUAL OBJECT BBOXES:"
    VISUAL_FLOW = "VISUAL EE FLOW:"
    VISUAL_AFFORDANCE = "VISUAL AFFORDANCE:"

def abbreviate_tag(tag: str):
    return tag[0] + tag[-2]


def get_cot_tags_list():
    return [
        CotTag.TASK.value,
        CotTag.PLAN.value,
        CotTag.VISIBLE_OBJECTS.value,
        CotTag.SUBTASK_REASONING.value,
        CotTag.SUBTASK.value,
        CotTag.MOVE_REASONING.value,
        CotTag.MOVE.value,
        CotTag.GRIPPER_POSITION.value,
        CotTag.VISUAL_BBOX.value,
        CotTag.VISUAL_FLOW.value,
        CotTag.VISUAL_AFFORDANCE.value,
        CotTag.ACTION.value,
    ]


def get_cot_database_keys():
    return {
        CotTag.TASK.value: "task",
        CotTag.PLAN.value: "plan",
        CotTag.VISIBLE_OBJECTS.value: "bboxes",
        CotTag.SUBTASK_REASONING.value: "subtask_reason",
        CotTag.SUBTASK.value: "subtask",
        CotTag.MOVE_REASONING.value: "move_reason",
        CotTag.MOVE.value: "move",
        CotTag.GRIPPER_POSITION.value: "gripper",
        CotTag.VISUAL_BBOX.value: "visual_bboxes",
        CotTag.VISUAL_FLOW.value: "visual_flow",
        CotTag.VISUAL_AFFORDANCE.value: "visual_affordance",
        CotTag.ACTION.value: "action",
    }
    
def get_language_cot_tags_list():
    return [
        CotTag.TASK.value,
        CotTag.PLAN.value,
        CotTag.VISIBLE_OBJECTS.value,
        CotTag.SUBTASK_REASONING.value,
        CotTag.SUBTASK.value,
        CotTag.MOVE_REASONING.value,
        CotTag.MOVE.value,
        CotTag.GRIPPER_POSITION.value,
    ]
    
def get_visual_cot_tags_list():
    return [
        CotTag.VISUAL_BBOX.value,
        CotTag.VISUAL_FLOW.value,
        CotTag.VISUAL_AFFORDANCE.value,
    ]