'''
Adapted from https://github.com/google-research/google-research/tree/master/android_in_the_wild
'''

import jax
import jax.numpy as jnp
import numpy as np

# import action_type as action_type_lib
import enum

class ActionType(enum.IntEnum):
    # Placeholders for unused enum values
    UNUSED_0 = 0
    UNUSED_1 = 1
    UNUSED_2 = 2
    UNUSED_8 = 8
    UNUSED_9 = 9

    ########### Agent actions ###########

    # A type action that sends text to the emulator. Note that this simply sends
    # text and does not perform any clicks for element focus or enter presses for
    # submitting text.
    TYPE = 3

    # The dual point action used to represent all gestures.
    DUAL_POINT = 4

    # These actions differentiate pressing the home and back button from touches.
    # They represent explicit presses of back and home performed using ADB.
    PRESS_BACK = 5
    PRESS_HOME = 6

    # An action representing that ADB command for hitting enter was performed.
    PRESS_ENTER = 7

    ########### Episode status actions ###########

    # An action used to indicate the desired task has been completed and resets
    # the environment. This action should also be used in the case that the task
    # has already been completed and there is nothing to do.
    # e.g. The task is to turn on the Wi-Fi when it is already on
    STATUS_TASK_COMPLETE = 10

    # An action used to indicate that desired task is impossible to complete and
    # resets the environment. This can be a result of many different things
    # including UI changes, Android version differences, etc.
    STATUS_TASK_IMPOSSIBLE = 11


_TAP_DISTANCE_THRESHOLD = 0.14  # Fraction of the screen
ANNOTATION_WIDTH_AUGMENT_FRACTION = 1.4
ANNOTATION_HEIGHT_AUGMENT_FRACTION = 1.4

# Interval determining if an action is a tap or a swipe.
_SWIPE_DISTANCE_THRESHOLD = 0.04


def _yx_in_bounding_boxes(
    yx, bounding_boxes
):
  """Check if the (y,x) point is contained in each bounding box.

  Args:
    yx: The (y, x) coordinate in pixels of the point.
    bounding_boxes: A 2D int array of shape (num_bboxes, 4), where each row
      represents a bounding box: (y_top_left, x_top_left, box_height,
      box_width). Note: containment is inclusive of the bounding box edges.

  Returns:
    is_inside: A 1D bool array where each element specifies if the point is
      contained within the respective box.
  """
  y, x = yx

  # `bounding_boxes` has shape (n_elements, 4); we extract each array along the
  # last axis into shape (n_elements, 1), then squeeze unneeded dimension.
  top, left, height, width = [
      jnp.squeeze(v, axis=-1) for v in jnp.split(bounding_boxes, 4, axis=-1)
  ]

  # The y-axis is inverted for AndroidEnv, so bottom = top + height.
  bottom, right = top + height, left + width

  return jnp.logical_and(y >= top, y <= bottom) & jnp.logical_and(
      x >= left, x <= right)


def _resize_annotation_bounding_boxes(
    annotation_positions, annotation_width_augment_fraction,
    annotation_height_augment_fraction):
  """Resize the bounding boxes by the given fractions.

  Args:
    annotation_positions: Array of shape (N, 4), where each row represents the
      (y, x, height, width) of the bounding boxes.
    annotation_width_augment_fraction: The fraction to augment the box widths,
      E.g., 1.4 == 240% total increase.
    annotation_height_augment_fraction: Same as described for width, but for box
      height.

  Returns:
    Resized bounding box.

  """
  height_change = (
      annotation_height_augment_fraction * annotation_positions[:, 2])
  width_change = (
      annotation_width_augment_fraction * annotation_positions[:, 3])

  # Limit bounding box positions to the screen.
  resized_annotations = jnp.stack([
      jnp.maximum(0, annotation_positions[:, 0] - (height_change / 2)),
      jnp.maximum(0, annotation_positions[:, 1] - (width_change / 2)),
      jnp.minimum(1, annotation_positions[:, 2] + height_change),
      jnp.minimum(1, annotation_positions[:, 3] + width_change),
  ],
                                  axis=1)
  return resized_annotations


def is_tap_action(normalized_start_yx,
                  normalized_end_yx):
  distance = jnp.linalg.norm(
      jnp.array(normalized_start_yx) - jnp.array(normalized_end_yx))
  return distance <= _SWIPE_DISTANCE_THRESHOLD


def _is_non_dual_point_action(action_type):
  return jnp.not_equal(action_type, ActionType.DUAL_POINT)


def _check_tap_actions_match(
    tap_1_yx,
    tap_2_yx,
    annotation_positions,
    matching_tap_distance_threshold_screen_percentage,
    annotation_width_augment_fraction,
    annotation_height_augment_fraction,
):
  """Determines if two tap actions are the same."""
  resized_annotation_positions = _resize_annotation_bounding_boxes(
      annotation_positions,
      annotation_width_augment_fraction,
      annotation_height_augment_fraction,
  )

  # Check if the ground truth tap action falls in an annotation's bounding box.
  tap1_in_box = _yx_in_bounding_boxes(tap_1_yx, resized_annotation_positions)
  tap2_in_box = _yx_in_bounding_boxes(tap_2_yx, resized_annotation_positions)
  both_in_box = jnp.max(tap1_in_box & tap2_in_box)

  # If the ground-truth tap action falls outside any of the annotation
  # bounding boxes or one of the actions is inside a bounding box and the other
  # is outside bounding box or vice versa, compare the points using Euclidean
  # distance.
  within_threshold = (
      jnp.linalg.norm(jnp.array(tap_1_yx) - jnp.array(tap_2_yx))
      <= matching_tap_distance_threshold_screen_percentage
  )
  return jnp.logical_or(both_in_box, within_threshold)


def _check_drag_actions_match(
    drag_1_touch_yx,
    drag_1_lift_yx,
    drag_2_touch_yx,
    drag_2_lift_yx,
):
  """Determines if two drag actions are the same."""
  # Store drag deltas (the change in the y and x coordinates from touch to
  # lift), magnitudes, and the index of the main axis, which is the axis with
  # the greatest change in coordinate value (e.g. a drag starting at (0, 0) and
  # ending at (0.3, 0.5) has a main axis index of 1).
  drag_1_deltas = drag_1_lift_yx - drag_1_touch_yx
  drag_1_magnitudes = jnp.abs(drag_1_deltas)
  drag_1_main_axis = np.argmax(drag_1_magnitudes)
  drag_2_deltas = drag_2_lift_yx - drag_2_touch_yx
  drag_2_magnitudes = jnp.abs(drag_2_deltas)
  drag_2_main_axis = np.argmax(drag_2_magnitudes)

  return jnp.equal(drag_1_main_axis, drag_2_main_axis)


def check_actions_match(
    action_1_touch_yx,
    action_1_lift_yx,
    action_1_action_type,
    action_2_touch_yx,
    action_2_lift_yx,
    action_2_action_type,
    annotation_positions,
    tap_distance_threshold = _TAP_DISTANCE_THRESHOLD,
    annotation_width_augment_fraction = ANNOTATION_WIDTH_AUGMENT_FRACTION,
    annotation_height_augment_fraction = ANNOTATION_HEIGHT_AUGMENT_FRACTION,
):
  """Determines if two actions are considered to be the same.

  Two actions being "the same" is defined here as two actions that would result
  in a similar screen state.

  Args:
    action_1_touch_yx: The (y, x) coordinates of the first action's touch.
    action_1_lift_yx: The (y, x) coordinates of the first action's lift.
    action_1_action_type: The action type of the first action.
    action_2_touch_yx: The (y, x) coordinates of the second action's touch.
    action_2_lift_yx: The (y, x) coordinates of the second action's lift.
    action_2_action_type: The action type of the second action.
    annotation_positions: The positions of the UI annotations for the screen. It
      is A 2D int array of shape (num_bboxes, 4), where each row represents a
      bounding box: (y_top_left, x_top_left, box_height, box_width). Note that
      containment is inclusive of the bounding box edges.
    tap_distance_threshold: The threshold that determines if two taps result in
      a matching screen state if they don't fall the same bounding boxes.
    annotation_width_augment_fraction: The fraction to increase the width of the
      bounding box by.
    annotation_height_augment_fraction: The fraction to increase the height of
      of the bounding box by.

  Returns:
    A boolean representing whether the two given actions are the same or not.
  """
  action_1_touch_yx = jnp.asarray(action_1_touch_yx)
  action_1_lift_yx = jnp.asarray(action_1_lift_yx)
  action_2_touch_yx = jnp.asarray(action_2_touch_yx)
  action_2_lift_yx = jnp.asarray(action_2_lift_yx)

  # Checks if at least one of the actions is global (i.e. not DUAL_POINT),
  # because if that is the case, only the actions' types need to be compared.
  has_non_dual_point_action = jnp.logical_or(
      _is_non_dual_point_action(action_1_action_type),
      _is_non_dual_point_action(action_2_action_type),
  )
  #print("non dual point: "+str(has_non_dual_point_action))

  different_dual_point_types = jnp.logical_xor(
      is_tap_action(action_1_touch_yx, action_1_lift_yx),
      is_tap_action(action_2_touch_yx, action_2_lift_yx),
  )
  #print("different dual type: "+str(different_dual_point_types))

  is_tap = jnp.logical_and(
      is_tap_action(action_1_touch_yx, action_1_lift_yx),
      is_tap_action(action_2_touch_yx, action_2_lift_yx),
  )
  #print("is tap: "+str(is_tap))

  taps_match = _check_tap_actions_match(
      action_1_touch_yx,
      action_2_touch_yx,
      annotation_positions,
      tap_distance_threshold,
      annotation_width_augment_fraction,
      annotation_height_augment_fraction,
  )
  #print("tap match: "+str(taps_match))

  taps_match = jnp.logical_and(is_tap, taps_match)
  #print("tap match: "+str(taps_match))

  drags_match = _check_drag_actions_match(
      action_1_touch_yx, action_1_lift_yx, action_2_touch_yx, action_2_lift_yx
  )
  drags_match = jnp.where(is_tap, False, drags_match)
  #print("drag match: "+str(drags_match))

  return jnp.where(
      has_non_dual_point_action,
      jnp.equal(action_1_action_type, action_2_action_type),
      jnp.where(
          different_dual_point_types,
          False,
          jnp.logical_or(taps_match, drags_match),
      ),
  )


def action_2_format(step_data):
    # 把test数据集中的动作格式转换为计算matching score的格式
    action_type = step_data["action_type_id"]

    if action_type == 4:
        if step_data["action_type_text"] == 'click':  # 点击
            touch_point = step_data["touch"]
            lift_point = step_data["lift"]
        else:  # 上下左右滑动
            if step_data["action_type_text"] == 'scroll down':
                touch_point = [0.5, 0.8]
                lift_point = [0.5, 0.2]
            elif step_data["action_type_text"] == 'scroll up':
                touch_point = [0.5, 0.2]
                lift_point = [0.5, 0.8]
            elif step_data["action_type_text"] == 'scroll left':
                touch_point = [0.2, 0.5]
                lift_point = [0.8, 0.5]
            elif step_data["action_type_text"] == 'scroll right':
                touch_point = [0.8, 0.5]
                lift_point = [0.2, 0.5]
    else:
        touch_point = [-1.0, -1.0]
        lift_point = [-1.0, -1.0]

    if action_type == 3:
        typed_text = step_data["type_text"]
    else:
        typed_text = ""

    action = {"action_type": action_type, "touch_point": touch_point, "lift_point": lift_point,
              "typed_text": typed_text}

    action["touch_point"] = [action["touch_point"][1], action["touch_point"][0]]
    action["lift_point"] = [action["lift_point"][1], action["lift_point"][0]]
    action["typed_text"] = action["typed_text"].lower()

    return action


def pred_2_format(step_data):
    # 把模型输出的内容转换为计算action_matching的格式
    action_type = step_data["action_type"]

    if action_type == 4:  # 点击
        action_type_new = 4
        touch_point = step_data["click_point"]
        lift_point = step_data["click_point"]
        typed_text = ""
    elif action_type == 0:
        action_type_new = 4
        touch_point = [0.5, 0.8]
        lift_point = [0.5, 0.2]
        typed_text = ""
    elif action_type == 1:
        action_type_new = 4
        touch_point = [0.5, 0.2]
        lift_point = [0.5, 0.8]
        typed_text = ""
    elif action_type == 8:
        action_type_new = 4
        touch_point = [0.2, 0.5]
        lift_point = [0.8, 0.5]
        typed_text = ""
    elif action_type == 9:
        action_type_new = 4
        touch_point = [0.8, 0.5]
        lift_point = [0.2, 0.5]
        typed_text = ""
    else:
        action_type_new = action_type
        touch_point = [-1.0, -1.0]
        lift_point = [-1.0, -1.0]
        typed_text = ""
        if action_type_new == 3:
            typed_text = step_data["typed_text"]

    action = {"action_type": action_type_new, "touch_point": touch_point, "lift_point": lift_point,
              "typed_text": typed_text}

    action["touch_point"] = [action["touch_point"][1], action["touch_point"][0]]
    action["lift_point"] = [action["lift_point"][1], action["lift_point"][0]]
    action["typed_text"] = action["typed_text"].lower()

    return action


def pred_2_format_simplified(step_data):
    # 把模型输出的内容转换为计算action_matching的格式
    action_type = step_data["action_type"]

    if action_type == 'click' :  # 点击
        action_type_new = 4
        touch_point = step_data["click_point"]
        lift_point = step_data["click_point"]
        typed_text = ""
    elif action_type == 'scroll' and step_data["direction"] == 'down':
        action_type_new = 4
        touch_point = [0.5, 0.8]
        lift_point = [0.5, 0.2]
        typed_text = ""
    elif action_type == 'scroll' and step_data["direction"] == 'up':
        action_type_new = 4
        touch_point = [0.5, 0.2]
        lift_point = [0.5, 0.8]
        typed_text = ""
    elif action_type == 'scroll' and step_data["direction"] == 'left':
        action_type_new = 4
        touch_point = [0.2, 0.5]
        lift_point = [0.8, 0.5]
        typed_text = ""
    elif action_type == 'scroll' and step_data["direction"] == 'right':
        action_type_new = 4
        touch_point = [0.8, 0.5]
        lift_point = [0.2, 0.5]
        typed_text = ""
    elif action_type == 'type':
        action_type_new = 3
        touch_point = [-1.0, -1.0]
        lift_point = [-1.0, -1.0]
        typed_text = step_data["text"]
    elif action_type == 'navigate_back':
        action_type_new = 5
        touch_point = [-1.0, -1.0]
        lift_point = [-1.0, -1.0]
        typed_text = ""
    elif action_type == 'navigate_home':
        action_type_new = 6
        touch_point = [-1.0, -1.0]
        lift_point = [-1.0, -1.0]
        typed_text = ""
    else:
        action_type_new = action_type
        touch_point = [-1.0, -1.0]
        lift_point = [-1.0, -1.0]
        typed_text = ""
        # if action_type_new == 'type':
        #     typed_text = step_data["text"]

    action = {"action_type": action_type_new, "touch_point": touch_point, "lift_point": lift_point,
              "typed_text": typed_text}

    action["touch_point"] = [action["touch_point"][1], action["touch_point"][0]]
    action["lift_point"] = [action["lift_point"][1], action["lift_point"][0]]
    action["typed_text"] = action["typed_text"].lower()

    return action