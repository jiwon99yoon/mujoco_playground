# /home/dyros/mujoco_playground/learning/load_ppo_inference_mujoco_view.py
"""Load a PPO agent using JAX and inference via mujoco simulator on the specified environment."""

from datetime import datetime
import functools
import json
import os
import time
import warnings

from absl import app
from absl import flags
from absl import logging
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from etils import epath
from flax.training import orbax_utils
import jax
import jax.numpy as jp
import mediapy as media
from ml_collections import config_dict
import mujoco
from orbax import checkpoint as ocp
from tensorboardX import SummaryWriter
import wandb

import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)

# Suppress warnings

# Suppress RuntimeWarnings from JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")

# absl.flags가 자동으로 sys.argv를 훑으며, --num_timesteps 뒤에 붙은 값을 읽어 FLAGS.num_timesteps에 값을 할당
# flag parser --로 시작하는 토큰 : =이든 띄어쓰기든 다음 커멘트라인 토큰을 값으로 취함

_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "LeapCubeReorient",
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)
_VISION = flags.DEFINE_boolean("vision", False, "Use vision input")
_LOAD_CHECKPOINT_PATH = flags.DEFINE_string(
    "load_checkpoint_path", None, "Path to load checkpoint from"
)
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name")
# _PLAY_ONLY = flags.DEFINE_boolean(
#     "play_only", True, "If true, only play with the model and do not train"
# )
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "domain_randomization", False, "Use domain randomization"
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_REWARD_SCALING = flags.DEFINE_float("reward_scaling", 0.1, "Reward scaling")
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 1000, "Episode length")
_NORMALIZE_OBSERVATIONS = flags.DEFINE_boolean(
    "normalize_observations", True, "Normalize observations"
)
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 1, "Action repeat")
_UNROLL_LENGTH = flags.DEFINE_integer("unroll_length", 10, "Unroll length")
_NUM_MINIBATCHES = flags.DEFINE_integer(
    "num_minibatches", 8, "Number of minibatches"
)
_NUM_UPDATES_PER_BATCH = flags.DEFINE_integer(
    "num_updates_per_batch", 8, "Number of updates per batch"
)
_DISCOUNTING = flags.DEFINE_float("discounting", 0.97, "Discounting")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 5e-4, "Learning rate")
_ENTROPY_COST = flags.DEFINE_float("entropy_cost", 5e-3, "Entropy cost")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 1024, "Number of environments")
_NUM_EVAL_ENVS = flags.DEFINE_integer(
    "num_eval_envs", 128, "Number of evaluation environments"
)
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 256, "Batch size")
_MAX_GRAD_NORM = flags.DEFINE_float("max_grad_norm", 1.0, "Max grad norm")
_CLIPPING_EPSILON = flags.DEFINE_float(
    "clipping_epsilon", 0.2, "Clipping epsilon for PPO"
)
_POLICY_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "policy_hidden_layer_sizes",
    [64, 64, 64],
    "Policy hidden layer sizes",
)
_VALUE_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "value_hidden_layer_sizes",
    [64, 64, 64],
    "Value hidden layer sizes",
)
_POLICY_OBS_KEY = flags.DEFINE_string(
    "policy_obs_key", "state", "Policy obs key"
)
_VALUE_OBS_KEY = flags.DEFINE_string("value_obs_key", "state", "Value obs key")



# ml_collections의 import config_dict로서 모델 정보 저장
# 먼저, env_config = manipulation.get_default_config(env_name)으로 각 환경에 정의된 episode_length, action_repeat, 

# get_rl_config는 입력받은 env_name에 맞는 환경에 최적화된 brax내 ppo_config값을 불러옴
# 예로, num_timesteps, num_evals, unroll_length, num_minibatches, num_updates_per_batch, discounting, learning_rate, 
# entropy_cost, num_envs, batch_size, num_resets_per_eval, max_grad_norm, 
# 또한, network_factory = config_dict.create(policy_hidden_layer_size, value_hidden_layer_size, policy_obs_key, value_obs_key)
# 등의 ppo model에 대한 hyperparameter 값을 지정 

def get_rl_config(env_name: str) -> config_dict.ConfigDict:
  if env_name in mujoco_playground.manipulation._envs:
    if _VISION.value:
      return manipulation_params.brax_vision_ppo_config(env_name)
    return manipulation_params.brax_ppo_config(env_name)
  elif env_name in mujoco_playground.locomotion._envs:
    if _VISION.value:
      return locomotion_params.brax_vision_ppo_config(env_name)
    return locomotion_params.brax_ppo_config(env_name)
  elif env_name in mujoco_playground.dm_control_suite._envs:
    if _VISION.value:
      return dm_control_suite_params.brax_vision_ppo_config(env_name)
    return dm_control_suite_params.brax_ppo_config(env_name)

  raise ValueError(f"Env {env_name} not found in {registry.ALL_ENVS}.")

def main(argv):
  """Run training and evaluation for the specified environment."""

  del argv

  # Load environment configuration
  # 환경에 대한 ctrl_dt, sim_dt, episode_lenth, action_repeat, action_scale과 함꼐
  # reward_config에 대한 정보도 제공 : get_default_config()를 통해 pick~~과 같은 정확한 python file에 접근
  env_cfg = registry.get_default_config(_ENV_NAME.value)

  # env_name에 맞는 ppo_params 불러오기 : ppo hyperparameter
  ppo_params = get_rl_config(_ENV_NAME.value)
  ppo_params.num_timesteps = 0
  # 아래 : 사용자 입력에 따른 num_timestep 등 다양한 hyperparameter들 대입
  # if _PLAY_ONLY.present:
  #   ppo_params.num_timesteps = 0
  if _REWARD_SCALING.present:
    ppo_params.reward_scaling = _REWARD_SCALING.value
  if _EPISODE_LENGTH.present:
    ppo_params.episode_length = _EPISODE_LENGTH.value
  if _NORMALIZE_OBSERVATIONS.present:
    ppo_params.normalize_observations = _NORMALIZE_OBSERVATIONS.value
  if _ACTION_REPEAT.present:
    ppo_params.action_repeat = _ACTION_REPEAT.value
  if _UNROLL_LENGTH.present:
    ppo_params.unroll_length = _UNROLL_LENGTH.value
  if _NUM_MINIBATCHES.present:
    ppo_params.num_minibatches = _NUM_MINIBATCHES.value
  if _NUM_UPDATES_PER_BATCH.present:
    ppo_params.num_updates_per_batch = _NUM_UPDATES_PER_BATCH.value
  if _DISCOUNTING.present:
    ppo_params.discounting = _DISCOUNTING.value
  if _LEARNING_RATE.present:
    ppo_params.learning_rate = _LEARNING_RATE.value
  if _ENTROPY_COST.present:
    ppo_params.entropy_cost = _ENTROPY_COST.value
  if _NUM_ENVS.present:
    ppo_params.num_envs = _NUM_ENVS.value
  if _NUM_EVAL_ENVS.present:
    ppo_params.num_eval_envs = _NUM_EVAL_ENVS.value
  if _BATCH_SIZE.present:
    ppo_params.batch_size = _BATCH_SIZE.value
  if _MAX_GRAD_NORM.present:
    ppo_params.max_grad_norm = _MAX_GRAD_NORM.value
  if _CLIPPING_EPSILON.present:
    ppo_params.clipping_epsilon = _CLIPPING_EPSILON.value
  if _POLICY_HIDDEN_LAYER_SIZES.present:
    ppo_params.network_factory.policy_hidden_layer_sizes = list(
        map(int, _POLICY_HIDDEN_LAYER_SIZES.value)
    )
  if _VALUE_HIDDEN_LAYER_SIZES.present:
    ppo_params.network_factory.value_hidden_layer_sizes = list(
        map(int, _VALUE_HIDDEN_LAYER_SIZES.value)
    )
  if _POLICY_OBS_KEY.present:
    ppo_params.network_factory.policy_obs_key = _POLICY_OBS_KEY.value
  if _VALUE_OBS_KEY.present:
    ppo_params.network_factory.value_obs_key = _VALUE_OBS_KEY.value
  if _VISION.value:
    env_cfg.vision = True
    env_cfg.vision_config.render_batch_size = ppo_params.num_envs
  
  # 환경에 대한 정보 모두 load
  # 각 task 별 python file에 있는 env class를 불러옴
  # class type은 mjx_env.py에 있는 MjxEnv() class 형태 
  env = registry.load(_ENV_NAME.value, config=env_cfg)

  print(f"Environment Config:\n{env_cfg}")
  print(f"PPO Training Parameters:\n{ppo_params}")

  # 파일 명 생성
  # Generate unique experiment name
  now = datetime.now()
  timestamp = now.strftime("%Y%m%d-%H%M%S")
  exp_name = f"{_ENV_NAME.value}-{timestamp}"
  if _SUFFIX.value is not None:
    exp_name += f"-{_SUFFIX.value}"
  print(f"Experiment name: {exp_name}")

  # Handle checkpoint loading
  if _LOAD_CHECKPOINT_PATH.value is not None:
    # Convert to absolute path
    ckpt_path = epath.Path(_LOAD_CHECKPOINT_PATH.value).resolve()
    if ckpt_path.is_dir():
      latest_ckpts = list(ckpt_path.glob("*"))            #glob()로 모든 항목을 나열
      latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]       # dir()체크로 폴더만 골라
      latest_ckpts.sort(key=lambda x: int(x.name))                          # 정수 변환하여 정렬
      latest_ckpt = latest_ckpts[-1]                                        # 제일 큰 숫자를 골라
      restore_checkpoint_path = latest_ckpt                                 # 가장 최근 체크포인트로 간주
      print(f"Restoring from: {restore_checkpoint_path}")
      # 복원 경로는 latest_ckpts 중 가장 timestep이 많이 실행된 것 (제일 많은 학습이 이뤄진 것) 파일을 경로로 지정
      # ex. /home/dyros/mujoco_playground/logs/PandaPickCube-20250805-174125/checkpoints/000020152320/
    else:
      restore_checkpoint_path = ckpt_path
      print(f"Restoring from checkpoint: {restore_checkpoint_path}")
  else:
    print("No checkpoint path provided, not restoring from checkpoint")
    restore_checkpoint_path = None

  training_params = dict(ppo_params)
  if "network_factory" in training_params:
    del training_params["network_factory"]
    # del : 객체를 삭제하거나 변수의 참조를 제거

  # vision이 있으면 brax 내 vision ppo_network 참조, 아니라면 그냥 ppo_network참조
  # ppo의 기본 network 구조를 class 형태로 저장
  network_fn = (
      ppo_networks_vision.make_ppo_networks_vision
      if _VISION.value
      else ppo_networks.make_ppo_networks
  ) 

  # hasattr (object, name) : object에 name 속성이 존재하면 true값 반환
  if hasattr(ppo_params, "network_factory"):
    network_factory = functools.partial(
        network_fn, **ppo_params.network_factory
    )
  else:
    network_factory = network_fn

  if _DOMAIN_RANDOMIZATION.value:
    training_params["randomization_fn"] = registry.get_domain_randomizer(
        _ENV_NAME.value
    )

  # LeapCubeRotateZAxis / LeapCubeReorient의 경우 domain randomization 요소 넣음

  if _VISION.value:
    env = wrapper.wrap_for_brax_training(
        env,
        vision=True,
        num_vision_envs=env_cfg.vision_config.render_batch_size,
        episode_length=ppo_params.episode_length,
        action_repeat=ppo_params.action_repeat,
        randomization_fn=training_params.get("randomization_fn"),
    )

  # vision의 경우 mujoco_playground 내 wrap_for_brax_training()에서 MadronaWrapper() 통해서
  # DOmain Randomziser 등 불러옴

  num_eval_envs = (
      ppo_params.num_envs
      if _VISION.value
      else ppo_params.get("num_eval_envs", 128)
  )

  if "num_eval_envs" in training_params:
    del training_params["num_eval_envs"]

  # training_params : 파이썬 dict file 유형 -> ppo.py의 def train()에 ppo hyperparameter 값으로 할당됨
  # restore_checkpoint_path
  # save_checkpoint_path
  # wrap_env_fn
  # num_eval_envs
  train_fn = functools.partial(
      ppo.train,
      **training_params,
      network_factory=network_factory,
      seed=_SEED.value,
      restore_checkpoint_path=restore_checkpoint_path,
      wrap_env_fn=None if _VISION.value else wrapper.wrap_for_brax_training,
      num_eval_envs=num_eval_envs,
  )
  # train_fn은 brax/ppo/train.py의 def train()함수의 일부분
  # def train()의 output : make policy, params, metrics

  # Load evaluation environment
  eval_env = (
      None if _VISION.value else registry.load(_ENV_NAME.value, config=env_cfg)
  )

  print("Starting model inference...")
  # inference time checking 용
  inf_model_start = time.monotonic()  

  # Train or load the model
  make_inference_fn, params, _ = train_fn(  # pylint: disable=no-value-for-parameter
      environment=env,
      eval_env=None if _VISION.value else eval_env,
  )

  inference_fn = make_inference_fn(params, deterministic=True)
  jit_inference_fn = jax.jit(inference_fn)

  # Prepare for evaluation
  eval_env = (
      None if _VISION.value else registry.load(_ENV_NAME.value, config=env_cfg)
  )
  num_envs = 1
  if _VISION.value:
    eval_env = env
    num_envs = env_cfg.vision_config.render_batch_size

  jit_reset = jax.jit(eval_env.reset)
  jit_step = jax.jit(eval_env.step)

  rng = jax.random.PRNGKey(123)
  rng, reset_rng = jax.random.split(rng)
  if _VISION.value:
    reset_rng = jp.asarray(jax.random.split(reset_rng, num_envs))
  state = jit_reset(reset_rng)
  state0 = (
      jax.tree_util.tree_map(lambda x: x[0], state) if _VISION.value else state
  )
  #rollout = [state0]

  # model inference time checking 용
  inf_model_end = time.monotonic()
  print(f"Total model inference time:      {inf_model_end - inf_model_start:.3f}s")

  # --------------------------- mujoco-simulator viewer ---------------------------

  import mujoco.viewer
  import numpy as np
  from mujoco.mjx._src import math

  # ───  constants we need once ─────────────────────────────────
  GRIPPER_SITE  = eval_env._gripper_site        
  OBJ_BODY      = eval_env._obj_body            
  MOCAP_TARGET  = eval_env._mocap_target        
  ARM_IDX       = eval_env._robot_qposadr[:-1] 
  TARGET_POS    = jp.asarray(state0.info["target_pos"]) #이렇게 하면 target_pos에 대한 값 갱신 x, 실시간으로 target이 움직일 때 업데이트가 안됨 "근데 그렇게 학습되는게 아님. 굳이 그렇게 보여줄 필요가 없는 것 같음 - 추후 논의"

  def build_obs_pick_cube(qpos, qvel, site_xpos, site_xmat, body_xpos, body_xmat, 
                       mocap_pos, mocap_quat, ctrl, target_pos):
    """PandaPickCube용 observation builder (66차원)"""
    gripper_pos = site_xpos[GRIPPER_SITE]
    gripper_mat = site_xmat[GRIPPER_SITE].ravel()
    target_mat = math.quat_to_mat(mocap_quat[MOCAP_TARGET])

    obs = jp.concatenate([
        qpos,                                                    # joint positions (16) = 7 (arm joint) + 2(finger joint) + 7(box free joint)
        qvel,                                                    # joint velocities (15) = 7 (arm velocity) + 2(finger velocity) + 6(box linear3 angular3)
        gripper_pos,                                            # gripper position (3) = gripper pos
        gripper_mat[3:],                                        # gripper orientation (6) = rotation matrix sym
        body_xmat[OBJ_BODY].ravel()[3:],                       # box orientation (6) = rotation matrix sym
        body_xpos[OBJ_BODY] - gripper_pos,                     # box-gripper relative pos (3) = position
        target_pos - body_xpos[OBJ_BODY],                      # target-box relative pos (3) = 
        target_mat.ravel()[:6] - body_xmat[OBJ_BODY].ravel()[:6], # target-box relative orientation (6)
        ctrl - qpos[ARM_IDX],                                   # control error (8) = 7 (arm actuators) + 1(gripper actuator)
    ])
    return obs

  def build_obs_pick_ball(qpos, qvel, site_xpos, site_xmat, body_xpos, body_xmat, 
                       mocap_pos, mocap_quat, ctrl, target_pos):
    """PandaPickCube용 observation builder (54차원)"""
    gripper_pos = site_xpos[GRIPPER_SITE]
    gripper_mat = site_xmat[GRIPPER_SITE].ravel()
    #target_mat = math.quat_to_mat(mocap_quat[MOCAP_TARGET])

    obs = jp.concatenate([
        qpos,                                                    # joint positions (16) = 7 (arm joint) + 2(finger joint) + 7(box free joint)
        qvel,                                                    # joint velocities (15) = 7 (arm velocity) + 2(finger velocity) + 6(box linear3 angular3)
        gripper_pos,                                            # gripper position (3) = gripper pos
        gripper_mat[3:],                                        # gripper orientation (6) = rotation matrix sym
        #body_xmat[OBJ_BODY].ravel()[3:],                       # box orientation (6) = rotation matrix sym
        body_xpos[OBJ_BODY] - gripper_pos,                     # box-gripper relative pos (3) = position
        target_pos - body_xpos[OBJ_BODY],                      # target-box relative pos (3) = 
        #target_mat.ravel()[:6] - body_xmat[OBJ_BODY].ravel()[:6], # target-box relative orientation (6)
        ctrl - qpos[ARM_IDX],                                   # control error (8) = 7 (arm actuators) + 1(gripper actuator)
    ])
    return obs
  
  def build_obs_pick_cube_cartesian(qpos, qvel, site_xpos, site_xmat, body_xpos, body_xmat,
                                 mocap_pos, mocap_quat, ctrl, target_pos, 
                                 no_soln=0.0, prev_action=None):
    """PandaPickCubeCartesian용 observation builder (70차원)"""
    if prev_action is None:
        prev_action = jp.zeros(3)
    
    # 기본 PandaPickCube observation (66차원)
    base_obs = build_obs_pick_cube(qpos, qvel, site_xpos, site_xmat, body_xpos, body_xmat,
                                  mocap_pos, mocap_quat, ctrl, target_pos)
    
    # Cartesian 환경의 추가 정보 (4차원)
    cartesian_obs = jp.concatenate([
        base_obs,                    # 66차원
        jp.array([no_soln]),        # 1차원: no IK solution flag 
        prev_action                  # 3차원: previous action
    ])
    return cartesian_obs

  def get_obs_builder(env_name):
    """환경 이름에 따라 적절한 observation builder 반환"""
    if "Cartesian" in env_name:
        return build_obs_pick_cube_cartesian, True  # (builder_func, needs_prev_action)
    elif "Ball" in env_name:
        return build_obs_pick_ball, False  
    else:
        return build_obs_pick_cube, False

  def simple_cartesian_controller(data, action, eval_env):
      """간단한 cartesian controller - MJX 환경 사용하지 않음"""
      try:
          from mujoco_playground._src.manipulation.franka_emika_panda import panda_kinematics
          
          action_scale = eval_env._config.action_scale  # 0.005
          
          # 현재 gripper 위치/방향 계산
          current_tip_transform = panda_kinematics.compute_franka_fk(jp.asarray(data.ctrl[:7]))
          current_tip_pos = np.asarray(current_tip_transform[:3, 3])
          current_tip_rot = np.asarray(current_tip_transform[:3, :3])
          
          # Cartesian increment (action[0]=y, action[1]=z)
          pos_increment = np.zeros(3)
          pos_increment[1] = action[0] * action_scale  # y movement
          pos_increment[2] = action[1] * action_scale  # z movement
          
          new_tip_pos = current_tip_pos + pos_increment
          
          # Position constraints (PandaPickCubeCartesian에서 가져온 값들)
          new_tip_pos[0] = np.clip(new_tip_pos[0], 0.25, 0.77)
          new_tip_pos[1] = np.clip(new_tip_pos[1], -0.32, 0.32)
          new_tip_pos[2] = np.clip(new_tip_pos[2], 0.02, 0.5)
          
          # 새로운 transformation matrix
          new_tip_mat = np.eye(4, dtype=np.float64)
          new_tip_mat[:3, :3] = current_tip_rot
          new_tip_mat[:3, 3] = new_tip_pos
          
          # IK 계산
          new_joint_pos = panda_kinematics.compute_franka_ik(
              jp.asarray(new_tip_mat), 
              jp.asarray(data.ctrl[6]), 
              jp.asarray(data.ctrl[:7])
          )
          
          # IK 해가 유효한지 확인
          if not np.any(np.isnan(new_joint_pos)):
              data.ctrl[:7] = np.asarray(new_joint_pos)
          
      except Exception as e:
          print(f"IK failed: {e}, keeping current position")
          # IK 실패시 현재 위치 유지
          pass
      
      # Gripper action (action[2])
      gripper_cmd = -1.0 if action[2] < 0 else 1.0
      gripper_delta = gripper_cmd * 0.02  # 2cm per step
      new_gripper_pos = data.ctrl[7] + gripper_delta
      data.ctrl[7] = np.clip(new_gripper_pos, -0.04, 0.04)
      
      # 전체 joint limits 적용
      data.ctrl[:] = np.clip(data.ctrl, 
                            np.asarray(eval_env._lowers), 
                            np.asarray(eval_env._uppers))

  def build_observation(data, target_pos, obs_builder, prev_action=None):
    """통합된 observation 빌더 - obs_builder 함수를 활용"""
    obs_args = [
        jp.asarray(data.qpos),
        jp.asarray(data.qvel),
        jp.asarray(data.site_xpos),
        jp.asarray(data.site_xmat),
        jp.asarray(data.xpos),
        jp.asarray(data.xmat),
        jp.asarray(data.mocap_pos),
        jp.asarray(data.mocap_quat),
        jp.asarray(data.ctrl),
        target_pos
    ]
    
    # Cartesian 환경인 경우 추가 인자 전달
    if obs_builder == build_obs_pick_cube_cartesian:
        obs_args.extend([0.0, prev_action])  # no_soln=0.0, prev_action
    
    return obs_builder(*obs_args)
  
  # # bulid_observation_ball엔 아래와 같은 obs_builder다 포함해야함 <- 그 이유가 뭐지...
  # def build_observation_ball(data, target_pos, obs_builder, prev_action=None):
  #   """통합된 observation 빌더 - obs_builder 함수를 활용"""
  #   obs_args = [
  #       jp.asarray(data.qpos),
  #       jp.asarray(data.qvel),
  #       jp.asarray(data.site_xpos),
  #       jp.asarray(data.site_xmat),
  #       jp.asarray(data.xpos),
  #       jp.asarray(data.xmat),
  #       jp.asarray(data.mocap_pos),
  #       jp.asarray(data.mocap_quat),
  #       jp.asarray(data.ctrl),
  #       target_pos
  #   ]
    
  #   return obs_builder(*obs_args)

  # ═══════════════════════════════════════════════════════════════
  # 여러 episode의 초기 상태 생성 및 저장

  NUM_INITIAL_STATES = 10  # 저장할 초기 상태 개수 (원하는 만큼 조정 가능)

  print(f"Generating {NUM_INITIAL_STATES} different initial states...")
  initial_states = []
  rng_init = jax.random.PRNGKey(456)  # 다른 seed 사용

  for i in range(NUM_INITIAL_STATES):
      rng_init, reset_key = jax.random.split(rng_init)
      if _VISION.value:
          reset_key = jp.asarray(jax.random.split(reset_key, num_envs))

      # 새로운 초기 상태 생성
      state_init = jit_reset(reset_key)
      state_init = (
          jax.tree_util.tree_map(lambda x: x[0], state_init) 
          if _VISION.value else state_init
      )
      initial_states.append(state_init)
      print(f"  Generated initial state {i+1}/{NUM_INITIAL_STATES}")

  # 첫 번째 초기 상태 사용
  current_state_idx = 0
  current_initial_state = initial_states[current_state_idx]

  print(f"Using initial state {current_state_idx + 1}/{NUM_INITIAL_STATES}")


  # ═══════════════════════════════════════════════════════════════

  # 환경별 observation builder 선택
  obs_builder, needs_prev_action = get_obs_builder(_ENV_NAME.value)
  is_cartesian = "Cartesian" in _ENV_NAME.value
  
  print(f"Environment: {_ENV_NAME.value}")
  print(f"Is cartesian: {is_cartesian}")
  print(f"Using obs_builder: {obs_builder.__name__}")

  # ───  viewer setup ───────────────────────────────────────────
  model = eval_env.mj_model
  data  = mujoco.MjData(model)

  # === WARM-UP INFERENCE (중요!) ===
  # JIT 컴파일을 미리 수행하여 첫 실행 지연 방지
  print("Warming up JIT compilation...")
  warmup_start = time.perf_counter()

  def get_obs_dim(env_name):
     if "Cartesian" in env_name:
        return 70
     elif "Ball" in env_name:
        return 54
     else:
        return 66
  
  obs_dim = get_obs_dim(_ENV_NAME.value)

  # Dummy observation 생성 (실제 차원과 동일해야 함)
  dummy_obs = jp.zeros(obs_dim)  # 환경에 맞는 observation 차원
  dummy_rng = jax.random.PRNGKey(42)

  # JIT 컴파일 트리거 (첫 호출에서 컴파일됨)
  _ = jit_inference_fn(dummy_obs, dummy_rng)
  _ = jit_inference_fn(dummy_obs, dummy_rng)  # 두 번째 호출로 확실히 캐싱

  warmup_end = time.perf_counter()
  print(f"JIT warmup completed in {warmup_end - warmup_start:.3f}s")
  
  # === WARM-UP INFERENCE ===

  # sync viewer pose with env reset
  data.qpos[:] = np.asarray(state0.data.qpos)
  data.qvel[:] = np.asarray(state0.data.qvel)
  data.ctrl[:] = np.asarray(state0.data.ctrl)
  data.mocap_pos[:] = np.asarray(state0.data.mocap_pos).ravel()
  data.mocap_quat[:] = np.asarray(state0.data.mocap_quat).ravel()
  mujoco.mj_forward(model, data)

  ctrl_dt = env_cfg.ctrl_dt
  sim_dt = model.opt.timestep
  viewer_fps = 60
  init = True
  control_enabled = True  # 제어 활성화 플래그 추가
  episode_count = 0

  rng = jax.random.PRNGKey(0)
  prev_action = jp.zeros(3) if needs_prev_action else None  # 필요할 때만 초기화


  with mujoco.viewer.launch_passive(model, data,
                                    show_left_ui=False,
                                    show_right_ui=False) as viewer:
    sim_time = data.time
    last_view_time = sim_time

    while viewer.is_running():
      step_start = time.perf_counter()

      if control_enabled and (data.time - sim_time) >= ctrl_dt:
          
          obs = build_observation(data, TARGET_POS, obs_builder, prev_action)
          

          # Policy query
          rng, sub = jax.random.split(rng)
          action, _ = jit_inference_fn(obs, sub)

          if is_cartesian:
              # 환경의 step을 사용해서 올바른 control 계산
              simple_cartesian_controller(data, np.asarray(action), eval_env)
              
              # 이전 액션 업데이트
              if needs_prev_action:
                  prev_action = jp.asarray(action)

          else:
              # Joint space는 직접 적용
              delta = np.asarray(action) * ctrl_dt
              new_ctrl = np.clip(
                  data.ctrl + delta,
                  np.asarray(eval_env._lowers), 
                  np.asarray(eval_env._uppers)
              )
              data.ctrl[:] = new_ctrl
          sim_time = data.time
          
      # Reset 처리 # 새로운 초기 상태 사용
      if not init and data.time == 0.0:


          # ═══════════════════════════════════════════════════════════════
          # 여러 episode의 초기 상태 생성 및 저장 
          # 다음 초기 상태로 순환
          current_state_idx = (current_state_idx + 1) % NUM_INITIAL_STATES
          current_initial_state = initial_states[current_state_idx]
          episode_count += 1
          
          print(f"===== Reset detected (Episode {episode_count}) =====")
          print(f"Switching to initial state {current_state_idx + 1}/{NUM_INITIAL_STATES}")

          # Reset 중에는 제어 일시 중지
          control_enabled = False
          
          if needs_prev_action:
              prev_action = jp.zeros(3)
          
          # 새로운 초기 상태로 리셋
          data.qpos[:] = np.asarray(current_initial_state.data.qpos)
          data.qvel[:] = np.asarray(current_initial_state.data.qvel)
          data.ctrl[:] = np.asarray(current_initial_state.data.ctrl)
          data.mocap_pos[:] = np.asarray(current_initial_state.data.mocap_pos).ravel()
          data.mocap_quat[:] = np.asarray(current_initial_state.data.mocap_quat).ravel()
          
          # 새로운 target position 업데이트
          TARGET_POS = jp.asarray(current_initial_state.info["target_pos"])

          # ═══════════════════════════════════════════════════════════════
            
          ## 기존 : reset시 state유지

          # print("Reset detected, re-initializing...")
          # # Reset 중에는 제어 일시 중지
          # control_enabled = False

          # if needs_prev_action:
          #     prev_action = jp.zeros(3)
          
          # data.qpos[:] = np.asarray(state0.data.qpos)
          # data.qvel[:] = np.asarray(state0.data.qvel)
          # data.ctrl[:] = np.asarray(state0.data.ctrl)
          # data.mocap_pos[:] = np.asarray(state0.data.mocap_pos).ravel()
          # data.mocap_quat[:] = np.asarray(state0.data.mocap_quat).ravel()
          
          # ═══════════════════════════════════════════════════════════════  

          # Forward Dynamics 한 번 실행
          mujoco.mj_forward(model, data)

          # 시간 동기화
          sim_time = data.time
          last_view_time = sim_time
          
          # 제어 즉시 재활성화
          control_enabled = True
          print("Reset complete, control re-enabled!")          
          
          mujoco.mj_step(model, data)
          continue

      mujoco.mj_step(model, data)
      init = False

      # ── viewer refresh ───────────────────────────────────────
      if (data.time - last_view_time) >= 1.0 / viewer_fps:
              viewer.sync()
              last_view_time = data.time

      leftover = sim_dt - (time.perf_counter() - step_start)
      if leftover > 0:
          time.sleep(leftover)

if __name__ == "__main__":
  app.run(main)
