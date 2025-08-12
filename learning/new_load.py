# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
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

  # ─── 환경 타입 감지 및 통합 함수들 ───────────────────────────────────────────────
  def is_cartesian_env(eval_env):
    """환경 이름으로 Cartesian 환경 여부 판단"""
    env_class_name = eval_env.__class__.__name__
    return "Cartesian" in env_class_name

  def get_unified_obs(data, target_pos, prev_action=None):
    """환경의 내장 _get_obs() 메서드를 활용한 통합 observation"""
    info = {"target_pos": target_pos}
    
    # 환경의 기본 obs 계산 (환경 내부 로직 활용)
    base_obs = eval_env._get_obs(data, info)
    
    # Cartesian 환경인 경우 추가 정보 append
    if is_cartesian:
      no_soln = jp.array([0.0])  # viewer에서는 항상 IK 성공으로 가정
      prev_action = prev_action if prev_action is not None else jp.zeros(3)
      return jp.concatenate([base_obs, no_soln, prev_action])
    
    return base_obs

  def apply_action_unified(data, action, state_info=None):
    """환경별 action 적용 로직"""
    
    if is_cartesian:
      # Cartesian 환경: 환경의 _move_tip 메서드 활용
      if hasattr(eval_env, '_move_tip'):
        current_pos = state_info.get('current_pos', jp.array([0.5, 0.0, 0.2]))
        current_rot = eval_env._start_tip_transform[:3, :3]
        
        # Cartesian 환경의 action format: [0, y, z, gripper]
        increment = jp.zeros(4).at[1:].set(action)
        
        ctrl, new_pos, no_soln = eval_env._move_tip(
          current_pos, current_rot, jp.asarray(data.ctrl), increment
        )
        
        # Control limits 적용
        data.ctrl[:] = np.clip(np.asarray(ctrl), 
                              np.asarray(eval_env._lowers), 
                              np.asarray(eval_env._uppers))
        
        return {'current_pos': new_pos, 'no_soln': no_soln}
      
      else:
        print("Warning: Cartesian environment detected but _move_tip not found")
        return {}
    
    else:
      # Joint space 환경: 기본 delta control
      delta = np.asarray(action) * ctrl_dt
      new_ctrl = np.clip(
        data.ctrl + delta,
        np.asarray(eval_env._lowers), 
        np.asarray(eval_env._uppers)
      )
      data.ctrl[:] = new_ctrl
      return {}

  # ─── 환경 타입 감지 및 설정 ─────────────────────────────────────────────────────
  is_cartesian = is_cartesian_env(eval_env)
  print(f"Environment: {eval_env.__class__.__name__}")
  print(f"Environment type: {'Cartesian' if is_cartesian else 'Joint Space'}")
  print(f"Action size: {eval_env.action_size}")

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
  TARGET_POS = jp.asarray(current_initial_state.info["target_pos"])

  print(f"Using initial state {current_state_idx + 1}/{NUM_INITIAL_STATES}")
  # ═══════════════════════════════════════════════════════════════

  # ─── viewer setup ─────────────────────────────────────────────
  model = eval_env.mj_model
  data = mujoco.MjData(model)

  # === WARM-UP INFERENCE (중요!) ===
  # JIT 컴파일을 미리 수행하여 첫 실행 지연 방지
  print("Warming up JIT compilation...")
  warmup_start = time.perf_counter()

  # 환경에 맞는 observation 차원으로 더미 데이터 생성
  dummy_obs = get_unified_obs(data, TARGET_POS, jp.zeros(3) if is_cartesian else None)
  dummy_rng = jax.random.PRNGKey(42)

  # JIT 컴파일 트리거 (첫 호출에서 컴파일됨)
  _ = jit_inference_fn(dummy_obs, dummy_rng)
  _ = jit_inference_fn(dummy_obs, dummy_rng)  # 두 번째 호출로 확실히 캐싱

  warmup_end = time.perf_counter()
  print(f"JIT warmup completed in {warmup_end - warmup_start:.3f}s")
  print(f"Observation dimension: {dummy_obs.shape}")
  # === WARM-UP INFERENCE ===

  # sync viewer pose with env reset
  data.qpos[:] = np.asarray(current_initial_state.data.qpos)
  data.qvel[:] = np.asarray(current_initial_state.data.qvel)
  data.ctrl[:] = np.asarray(current_initial_state.data.ctrl)
  data.mocap_pos[:] = np.asarray(current_initial_state.data.mocap_pos).ravel()
  data.mocap_quat[:] = np.asarray(current_initial_state.data.mocap_quat).ravel()
  mujoco.mj_forward(model, data)

  ctrl_dt = env_cfg.ctrl_dt
  sim_dt = model.opt.timestep
  viewer_fps = 60
  init = True
  control_enabled = True  # 제어 활성화 플래그 추가
  episode_count = 0

  rng = jax.random.PRNGKey(0)
  
  # Cartesian 환경용 상태 추적
  state_info = {}
  prev_action = None
  if is_cartesian:
    state_info = {'current_pos': jp.array([0.5, 0.0, 0.2])}
    prev_action = jp.zeros(3)

  with mujoco.viewer.launch_passive(model, data,
                                    show_left_ui=False,
                                    show_right_ui=False) as viewer:
    sim_time = data.time
    last_view_time = sim_time

    while viewer.is_running():
      step_start = time.perf_counter()

      if control_enabled and (data.time - sim_time) >= ctrl_dt:
          
        # 1. 통합된 observation 생성
        obs = get_unified_obs(data, TARGET_POS, prev_action)

        # 2. Policy query
        rng, sub = jax.random.split(rng)
        action, _ = jit_inference_fn(obs, sub)

        # 3. 통합된 action 적용
        result = apply_action_unified(data, action, state_info)
        state_info.update(result)
        
        # 4. Cartesian 환경의 이전 action 업데이트
        if is_cartesian:
          prev_action = jp.asarray(action)

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
        
        # 상태 리셋
        if is_cartesian:
          prev_action = jp.zeros(3)
          state_info = {'current_pos': jp.array([0.5, 0.0, 0.2])}
        
        # 새로운 초기 상태로 리셋
        data.qpos[:] = np.asarray(current_initial_state.data.qpos)
        data.qvel[:] = np.asarray(current_initial_state.data.qvel)
        data.ctrl[:] = np.asarray(current_initial_state.data.ctrl)
        data.mocap_pos[:] = np.asarray(current_initial_state.data.mocap_pos).ravel()
        data.mocap_quat[:] = np.asarray(current_initial_state.data.mocap_quat).ravel()
        
        # 새로운 target position 업데이트
        TARGET_POS = jp.asarray(current_initial_state.info["target_pos"])

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
