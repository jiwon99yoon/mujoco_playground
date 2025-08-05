# # Copyright 2025 DeepMind Technologies Limited
# # ... (라이센스 헤더)

# """Load trained PPO model and perform real-time inference with MuJoCo viewer."""

# import functools
# import json
# import time
# import threading
# import queue
# from typing import Optional, Dict, Any

# from absl import app
# from absl import flags
# from brax.training.agents.ppo import networks as ppo_networks
# from etils import epath
# import jax
# import jax.numpy as jp
# from ml_collections import config_dict
# import mujoco
# import numpy as np
# from orbax import checkpoint as ocp

# import mujoco_playground
# from mujoco_playground import registry

# # 플래그 정의
# _CHECKPOINT_PATH = flags.DEFINE_string(
#     "checkpoint_path", None, "Path to checkpoint directory", required=True
# )
# _REALTIME_MODE = flags.DEFINE_boolean(
#     "realtime_mode", False, "Enable real-time inference mode"
# )
# _FPS = flags.DEFINE_integer(
#     "fps", 30, "Target FPS for visualization"
# )
# _USE_ROS = flags.DEFINE_boolean(
#     "use_ros", False, "Use ROS for state input (requires ROS installation)"
# )

# class PPOInferencePlayer:
#     """실시간 PPO 추론 및 MuJoCo 시각화 클래스"""
    
#     def __init__(self, checkpoint_path: str):
#         self.checkpoint_path = epath.Path(checkpoint_path)
#         self.config = self._load_config()
#         self.env = self._load_environment()
#         self.inference_fn = self._load_model()
#         self.current_state = None
#         self.rng = jax.random.PRNGKey(0)
        
#         # ROS 또는 실시간 모드용 큐
#         self.state_queue = queue.Queue(maxsize=10)
#         self.action_queue = queue.Queue(maxsize=10)
        
#     def _load_config(self) -> Dict[str, Any]:
#         """체크포인트에서 설정 로드"""
#         config_path = self.checkpoint_path / "config.json"
#         with open(config_path, 'r') as f:
#             return json.load(f)
    
#     def _load_environment(self):
#         """환경 로드"""
#         env_name = self.config["env_name"]
#         env_config = self.config["env_config"]
        
#         # ConfigDict로 변환
#         env_cfg = config_dict.ConfigDict(env_config)
        
#         print(f"Loading environment: {env_name}")
#         return registry.load(env_name, config=env_cfg)
    
#     def _load_model(self):
#         """학습된 모델 로드"""
#         print(f"Loading checkpoint from: {self.checkpoint_path}")
        
#         # 최신 체크포인트 찾기
#         checkpoints = sorted(
#             [d for d in self.checkpoint_path.glob("*") if d.is_dir() and d.name.isdigit()],
#             key=lambda x: int(x.name)
#         )
        
#         if not checkpoints:
#             raise ValueError(f"No checkpoints found in {self.checkpoint_path}")
        
#         latest_checkpoint = checkpoints[-1]
#         print(f"Using checkpoint: {latest_checkpoint}")
        
#         # Orbax로 체크포인트 로드
#         ckpt_manager = ocp.CheckpointManager(self.checkpoint_path)
#         step = int(latest_checkpoint.name)
#         restored = ckpt_manager.restore(step)
        
#         params = restored['params']
        
#         # 네트워크 재구성
#         ppo_config = self.config["ppo_params"]
#         network_factory = functools.partial(
#             ppo_networks.make_ppo_networks,
#             policy_hidden_layer_sizes=tuple(ppo_config["policy_hidden_layer_sizes"]),
#             value_hidden_layer_sizes=tuple(ppo_config["value_hidden_layer_sizes"]),
#             policy_obs_key=ppo_config["policy_obs_key"],
#             value_obs_key=ppo_config["value_obs_key"],
#         )
        
#         ppo_network = network_factory(
#             self.env.observation_size,
#             self.env.action_size,
#             preprocess_observations_fn=None
#         )
        
#         # 추론 함수 생성
#         def inference_fn(obs, rng):
#             logits, _ = ppo_network.policy_network.apply(
#                 params['policy'], obs
#             )
#             action = jax.nn.tanh(logits)  # 결정적 행동
#             return action, {}
        
#         return jax.jit(inference_fn)
    
#     def get_current_mujoco_state(self, data: mujoco.MjData) -> Dict:
#         """현재 MuJoCo 시뮬레이터의 상태를 환경 observation 형식으로 변환"""
#         # 환경별로 observation 구성이 다르므로 적절히 변환
#         # 예시: PandaPickCubeCartesian의 경우
#         obs_dict = {
#             'state': jp.concatenate([
#                 data.qpos[:7],  # 로봇 관절 위치
#                 data.qvel[:7],  # 로봇 관절 속도
#                 data.qpos[7:10],  # 물체 위치
#                 # 필요한 다른 정보들...
#             ])
#         }
        
#         # 환경에 따라 다른 observation 구조 처리
#         if hasattr(self.env, 'get_obs_from_mjdata'):
#             obs_dict = self.env.get_obs_from_mjdata(data)
        
#         return obs_dict
    
#     def run_interactive(self):
#         """인터랙티브 MuJoCo viewer 실행"""
#         model = self.env.mj_model
#         data = mujoco.MjData(model)
        
#         # 초기 상태 설정
#         self.current_state = self.env.reset(self.rng)
#         data.qpos[:] = self.current_state.data.qpos
#         data.qvel[:] = self.current_state.data.qvel
#         mujoco.mj_forward(model, data)
        
#         with mujoco.viewer.launch_passive(model, data) as viewer:
#             print("\n=== PPO Inference Player ===")
#             print("Controls:")
#             print("- Space: Toggle real-time inference")
#             print("- R: Reset environment")
#             print("- Q: Quit")
#             print("===========================\n")
            
#             inference_active = True
#             last_time = time.time()
            
#             while viewer.is_running():
#                 current_time = time.time()
#                 dt = current_time - last_time
                
#                 if inference_active and dt >= 1.0 / _FPS.value:
#                     # 현재 MuJoCo 상태에서 observation 생성
#                     obs = self.get_current_mujoco_state(data)
                    
#                     # PPO 추론
#                     self.rng, act_rng = jax.random.split(self.rng)
#                     action, _ = self.inference_fn(obs, act_rng)
                    
#                     # Action 적용
#                     if hasattr(self.env, 'apply_action_to_mjdata'):
#                         self.env.apply_action_to_mjdata(data, action)
#                     else:
#                         # 기본 동작: action을 ctrl로 직접 사용
#                         data.ctrl[:] = jp.clip(action, -1, 1)
                    
#                     # 물리 시뮬레이션 스텝
#                     mujoco.mj_step(model, data)
                    
#                     # 상태 정보 출력
#                     if int(current_time * 10) % 10 == 0:  # 1초마다
#                         print(f"Action: {action[:3]}... | qpos: {data.qpos[:3]}...", end='\r')
                    
#                     last_time = current_time
                
#                 viewer.sync()
    
#     def run_with_ros(self):
#         """ROS 통신을 통한 실시간 추론 (구현 예시)"""
#         try:
#             import rospy
#             from std_msgs.msg import Float64MultiArray
#             from sensor_msgs.msg import JointState
#         except ImportError:
#             print("ROS not installed. Please install ROS or use --use_ros=False")
#             return
        
#         print("Initializing ROS node...")
#         rospy.init_node('ppo_inference_node')
        
#         # ROS 콜백 함수
#         def state_callback(msg):
#             # JointState 메시지를 observation으로 변환
#             obs = {
#                 'state': jp.array(msg.position + msg.velocity)
#             }
            
#             # 추론 실행
#             rng, act_rng = jax.random.split(self.rng, 2)
#             action, _ = self.inference_fn(obs, act_rng)
            
#             # Action 발행
#             action_msg = Float64MultiArray()
#             action_msg.data = action.tolist()
#             action_pub.publish(action_msg)
            
#             print(f"Received state, published action: {action[:3]}...", end='\r')
        
#         # ROS 구독자/발행자 설정
#         state_sub = rospy.Subscriber('/robot/joint_states', JointState, state_callback)
#         action_pub = rospy.Publisher('/robot/action_command', Float64MultiArray, queue_size=1)
        
#         print("ROS node running. Waiting for state messages...")
#         rospy.spin()

# def main(argv):
#     """메인 실행 함수"""
#     del argv
    
#     # PPO 추론 플레이어 생성
#     player = PPOInferencePlayer(_CHECKPOINT_PATH.value)
    
#     if _USE_ROS.value:
#         # ROS 모드
#         player.run_with_ros()
#     else:
#         # 인터랙티브 MuJoCo viewer 모드
#         player.run_interactive()

# if __name__ == "__main__":
#     app.run(main)


# # Copyright 2025 DeepMind Technologies Limited
# # ... (라이센스 헤더)

# """Load trained PPO model and perform real-time inference with MuJoCo viewer."""

# import functools
# import json
# import time
# import threading
# import queue
# from typing import Optional, Dict, Any
# import os

# from absl import app
# from absl import flags
# from brax.training.agents.ppo import networks as ppo_networks
# from etils import epath
# import jax
# import jax.numpy as jp
# from ml_collections import config_dict
# import mujoco
# import numpy as np
# from orbax import checkpoint as ocp

# import mujoco_playground
# from mujoco_playground import registry
# from mujoco_playground.config import dm_control_suite_params
# from mujoco_playground.config import locomotion_params
# from mujoco_playground.config import manipulation_params

# # 플래그 정의
# _CHECKPOINT_PATH = flags.DEFINE_string(
#     "checkpoint_path", None, "Path to checkpoint directory", required=True
# )
# _ENV_NAME = flags.DEFINE_string(
#     "env_name", None, "Environment name (auto-detected if not specified)"
# )
# _REALTIME_MODE = flags.DEFINE_boolean(
#     "realtime_mode", False, "Enable real-time inference mode"
# )
# _FPS = flags.DEFINE_integer(
#     "fps", 30, "Target FPS for visualization"
# )
# _USE_ROS = flags.DEFINE_boolean(
#     "use_ros", False, "Use ROS for state input (requires ROS installation)"
# )

# class PPOInferencePlayer:
#     """실시간 PPO 추론 및 MuJoCo 시각화 클래스"""
    
#     def __init__(self, checkpoint_path: str, env_name: Optional[str] = None):
#         self.checkpoint_path = epath.Path(checkpoint_path)
#         self.env_name = env_name or self._detect_env_name()
#         self.config = self._load_config()
#         self.env = self._load_environment()
#         self.inference_fn, self.make_inference_fn = self._load_model()
#         self.current_state = None
#         self.rng = jax.random.PRNGKey(0)
        
#         # ROS 또는 실시간 모드용 큐
#         self.state_queue = queue.Queue(maxsize=10)
#         self.action_queue = queue.Queue(maxsize=10)
    
#     def _detect_env_name(self) -> str:
#         """체크포인트 경로에서 환경 이름 추출"""
#         # 경로에서 환경 이름 추측 (예: PandaPickCube-20250805-181903)
#         parent_name = self.checkpoint_path.parent.name
        
#         # 타임스탬프 패턴 제거
#         import re
#         env_name = re.sub(r'-\d{8}-\d{6}.*$', '', parent_name)
        
#         # 환경 이름 확인
#         all_envs = registry.ALL_ENVS
#         if env_name not in all_envs:
#             # 유사한 이름 찾기
#             for env in all_envs:
#                 if env.lower() in parent_name.lower():
#                     env_name = env
#                     break
        
#         print(f"Detected environment: {env_name}")
#         return env_name
        
#     def _load_config(self) -> Dict[str, Any]:
#         """체크포인트에서 설정 로드"""
#         config_path = self.checkpoint_path / "config.json"
        
#         if config_path.exists():
#             with open(config_path, 'r') as f:
#                 saved_config = json.load(f)
#         else:
#             print(f"Warning: config.json not found at {config_path}")
#             saved_config = {}
        
#         # 기본 설정 생성
#         config = {
#             "env_name": self.env_name,
#             "env_config": saved_config,  # 기존 config.json 내용
#         }
        
#         return config
    
#     def _get_rl_config(self, env_name: str):
#         """환경에 맞는 기본 PPO 설정 가져오기"""
#         if env_name in mujoco_playground.manipulation._envs:
#             return manipulation_params.brax_ppo_config(env_name)
#         elif env_name in mujoco_playground.locomotion._envs:
#             return locomotion_params.brax_ppo_config(env_name)
#         elif env_name in mujoco_playground.dm_control_suite._envs:
#             return dm_control_suite_params.brax_ppo_config(env_name)
#         else:
#             raise ValueError(f"Unknown environment: {env_name}")
    
#     def _load_environment(self):
#         """환경 로드"""
#         env_config = self.config.get("env_config", {})
        
#         # 기본 설정 가져오기
#         default_config = registry.get_default_config(self.env_name)
        
#         # 저장된 설정으로 업데이트
#         for key, value in env_config.items():
#             if hasattr(default_config, key):
#                 setattr(default_config, key, value)
        
#         print(f"Loading environment: {self.env_name}")
#         return registry.load(self.env_name, config=default_config)
    
#     def _load_model(self):
#         """학습된 모델 로드"""
#         print(f"Loading checkpoint from: {self.checkpoint_path}")
        
#         # 최신 체크포인트 찾기
#         checkpoints = sorted(
#             [d for d in self.checkpoint_path.glob("*") if d.is_dir() and d.name.isdigit()],
#             key=lambda x: int(x.name)
#         )
        
#         if not checkpoints:
#             raise ValueError(f"No checkpoints found in {self.checkpoint_path}")
        
#         latest_checkpoint = checkpoints[-1]
#         print(f"Using checkpoint: {latest_checkpoint}")
        
#         # 기본 PPO 설정 가져오기
#         rl_config = self._get_rl_config(self.env_name)
        
#         # train_fn에서 make_inference_fn 가져오기 위한 설정
#         from brax.training.agents.ppo import train as ppo
        
#         network_fn = ppo_networks.make_ppo_networks
#         network_factory = functools.partial(
#             network_fn,
#             policy_hidden_layer_sizes=rl_config.network_factory.policy_hidden_layer_sizes,
#             value_hidden_layer_sizes=rl_config.network_factory.value_hidden_layer_sizes,
#             policy_obs_key=rl_config.network_factory.policy_obs_key,
#             value_obs_key=rl_config.network_factory.value_obs_key,
#         )
        
#         # 간단한 train_fn 호출로 make_inference_fn 가져오기
#         train_fn = functools.partial(
#             ppo.train,
#             num_timesteps=0,  # 학습하지 않음
#             episode_length=1000,
#             num_envs=128,
#             normalize_observations=rl_config.normalize_observations,
#             network_factory=network_factory,
#             restore_checkpoint_path=latest_checkpoint,
#         )
        
#         # make_inference_fn과 params 가져오기
#         make_inference_fn, params, _ = train_fn(environment=self.env)
        
#         # 결정적 추론 함수 생성
#         inference_fn = make_inference_fn(params, deterministic=True)
        
#         return jax.jit(inference_fn), make_inference_fn
    
#     def get_current_mujoco_state(self, data: mujoco.MjData):
#         """현재 MuJoCo 시뮬레이터의 상태를 환경 observation 형식으로 변환"""
#         # 환경의 _get_obs 메서드 활용
#         if hasattr(self.env, '_get_obs'):
#             # 임시 state 객체 생성
#             from types import SimpleNamespace
#             temp_state = SimpleNamespace()
#             temp_state.qpos = data.qpos
#             temp_state.qvel = data.qvel
#             temp_state.time = data.time
            
#             # 환경의 observation 생성 메서드 호출
#             obs = self.env._get_obs(temp_state)
#             return obs
#         else:
#             # 기본 observation 구조
#             return {
#                 'state': jp.concatenate([
#                     data.qpos,
#                     data.qvel,
#                 ])
#             }
    
#     def run_interactive(self):
#         """인터랙티브 MuJoCo viewer 실행"""
#         model = self.env.mj_model
#         data = mujoco.MjData(model)
        
#         # 초기 상태 설정
#         self.current_state = self.env.reset(self.rng)
#         data.qpos[:] = self.current_state.data.qpos
#         data.qvel[:] = self.current_state.data.qvel
#         mujoco.mj_forward(model, data)
        
#         # 환경 설정 가져오기
#         env_cfg = registry.get_default_config(self.env_name)
#         ctrl_dt = env_cfg.ctrl_dt if hasattr(env_cfg, 'ctrl_dt') else 0.05
#         action_scale = env_cfg.action_scale if hasattr(env_cfg, 'action_scale') else 1.0
        
#         with mujoco.viewer.launch_passive(model, data) as viewer:
#             print("\n=== PPO Inference Player ===")
#             print(f"Environment: {self.env_name}")
#             print(f"Control dt: {ctrl_dt}")
#             print(f"Action scale: {action_scale}")
#             print("\nControls:")
#             print("- Space: Pause/Resume")
#             print("- R: Reset environment")
#             print("- ESC: Quit")
#             print("===========================\n")
            
#             inference_active = True
#             sim_time = data.time
#             last_time = time.time()
            
#             while viewer.is_running():
#                 current_time = time.time()
                
#                 if inference_active and (data.time - sim_time) >= ctrl_dt:
#                     # 현재 상태를 observation으로 변환
#                     obs = self.get_current_mujoco_state(data)
                    
#                     # PPO 추론
#                     self.rng, act_rng = jax.random.split(self.rng)
#                     action, _ = self.inference_fn(obs, act_rng)
                    
#                     # Action 적용 (환경별 처리)
#                     if self.env_name == "PandaPickCubeCartesian":
#                         # Cartesian 속도 명령
#                         vel_cmd = np.asarray(action)
#                         delta = vel_cmd * action_scale
#                         # 환경의 step 함수를 통해 적절한 변환 수행
#                         # 여기서는 간단히 처리
#                         data.ctrl[:3] = delta  # 예시
#                     else:
#                         # 일반적인 경우
#                         data.ctrl[:] = np.asarray(action)
                    
#                     sim_time = data.time
                    
#                     # 상태 정보 출력
#                     if int(data.time * 10) % 10 == 0:  # 1초마다
#                         print(f"Time: {data.time:.2f} | Action: {action[:3]} | qpos: {data.qpos[:3]}", end='\r')
                
#                 # 물리 시뮬레이션 스텝
#                 mujoco.mj_step(model, data)
                
#                 # 화면 갱신
#                 if (current_time - last_time) >= 1.0 / _FPS.value:
#                     viewer.sync()
#                     last_time = current_time
                
#                 # 프레임 레이트 제어
#                 time.sleep(0.001)

# def main(argv):
#     """메인 실행 함수"""
#     del argv
    
#     # PPO 추론 플레이어 생성
#     player = PPOInferencePlayer(
#         _CHECKPOINT_PATH.value,
#         env_name=_ENV_NAME.value
#     )
    
#     if _USE_ROS.value:
#         # ROS 모드
#         player.run_with_ros()
#     else:
#         # 인터랙티브 MuJoCo viewer 모드
#         player.run_interactive()

# if __name__ == "__main__":
#     app.run(main)

"""Load trained PPO model and perform real-time inference with MuJoCo viewer."""

import functools
import json
import time
from typing import Optional, Dict, Any
import os

from absl import app
from absl import flags
from brax.training.agents.ppo import networks as ppo_networks
from etils import epath
import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer 
import numpy as np

import mujoco_playground
from mujoco_playground import registry
from mujoco_playground.config import manipulation_params

# 환경 설정
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

# 플래그 정의
_CHECKPOINT_PATH = flags.DEFINE_string("checkpoint_path", None, "Path to checkpoint directory", required=True)
_ENV_NAME = flags.DEFINE_string("env_name", None, "Environment name")
_FPS = flags.DEFINE_integer("fps", 30, "Target FPS for visualization")

class PPOInferencePlayer:
    def __init__(self, checkpoint_path: str, env_name: Optional[str] = None):
        self.checkpoint_path = epath.Path(checkpoint_path)
        self.env_name = env_name or self._detect_env_name()
        self.config = self._load_config()
        self.env = self._load_environment()
        self.inference_fn = self._load_model_from_brax_checkpoint()
        self.rng = jax.random.PRNGKey(0)
        
    def _detect_env_name(self) -> str:
        """체크포인트 경로에서 환경 이름 추출"""
        parent_name = self.checkpoint_path.parent.name
        import re
        env_name = re.sub(r'-\d{8}-\d{6}.*$', '', parent_name)
        print(f"Detected environment: {env_name}")
        return env_name
        
    def _load_config(self) -> Dict[str, Any]:
        """config.json 로드"""
        config_path = self.checkpoint_path / "config.json"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
                
            # 새로운 형식 (train_jax_ppo_and_store.py)
            if "env_name" in saved_config:
                return saved_config
            # 기존 형식 (env_cfg만 저장된 경우)
            else:
                return {
                    "env_name": self.env_name,
                    "env_config": saved_config,
                    "ppo_params": self._get_default_ppo_params()
                }
        else:
            print(f"Warning: config.json not found")
            return {
                "env_name": self.env_name,
                "env_config": {},
                "ppo_params": self._get_default_ppo_params()
            }
    
    def _get_default_ppo_params(self):
        """기본 PPO 파라미터"""
        rl_config = manipulation_params.brax_ppo_config(self.env_name)
        return {
            "policy_hidden_layer_sizes": list(rl_config.network_factory.policy_hidden_layer_sizes),
            "value_hidden_layer_sizes": list(rl_config.network_factory.value_hidden_layer_sizes),
            "policy_obs_key": "state",
            "value_obs_key": "state",
            "normalize_observations": True,
        }
    
    def _load_environment(self):
        """환경 로드"""
        env_config = self.config.get("env_config", {})
        default_config = registry.get_default_config(self.env_name)
        
        for key, value in env_config.items():
            if hasattr(default_config, key):
                setattr(default_config, key, value)
        
        print(f"Loading environment: {self.env_name}")
        return registry.load(self.env_name, config=default_config)
    
    def _load_model_from_brax_checkpoint(self):
        """Brax/Orbax 체크포인트에서 모델 로드"""
        print(f"Loading checkpoint from: {self.checkpoint_path}")
        
        # 최신 체크포인트 찾기
        checkpoints = sorted(
            [d for d in self.checkpoint_path.glob("*") if d.is_dir() and d.name.isdigit()],
            key=lambda x: int(x.name)
        )
        
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {self.checkpoint_path}")
        
        latest_checkpoint = checkpoints[-1]
        print(f"Using checkpoint: {latest_checkpoint}")
        
        # Brax의 방식으로 체크포인트 로드
        from brax.training.agents.ppo import train as ppo
        from brax.training import acting
        
        # PPO 네트워크 설정
        ppo_params = self.config.get("ppo_params", self._get_default_ppo_params())
        
        def identity_preprocessor(obs, _processor_params=None):
            return obs
        
        network_fn = ppo_networks.make_ppo_networks
        ppo_network = network_fn(
            observation_size=self.env.observation_size,
            action_size=self.env.action_size,
            preprocess_observations_fn=identity_preprocessor,
            policy_hidden_layer_sizes=tuple(ppo_params["policy_hidden_layer_sizes"]),
            value_hidden_layer_sizes=tuple(ppo_params["value_hidden_layer_sizes"]),
        )
        
        # # Orbax로 직접 로드
        # from orbax import checkpoint as ocp
        # ckpt_manager = ocp.CheckpointManager(
        #     self.checkpoint_path,
        #     checkpointers=ocp.PyTreeCheckpointer(),
        # )
                
        # from orbax.checkpoint import PyTreeCheckpointer, PytreeRestoreArgs

        # # 체크포인트 디렉토리 (zero-padded 폴더) 경로
        # ckpt_dir = os.path.join(self.checkpoint_path, padded)
        # checkpointer = PyTreeCheckpointer()
        # # args 는 비워두면 default pytree restore args 가 적용됩니다.
        # restored = checkpointer.restore(ckpt_dir, PytreeRestoreArgs())
        
        
        # # 체크포인트 복원
        # step = int(latest_checkpoint.name)
        # padded = str(step).zfill(12)
        # restored = ckpt_manager.restore(padded)
        import os
        from orbax.checkpoint import PyTreeCheckpointer
        from orbax.checkpoint._src.handlers.pytree_checkpoint_handler import PyTreeRestoreArgs

        # 1) 체크포인트 step & zero-pad 유지
        step = int(latest_checkpoint.name)
        padded = str(step).zfill(12)

        # 2) 디렉토리 경로 구성

        # 2) 상대경로 결합 → 절대경로 변환
        relative_path = os.path.join(self.checkpoint_path, padded)
        ckpt_dir = os.path.abspath(relative_path)
        
        # 3) non-composite pytree 복원
        checkpointer = PyTreeCheckpointer()
        restored = checkpointer.restore(ckpt_dir, PyTreeRestoreArgs())        

        # params 추출
        if isinstance(restored, dict) and 'params' in restored:
            params = restored['params']
        else:
            params = restored
        
        # 추론 함수 생성
        def inference_fn(obs, rng):
            # normalizer 처리
            if 'normalizer_params' in restored:
                norm_params = restored['normalizer_params']
                if hasattr(norm_params, 'mean') and hasattr(norm_params, 'std'):
                    obs = (obs - norm_params.mean) / (norm_params.std + 1e-8)
            
            # policy 적용
            if isinstance(params, dict) and 'policy' in params:
                policy_params = params['policy']
            else:
                policy_params = params
                
            logits = ppo_network.policy_network.apply(policy_params, rng, obs)
            action = jp.tanh(logits.loc)  # deterministic
            return action, {}
        
        return jax.jit(inference_fn)
    
    def get_obs_from_mjdata(self, data: mujoco.MjData):
        """MuJoCo 데이터를 observation으로 변환"""
        if self.env_name == "PandaPickCube":
            # PandaPickCube의 observation 구조
            robot_qpos = data.qpos[:8]
            robot_qvel = data.qvel[:8]
            
            # 나머지 필요한 정보들
            obs = jp.concatenate([robot_qpos, robot_qvel])
            
            # 환경의 observation 크기에 맞게 패딩
            if len(obs) < self.env.observation_size:
                obs = jp.pad(obs, (0, self.env.observation_size - len(obs)))
                
            return obs
        else:
            return jp.concatenate([data.qpos, data.qvel])
    
    def run_interactive(self):
        """인터랙티브 MuJoCo viewer 실행"""
        model = self.env.mj_model
        data = mujoco.MjData(model)
        
        # 초기 상태
        self.rng, reset_rng = jax.random.split(self.rng)
        state = self.env.reset(reset_rng)
        data.qpos[:] = state.data.qpos
        data.qvel[:] = state.data.qvel
        mujoco.mj_forward(model, data)
        
        # 환경 설정
        ctrl_dt = self.config['env_config'].get('ctrl_dt', 0.02)
        action_scale = self.config['env_config'].get('action_scale', 0.04)
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print(f"\nPPO Inference - {self.env_name}")
            print(f"Control dt: {ctrl_dt}, Action scale: {action_scale}")
            print("Press ESC to exit\n")
            
            sim_time = data.time
            episode_step = 0
            
            while viewer.is_running():
                step_start = time.perf_counter()
                
                if (data.time - sim_time) >= ctrl_dt:
                    # Observation 생성
                    obs = self.get_obs_from_mjdata(data)
                    
                    # PPO 추론
                    self.rng, act_rng = jax.random.split(self.rng)
                    action, _ = self.inference_fn(obs, act_rng)
                    
                    # Action 적용
                    action_np = np.asarray(action)
                    scaled_action = action_np * action_scale
                    
                    # 제어 입력 설정
                    if len(scaled_action) <= len(data.ctrl):
                        data.ctrl[:len(scaled_action)] = data.qpos[:len(scaled_action)] + scaled_action
                    
                    sim_time = data.time
                    episode_step += 1
                    
                    if episode_step % 50 == 0:
                        print(f"Step: {episode_step}, Action: {action_np[:3]}", end='\r')
                
                # 물리 시뮬레이션
                mujoco.mj_step(model, data)
                
                # 화면 갱신
                if (time.perf_counter() - step_start) < model.opt.timestep:
                    time.sleep(model.opt.timestep - (time.perf_counter() - step_start))
                    
                viewer.sync()

def main(argv):
    del argv
    
    player = PPOInferencePlayer(
        _CHECKPOINT_PATH.value,
        env_name=_ENV_NAME.value
    )
    
    player.run_interactive()

if __name__ == "__main__":
    app.run(main)