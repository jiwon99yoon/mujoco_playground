# /home/dyros/mujoco_playground/mujoco_playground/_src/manipulation/franka_emika_panda/pickrollingball.py
# Copyright 2025 DeepMind Technologies Limited
# PandaPickRollingBall - 경사면에서 굴러오는 공을 잡는 환경

"""Pick a rolling ball from a ramp and bring to target."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import collision
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.franka_emika_panda import panda
from mujoco_playground._src.mjx_env import State


def default_config() -> config_dict.ConfigDict:
  """Returns the default config for rolling ball tasks."""
  config = config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=200,  # 더 긴 에피소드 (공이 굴러오는 시간 고려)
      action_repeat=1,
      action_scale=0.04,
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Gripper가 공으로 이동
              gripper_ball=4.0,
              # 공을 목표 위치로 이동
              ball_target=8.0,
              # reward추가위해
              gripper_target=1.0,
              catch_bonus=5.0,
              carry_bonus=3.0,
              progress_reward=2.0,
              # ramp와 충돌 방지
              no_ramp_collision = 0.25,
              # 바닥과 충돌 방지
              no_floor_collision=0.25,
              # 로봇 팔이 목표 자세 유지
              robot_target_qpos=0.3,
              # 움직이는 공을 성공적으로 잡았을 때 보너스
              catch_moving_ball=2.0,
          )
      ),
  )
  return config


class PandaPickRollingBall(panda.PandaBase):
  """Pick a rolling ball from a ramp and bring to target."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      sample_orientation: bool = False,
  ):
    xml_path = (
        mjx_env.ROOT_PATH
        / "manipulation"
        / "franka_emika_panda"
        / "xmls"
        / "mjx_single_moving_ball_with_ramp.xml"  # 새로운 XML 파일
    )
    super().__init__(
        xml_path,
        config,
        config_overrides,
    )
    self._post_init(obj_name="ball", keyframe="home")
    self._sample_orientation = sample_orientation
    # ramp_surface와의 충돌 방지를 위해 geom.id 불러오기  
    self._ramp_geom = self._mj_model.geom("ramp_surface").id

  def reset(self, rng: jax.Array) -> State:
    rng, rng_ball_x, rng_ball_y, rng_target = jax.random.split(rng, 4)

    # 30도 경사면에 맞춘 공의 시작 위치 설정 (2배 길이)
    # 경사면 정보:
    # - 중심: (0.9, 0, 0.25)  
    # - 크기: 0.7 x 0.2 x 0.01 (길이가 2배)
    # - 30도 기울기 (tan(30°) = 0.577)
    
    # 경사면의 실제 범위 계산:
    # X축: 0.9 - 0.7*cos(30°) ~ 0.9 + 0.7*cos(30°) 
    #     = 0.9 - 0.606 ~ 0.9 + 0.606
    #     = 0.294 ~ 1.506
    
    # X: 경사면 상단 부근 (상단 1/3 지점에서 시작)
    ball_x = jax.random.uniform(rng_ball_x, minval=1.15, maxval=1.35)
    
    # Y: 경사면 폭 내에서 (벽 사이, 안전 마진 포함)
    ball_y = jax.random.uniform(rng_ball_y, minval=-0.15, maxval=0.15)
    
    # Z: 30도 경사면 상의 높이 계산 + 추가 높이
    ramp_center_x = 0.9
    ramp_center_z = 0.25
    
    # 경사면 상의 높이 계산 (30도 경사)
    # Z = center_z + (x - center_x) * tan(30°)
    ramp_height = ramp_center_z + (ball_x - ramp_center_x) * 0.2679
    
    # 공을 경사면 위 5~10cm 높이에서 떨어뜨림 (시각적으로 명확하게)
    drop_height = jax.random.uniform(rng, minval=0.3, maxval=0.5)
    ball_z = ramp_height + 0.025 + drop_height  # 공 반경(0.025) + 드롭 높이
    
    ball_pos = jp.array([ball_x, ball_y, ball_z])

    # 목표 위치 설정 (로봇 작업 공간 내)
    target_pos = (
        jax.random.uniform(
            rng_target,
            (3,),
            minval=jp.array([-0.2, -0.2, 0.2]),
            maxval=jp.array([0.2, 0.2, 0.4]),
        )
        + jp.array([0.5, 0, 0.5])  # 로봇 작업 공간 중심
    )

    target_quat = jp.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    if self._sample_orientation:
        rng, rng_axis, rng_theta = jax.random.split(rng, 3)
        perturb_axis = jax.random.uniform(rng_axis, (3,), minval=-1, maxval=1)
        perturb_axis = perturb_axis / math.norm(perturb_axis)
        perturb_theta = jax.random.uniform(rng_theta, maxval=np.deg2rad(45))
        target_quat = math.axis_angle_to_quat(perturb_axis, perturb_theta)

    # 초기 상태 설정
    init_q = (
        jp.array(self._init_q)
        .at[self._obj_qposadr : self._obj_qposadr + 3]
        .set(ball_pos)
    )
    
    # 공의 초기 속도 (자유낙하 시작이므로 매우 작게 또는 0)
    init_qvel = jp.zeros(self._mjx_model.nv, dtype=float)
    ball_vel_idx = self._mj_model.body(self._obj_body).dofadr
    
    # 매우 작은 초기 속도만 부여 (자연스러운 시작을 위해)
    init_qvel = init_qvel.at[ball_vel_idx].set(-0.01)  # 거의 정지 상태에서 시작
    
    data = mjx_env.init(
        self._mjx_model,
        init_q,
        init_qvel,
        ctrl=self._init_ctrl,
    )

    # 목표 mocap 위치 설정
    data = data.replace(
        mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos),
        mocap_quat=data.mocap_quat.at[self._mocap_target, :].set(target_quat),
    )

    # 메트릭 및 정보 초기화
    metrics = {
        "out_of_bounds": jp.array(0.0, dtype=float),
        **{k: 0.0 for k in self._config.reward_config.scales.keys()},
    }
    info = {
        "rng": rng, 
        "target_pos": target_pos, 
        "reached_ball": 0.0,
        "ball_was_moving": 1.0,
        "caught_moving": 0.0,
    }
    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    state = State(data, obs, reward, done, metrics, info)
    return state

  def step(self, state: State, action: jax.Array) -> State:
    delta = action * self._action_scale
    ctrl = state.data.ctrl + delta
    ctrl = jp.clip(ctrl, self._lowers, self._uppers)

    data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

    raw_rewards = self.get_reward(data, state.info)
    rewards = {
        k: v * self._config.reward_config.scales[k]
        for k, v in raw_rewards.items()
    }
    reward = jp.clip(sum(rewards.values()), -1e4, 1e4)
    
    ball_pos = data.xpos[self._obj_body]
    out_of_bounds = jp.any(jp.abs(ball_pos) > 1.5)  # 더 큰 경계
    out_of_bounds |= ball_pos[2] < 0.0
    done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)

    state.metrics.update(
        **raw_rewards, out_of_bounds=out_of_bounds.astype(float)
    )

    # 공의 속도 추적
    # ball_vel = data.qvel[self._mj_model.body(self._obj_body).dofadr : 
    #                      self._mj_model.body(self._obj_body).dofadr + 3]
    # 수정된 코드
    dof_addr = int(self._mj_model.body(self._obj_body).dofadr)
    ball_vel = data.qvel[dof_addr:dof_addr + 3]
    ball_speed = jp.linalg.norm(ball_vel)
    
    # 정보 업데이트
    new_info = dict(state.info)
    new_info["ball_was_moving"] = (ball_speed > 0.01).astype(float)
    
    obs = self._get_obs(data, new_info)
    state = State(data, obs, reward, done, state.metrics, new_info)

    return state

  def get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, Any]:
    target_pos = info["target_pos"]
    ball_pos = data.xpos[self._obj_body]
    gripper_pos = data.site_xpos[self._gripper_site]
    
    # 공의 속도 계산
    dof_addr = int(self._mj_model.body(self._obj_body).dofadr)
    ball_vel = data.qvel[dof_addr:dof_addr + 3]
    ball_speed = jp.linalg.norm(ball_vel)
    
    # 거리 계산
    pos_err = jp.linalg.norm(target_pos - ball_pos)
    gripper_ball_dist = jp.linalg.norm(ball_pos - gripper_pos)
    gripper_target_dist = jp.linalg.norm(target_pos - gripper_pos)
    
    # 1. 그리퍼가 공에 접근하기 (항상 활성화)
    gripper_ball = 1 - jp.tanh(5 * gripper_ball_dist)
    
    # 2. 공이 목표 위치에 접근하기 (항상 활성화)
    ball_target = 1 - jp.tanh(3 * pos_err)
    
    # 3. 그리퍼가 목표 위치에 접근하기 (보조 reward)
    gripper_target = 1 - jp.tanh(2 * gripper_target_dist)
    
    # 4. 공을 성공적으로 잡았을 때의 보너스
    ball_caught = (gripper_ball_dist < 0.015).astype(float)
    catch_bonus = ball_caught * 2.0
    
    # 5. 공을 잡고 목표로 이동시킬 때의 추가 보너스
    carry_bonus = ball_caught * ball_target * 1.5
    
    # 6. 진행 상황 기반 reward (JAX 호환)
    # 거리가 가까울수록 높은 가중치를 주는 연속 함수
    proximity_weight = 1 - jp.tanh(20 * gripper_ball_dist)  # 0.05 근처에서 급변
    progress_reward = proximity_weight * ball_target * 0.5
    
    # 바닥 충돌 체크
    hand_floor_collision = [
        collision.geoms_colliding(data, self._floor_geom, g)
        for g in [
            self._left_finger_geom,
            self._right_finger_geom,
            self._hand_geom,
        ]
    ]
    floor_collision = sum(hand_floor_collision) > 0
    no_floor_collision = (1 - floor_collision).astype(float)
    
    # 경사면과 로봇 충돌 체크
    hand_ramp_collision = [
        collision.geoms_colliding(data, self._ramp_geom, g)
        for g in [
          self._left_finger_geom,
          self._right_finger_geom,
          self._hand_geom,
        ]
    ]
    ramp_collision = sum(hand_ramp_collision) > 0
    no_ramp_collision = (1 - ramp_collision).astype(float)

    # 로봇 자세 유지
    robot_target_qpos = 1 - jp.tanh(
        jp.linalg.norm(
            data.qpos[self._robot_arm_qposadr]
            - self._init_q[self._robot_arm_qposadr]
        )
    )
    
    # 움직이는 공 관련 정보 업데이트
    info["reached_ball"] = jp.maximum(
        info["reached_ball"],
        ball_caught,
    )
    
    info["caught_moving"] = jp.maximum(
        info["caught_moving"],
        ball_caught * info["ball_was_moving"]
    )
    
    # 움직이는 공을 잡는 추가 보너스
    catch_moving_ball = info["caught_moving"] * 3.0
    
    rewards = {
        "gripper_ball": gripper_ball,
        "ball_target": ball_target,               # 항상 활성화!
        "gripper_target": gripper_target,
        "catch_bonus": catch_bonus,
        "carry_bonus": carry_bonus,
        "progress_reward": progress_reward,
        "no_floor_collision": no_floor_collision,
        "no_ramp_collision" : no_ramp_collision,    # ramp와의 충돌방지를 위한 reward
        "robot_target_qpos": robot_target_qpos,
        "catch_moving_ball": catch_moving_ball,
    }
    
    return rewards

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    gripper_pos = data.site_xpos[self._gripper_site]
    gripper_mat = data.site_xmat[self._gripper_site].ravel()
    
    # 공의 속도 정보 추가
    # ball_vel = data.qvel[self._mj_model.body(self._obj_body).dofadr : 
    #                      self._mj_model.body(self._obj_body).dofadr + 3]

    # 수정된 코드 - int()로 명시적 변환
    dof_addr = int(self._mj_model.body(self._obj_body).dofadr)
    ball_vel = data.qvel[dof_addr:dof_addr + 3]    
    obs = jp.concatenate([
        data.qpos,
        data.qvel,
        gripper_pos,
        gripper_mat[3:],
        data.xpos[self._obj_body] - gripper_pos,  # 공과 그리퍼의 상대 위치
        ball_vel,  # 공의 속도 정보 추가
        info["target_pos"] - data.xpos[self._obj_body],
        data.ctrl - data.qpos[self._robot_qposadr[:-1]],
    ])

    return obs