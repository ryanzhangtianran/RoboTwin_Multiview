from .._multiview_task import Base_Task as Base_Task_Multiview
from ..utils import *
import sapien
import numpy as np
from copy import deepcopy


class stack_blocks_two_single(Base_Task_Multiview):
    """
    Single-arm: move red block {A} to center, then stack green block {B} on top of {A}.
    """

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        block_half_size = 0.025
        block_pose_lst = []
        for i in range(2):
            block_pose = rand_pose(
                xlim=[-0.28, 0.28],
                ylim=[-0.08, 0.05],
                zlim=[0.741 + block_half_size],
                qpos=[1, 0, 0, 0],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0, 0, 0.75],
            )

            def check_block_pose(block_pose):
                for j in range(len(block_pose_lst)):
                    if (np.sum(pow(block_pose.p[:2] - block_pose_lst[j].p[:2], 2)) < 0.01):
                        return False
                return True

            while (abs(block_pose.p[0]) < 0.05 or np.sum(pow(block_pose.p[:2] - np.array([0, -0.1]), 2)) < 0.0225
                   or not check_block_pose(block_pose)):
                block_pose = rand_pose(
                    xlim=[-0.28, 0.28],
                    ylim=[-0.08, 0.05],
                    zlim=[0.741 + block_half_size],
                    qpos=[1, 0, 0, 0],
                    ylim_prop=True,
                    rotate_rand=True,
                    rotate_lim=[0, 0, 0.75],
                )
            block_pose_lst.append(deepcopy(block_pose))

        def create_block(block_pose, color):
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=(block_half_size, block_half_size, block_half_size),
                color=color,
                name="box",
            )

        self.block1 = create_block(block_pose_lst[0], (1, 0, 0))
        self.block2 = create_block(block_pose_lst[1], (0, 1, 0))
        self.add_prohibit_area(self.block1, padding=0.07)
        self.add_prohibit_area(self.block2, padding=0.07)
        self.prohibited_area.append([-0.04, -0.13, 0.04, -0.05])

    def play_once(self):
        # Single arm: choose arm by first block (red) position
        block1_pose = self.block1.get_pose().p
        arm_tag = ArmTag("left" if block1_pose[0] < 0 else "right")
        self.arm_tag = arm_tag

        # 1. Move red block {A} to center
        self.move(self.grasp_actor(self.block1, arm_tag=arm_tag, pre_grasp_dis=0.09))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))
        target_pose = [0, -0.13, 0.75 + self.table_z_bias, 0, 1, 0, 0]
        self.move(
            self.place_actor(
                self.block1,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.05,
                dis=0.0,
                pre_dis_axis="fp",
            ))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        # 2. Stack green block {B} on top of red block {A}
        self.move(self.grasp_actor(self.block2, arm_tag=arm_tag, pre_grasp_dis=0.09))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))
        target_pose = self.block1.get_functional_point(1)
        self.move(
            self.place_actor(
                self.block2,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.05,
                dis=0.0,
                pre_dis_axis="fp",
            ))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        self.info["info"] = {
            "{A}": "red block",
            "{B}": "green block",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        block1_pose = self.block1.get_pose().p
        block2_pose = self.block2.get_pose().p
        eps = np.array([0.025, 0.025, 0.012])
        stacked = np.all(np.abs(block2_pose - np.array(block1_pose[:2].tolist() + [block1_pose[2] + 0.05])) < eps)
        gripper_open = (self.robot.is_left_gripper_open() if self.arm_tag == "left"
                       else self.robot.is_right_gripper_open())
        return stacked and gripper_open
