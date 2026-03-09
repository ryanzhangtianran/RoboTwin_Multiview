from .._multiview_task import Base_Task as Base_Task_Multiview
from ..utils import *
import sapien
import numpy as np
from copy import deepcopy


class stack_bowls_three_single(Base_Task_Multiview):
    """
    Single-arm: stack the three bowls on top of each other (one arm does all).
    """

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        bowl_pose_lst = []
        for i in range(3):
            bowl_pose = rand_pose(
                xlim=[-0.3, 0.3],
                ylim=[-0.15, 0.15],
                qpos=[0.5, 0.5, 0.5, 0.5],
                ylim_prop=True,
                rotate_rand=False,
            )

            def check_bowl_pose(bowl_pose):
                for j in range(len(bowl_pose_lst)):
                    if (np.sum(pow(bowl_pose.p[:2] - bowl_pose_lst[j].p[:2], 2)) < 0.0169):
                        return False
                return True

            while (abs(bowl_pose.p[0]) < 0.09 or np.sum(pow(bowl_pose.p[:2] - np.array([0, -0.1]), 2)) < 0.0169
                   or not check_bowl_pose(bowl_pose)):
                bowl_pose = rand_pose(
                    xlim=[-0.3, 0.3],
                    ylim=[-0.15, 0.15],
                    qpos=[0.5, 0.5, 0.5, 0.5],
                    ylim_prop=True,
                    rotate_rand=False,
                )
            bowl_pose_lst.append(deepcopy(bowl_pose))

        bowl_pose_lst = sorted(bowl_pose_lst, key=lambda x: x.p[1])

        def create_bowl(bowl_pose):
            return create_actor(scene=self, pose=bowl_pose, modelname="002_bowl", model_id=3, convex=True)

        self.bowl1 = create_bowl(bowl_pose_lst[0])
        self.bowl2 = create_bowl(bowl_pose_lst[1])
        self.bowl3 = create_bowl(bowl_pose_lst[2])

        self.add_prohibit_area(self.bowl1, padding=0.07)
        self.add_prohibit_area(self.bowl2, padding=0.07)
        self.add_prohibit_area(self.bowl3, padding=0.07)
        self.prohibited_area.append([-0.1, -0.15, 0.1, -0.05])
        self.bowl1_target_pose = np.array([0, -0.1, 0.76])
        self.quat_of_target_pose = [0, 0.707, 0.707, 0]

    def play_once(self):
        # Single arm: choose by first bowl position
        bowl1_pose = self.bowl1.get_pose().p
        arm_tag = ArmTag("left" if bowl1_pose[0] < 0 else "right")
        self.arm_tag = arm_tag

        # 1. Move bowl1 to center base
        self.move(self.grasp_actor(
            self.bowl1,
            arm_tag=arm_tag,
            contact_point_id=[0, 2][int(arm_tag == "left")],
            pre_grasp_dis=0.1,
        ))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1))
        self.move(self.place_actor(
            self.bowl1,
            target_pose=self.bowl1_target_pose.tolist() + self.quat_of_target_pose,
            arm_tag=arm_tag,
            functional_point_id=0,
            pre_dis=0.09,
            dis=0,
            constrain="align",
        ))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.09))

        # 2. Stack bowl2 on bowl1
        self.move(self.grasp_actor(
            self.bowl2,
            arm_tag=arm_tag,
            contact_point_id=[0, 2][int(arm_tag == "left")],
            pre_grasp_dis=0.1,
        ))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1))
        target2 = self.bowl1.get_pose().p + np.array([0, 0, 0.05])
        self.move(self.place_actor(
            self.bowl2,
            target_pose=target2.tolist() + self.quat_of_target_pose,
            arm_tag=arm_tag,
            functional_point_id=0,
            pre_dis=0.09,
            dis=0,
            constrain="align",
        ))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.09))

        # 3. Stack bowl3 on bowl2
        self.move(self.grasp_actor(
            self.bowl3,
            arm_tag=arm_tag,
            contact_point_id=[0, 2][int(arm_tag == "left")],
            pre_grasp_dis=0.1,
        ))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1))
        target3 = self.bowl2.get_pose().p + np.array([0, 0, 0.05])
        self.move(self.place_actor(
            self.bowl3,
            target_pose=target3.tolist() + self.quat_of_target_pose,
            arm_tag=arm_tag,
            functional_point_id=0,
            pre_dis=0.09,
            dis=0,
            constrain="align",
        ))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.09))

        self.info["info"] = {
            "{A}": "002_bowl/base3",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        bowl1_pose = self.bowl1.get_pose().p
        bowl2_pose = self.bowl2.get_pose().p
        bowl3_pose = self.bowl3.get_pose().p
        bowl1_pose, bowl2_pose, bowl3_pose = sorted([bowl1_pose, bowl2_pose, bowl3_pose], key=lambda x: x[2])
        # 放置时使用 0.76, 0.81, 0.86 附近，用对称容差避免 (actual - target) < eps 对 0.76 vs 0.74 误判
        target_height = np.array([
            0.74 + self.table_z_bias,
            0.77 + self.table_z_bias,
            0.81 + self.table_z_bias,
        ])
        actual_height = np.array([bowl1_pose[2], bowl2_pose[2], bowl3_pose[2]])
        eps_height = 0.04
        eps2 = 0.04
        gripper_open = (self.robot.is_left_gripper_open() if self.arm_tag == "left"
                       else self.robot.is_right_gripper_open())
        return (np.all(np.abs(bowl1_pose[:2] - bowl2_pose[:2]) < eps2)
                and np.all(np.abs(bowl2_pose[:2] - bowl3_pose[:2]) < eps2)
                and np.all(np.abs(actual_height - target_height) < eps_height)
                and gripper_open)
