from .._multiview_task import Base_Task as Base_Task_Multiview
from ..utils import *
import sapien
import numpy as np


class put_bottles_dustbin_single(Base_Task_Multiview):
    """
    Single-arm: grab the three bottles one by one and put them into the dustbin on the ground behind the robot arm (using right arm).
    """

    def setup_demo(self, **kwags):
        super()._init_task_env_(table_xy_bias=[0.3, 0], **kwags)

    def load_actors(self):
        pose_lst = []

        def create_bottle(model_id):
            bottle_pose = rand_pose(
                xlim=[-0.25, 0.3],
                ylim=[0.03, 0.23],
                rotate_rand=False,
                rotate_lim=[0, 1, 0],
                qpos=[0.707, 0.707, 0, 0],
            )
            tag = True
            gen_lim = 100
            i = 1
            while tag and i < gen_lim:
                tag = False
                if np.abs(bottle_pose.p[0]) < 0.05:
                    tag = True
                for pose in pose_lst:
                    # 瓶子中心间距至少 0.22 m，抓第一个时手臂不易碰倒另一个
                    if (np.sum(np.power(np.array(pose[:2]) - np.array(bottle_pose.p[:2]), 2)) < 0.0484):
                        tag = True
                        break
                if tag:
                    i += 1
                    bottle_pose = rand_pose(
                        xlim=[-0.25, 0.3],
                        ylim=[0.03, 0.23],
                        rotate_rand=False,
                        rotate_lim=[0, 1, 0],
                        qpos=[0.707, 0.707, 0, 0],
                    )
            pose_lst.append(bottle_pose.p[:2])
            bottle = create_actor(
                scene=self,
                pose=bottle_pose,
                modelname="114_bottle",
                convex=True,
                model_id=model_id,
            )
            return bottle

        self.bottles = []
        self.bottle_id = [1, 2, 3]
        self.bottle_num = 3
        for i in range(self.bottle_num):
            bottle = create_bottle(self.bottle_id[i])
            self.bottles.append(bottle)
            self.add_prohibit_area(bottle, padding=0.1)

        # Dustbin on the ground, behind the robot arm (same as dual-arm style)
        self.dustbin_xy = np.array([-0.20, -0.7])
        self.dustbin = create_actor(
            scene=self.scene,
            pose=sapien.Pose([0, -1.10, 0], [0.5, 0.5, 0.5, 0.5]),
            modelname="011_dustbin",
            convex=True,
            is_static=True,
        )
        self.delay(2)
        # Pose above dustbin to drop bottle (x, y, z, qw, qx, qy, qz)
        self.drop_pose = [0.10, -0.90, 1.2, 0.65, -0.25, 0.25, 0.65]
        # 高抬后先移到该过渡点（y 已离开桌面），再移到 drop，避免平移时碰倒其他瓶子
        self.transit_pose = [0.10, -0.35, 0.95, 0.65, -0.25, 0.25, 0.65]
        self.above_quat = [0.65, -0.25, 0.25, 0.65]

    def play_once(self):
        # 先拿 y 最小的（最靠近机械臂），再拿靠里的，抓第一个时不易扫到另一个
        bottle_lst = sorted(self.bottles, key=lambda x: (x.get_pose().p[1], x.get_pose().p[0]))

        arm_tag = ArmTag("right")
        self.arm_tag = arm_tag

        for i in range(self.bottle_num):
            bottle = bottle_lst[i]
            bx, by = bottle.get_pose().p[0], bottle.get_pose().p[1]
            # 先到「瓶子上方 x、机械臂侧 y、高 z」的安全点，再沿 y 移到瓶正上方，再竖直下压抓取，避免扫到其他瓶子
            staging_pose = [bx, -0.28, 0.95] + self.above_quat
            above_pose = [bx, by, 0.92] + self.above_quat
            self.move(self.move_to_pose(arm_tag=arm_tag, target_pose=staging_pose))
            self.move(self.move_to_pose(arm_tag=arm_tag, target_pose=above_pose))
            self.move(self.grasp_actor(bottle, arm_tag=arm_tag, pre_grasp_dis=0.08))
            # 先高抬，确保超出桌上其他瓶子再平移，避免碰倒
            self.move(self.move_by_displacement(arm_tag, z=0.25))
            # 先到过渡点（高且 y 已离开桌面），再到垃圾桶上方
            self.move(self.move_to_pose(arm_tag=arm_tag, target_pose=self.transit_pose))
            self.move(self.move_to_pose(arm_tag=arm_tag, target_pose=self.drop_pose))
            # Open right gripper
            self.move(self.open_gripper(arm_tag))

        self.info["info"] = {
            "{A}": f"114_bottle/base{self.bottle_id[0]}",
            "{B}": f"114_bottle/base{self.bottle_id[1]}",
            "{C}": f"114_bottle/base{self.bottle_id[2]}",
            "{D}": f"011_dustbin/base0",
            "{a}": str(arm_tag),
        }
        return self.info

    def stage_reward(self):
        taget_pose = self.dustbin_xy
        eps = np.array([0.221, 0.325])
        reward = 0
        reward_step = 1 / 3
        for i in range(self.bottle_num):
            bottle_pose = self.bottles[i].get_pose().p
            if (np.all(np.abs(bottle_pose[:2] - taget_pose) < eps) and bottle_pose[2] > 0.1 and bottle_pose[2] < 0.6):
                reward += reward_step
        return reward

    def check_success(self):
        taget_pose = self.dustbin_xy
        eps = np.array([0.221, 0.325])
        for i in range(self.bottle_num):
            bottle_pose = self.bottles[i].get_pose().p
            if (np.all(np.abs(bottle_pose[:2] - taget_pose) < eps)
                    and bottle_pose[2] > 0.1 and bottle_pose[2] < 0.6):
                continue
            return False
        return True
