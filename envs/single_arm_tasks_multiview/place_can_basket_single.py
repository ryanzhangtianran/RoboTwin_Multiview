from .._multiview_task import Base_Task as Base_Task_Multiview
from ..utils import *
import sapien
import math


class place_can_basket_single(Base_Task_Multiview):
    """
    Single-arm version of place_can_basket.
    One arm sequentially:
      1. Picks up the can
      2. Places the can into the basket
      3. Returns, then grasps the basket and lifts it
    """

    def setup_demo(self, is_test=False, arm_tag=None, **kwags):
        env_arm_tag = kwags.get("arm_tag", arm_tag)
        if env_arm_tag is None:
            env_arm_tag = "right"
        self.arm_tag = ArmTag(env_arm_tag)
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.basket_name = "110_basket"
        self.basket_id = [0, 1][np.random.randint(0, 2)]

        can_dict = {
            "071_can": [0, 1, 2, 3, 5, 6],
        }
        self.can_name = "071_can"
        self.can_id = can_dict[self.can_name][np.random.randint(0, len(can_dict[self.can_name]))]

        if self.arm_tag == "left":
            self.basket = rand_create_actor(
                scene=self,
                modelname=self.basket_name,
                model_id=self.basket_id,
                xlim=[0.02, 0.02],
                ylim=[0.25, 0.35],
                qpos=[0.5, 0.5, 0.5, 0.5],
                convex=True,
            )
            self.can = rand_create_actor(
                scene=self,
                modelname=self.can_name,
                model_id=self.can_id,
                xlim=[-0.25, -0.15],
                ylim=[0.15, 0.25],
                qpos=[0.707225, 0.706849, -0.0100455, -0.00982061],
                convex=True,
            )
        else:
            self.basket = rand_create_actor(
                scene=self,
                modelname=self.basket_name,
                model_id=self.basket_id,
                xlim=[-0.20, -0.05],
                ylim=[0.25, 0.35],
                qpos=[0.5, 0.5, 0.5, 0.5],
                convex=True,
            )
            self.can = rand_create_actor(
                scene=self,
                modelname=self.can_name,
                model_id=self.can_id,
                xlim=[0.25, 0.35],
                ylim=[0.15, 0.25],
                qpos=[0.707225, 0.706849, -0.0100455, -0.00982061],
                convex=True,
            )

        self.start_height = self.basket.get_pose().p[2]
        self.basket.set_mass(0.5)
        self.can.set_mass(0.01)
        self.add_prohibit_area(self.can, padding=0.1)
        self.add_prohibit_area(self.basket, padding=0.05)
        self.object_start_height = self.can.get_pose().p[2]

    def play_once(self):
        self.move(self.grasp_actor(self.can, arm_tag=self.arm_tag, pre_grasp_dis=0.05))

        # Choose the closer functional point on the basket as placement target
        place_pose = self.get_arm_pose(arm_tag=self.arm_tag)
        f0 = np.array(self.basket.get_functional_point(0))
        f1 = np.array(self.basket.get_functional_point(1))
        if np.linalg.norm(f0[:2] - place_pose[:2]) < np.linalg.norm(f1[:2] - place_pose[:2]):
            place_pose = f0
            place_pose[:2] = f0[:2]
            place_pose[3:] = ((-1, 0, 0, 0) if self.arm_tag == "left" else (0.05, 0, 0, 0.99))
        else:
            place_pose = f1
            place_pose[:2] = f1[:2]
            place_pose[3:] = ((-1, 0, 0, 0) if self.arm_tag == "left" else (0.05, 0, 0, 0.99))

        # Place the can into the basket
        self.move(
            self.place_actor(
                self.can,
                arm_tag=self.arm_tag,
                target_pose=place_pose,
                dis=0.02,
                is_open=False,
                constrain="free",
            ))

        # Fallback: if planning failed, try a higher approach
        if self.plan_success is False:
            self.plan_success = True
            place_pose[0] += -0.15 if self.arm_tag == "left" else 0.15
            place_pose[2] += 0.15
            self.move(self.move_to_pose(arm_tag=self.arm_tag, target_pose=place_pose))
            self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=-0.1))
            self.move(self.open_gripper(arm_tag=self.arm_tag))
        else:
            self.move(self.open_gripper(arm_tag=self.arm_tag))
            self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.12))

        # Step 3: same arm returns, then grasps and lifts the basket
        # self.move(self.back_to_origin(arm_tag=self.arm_tag))
        self.move(self.grasp_actor(self.basket, arm_tag=self.arm_tag, pre_grasp_dis=0.08))
        self.move(self.close_gripper(arm_tag=self.arm_tag))
        self.move(
            self.move_by_displacement(
                arm_tag=self.arm_tag,
                x=-0.02 if self.arm_tag == "left" else 0.02,
                z=0.05,
            )
        )

        self.info["info"] = {
            "{A}": f"{self.can_name}/base{self.can_id}",
            "{B}": f"{self.basket_name}/base{self.basket_id}",
            "{a}": str(self.arm_tag),
        }
        return self.info

    def check_success(self):
        can_p = self.can.get_pose().p
        basket_p = self.basket.get_pose().p
        basket_axis = (self.basket.get_pose().to_transformation_matrix()[:3, :3] @ np.array([[0, 1, 0]]).T)
        can_contact_table = not self.check_actors_contact("071_can", "table")
        can_contact_basket = self.check_actors_contact("071_can", "110_basket")
        return (
            basket_p[2] - self.start_height > 0.02
            and can_p[2] - self.object_start_height > 0.02
            and np.dot(basket_axis.reshape(3), [0, 0, 1]) > 0.5
            and np.sum(np.sqrt(np.power(can_p - basket_p, 2))) < 0.15
            and can_contact_table
            and can_contact_basket
        )
