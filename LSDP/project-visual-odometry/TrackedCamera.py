import numpy as np
from custom_types import FrameID, CameraID
from icecream import ic


class TrackedCamera:
    def __init__(
        self,
        R,
        t,
        frame_id: FrameID,
        frame,
        camera_id: CameraID = CameraID(-1),
        fixed=False,
    ):
        assert R.shape == (3, 3)
        assert t.shape == (3, )
        self.R = R
        self.t = t
        self.frame_id: FrameID = frame_id
        self.camera_id: CameraID = camera_id
        self.frame = frame
        self.fixed = fixed

    def pose(self) -> np.ndarray:
        ret = np.eye(4)
        ret[:3, :3] = self.R
        ret[:3, 3] = self.t
        return ret

    def __repr__(self):
        inv_pose = np.linalg.inv(self.pose())
        return repr(
            "Camera %d [%s] %s (%f %f %f) %s (%f %f %f)"
            % (
                self.camera_id,
                self.frame_id,
                self.fixed,
                self.t[0],
                self.t[1],
                self.t[2],
                self.R,
                inv_pose[0, 3], 
                inv_pose[1, 3], 
                inv_pose[2, 3],
            )
        )
