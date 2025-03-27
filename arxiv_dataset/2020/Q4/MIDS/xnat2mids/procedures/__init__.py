
from abc import ABC, abstractmethod


class Procedures(ABC):
    def __int__(self):
        self.view_positions_2d = ['pa', 'll', 'lateral', 'ap', None]
        self.view_positions_3d = ['ax', 'sag', 'cor', None]

    @abstractmethod
    def reset_indexes(self):
        pass

    @staticmethod
    def get_mids_path(mids_path, department_id, subject_id, session_id):
        return mids_path.pathjoin(department_id, subject_id, session_id)

    @staticmethod
    def get_name(
            subject_id,
            session_id,
            acq_index,
            run_index,
            len_nifti_files,
            body_part,
            view_position,
            scan,
            ext
    ):
        return "".join([
            subject_id,
            (("_" + session_id) if session_id else ""),
            ("_acq-" + str(acq_index) if len_nifti_files > 1 else ""),
            "_run-" + run_index,
            (("_bp-" + body_part) if body_part != "head" else "")
            (("_vp-" + view_position) if view_position and body_part != "head" else ""),
            "_",
            scan,
            ext
        ])
