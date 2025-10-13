import cv2
import torch
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from src.utils.crops import wide_crop_face, crop_face, emoca_crop
from src.utils.keypoint_detector import DECAKeypoints
import face_alignment

def preprocess_image(image, app, kpt_det, parsing_func):

    det = app.get(image)[0]
    face_wide = wide_crop_face(image, det['kps'])
    
    kpts = fa.get_landmarks_from_image(face_wide)
    
    if kpts is not None:
        kpts = np.vstack(kpts[0])
        if kpts.shape[0] == 68: #one face
            idx_68 = [0] #used to keep indices of good frames
            face_arc = crop_face(image, det['kps'])
            
            face_emoca = emoca_crop(face_wide[np.newaxis, ...], kpts[np.newaxis, ...])
            face_emoca = (face_emoca[0].permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
            parsing = parsing_func(face_wide[np.newaxis, ...].transpose(0, 3, 1, 2))

            return {
                'face_arc': face_arc,
                'face_emoca': face_emoca,
                'face_wide': face_wide,
                'keypoints_68': kpts,
                'idx_68': idx_68,
                'face_wide_parsing_segformer_B5_ce': parsing
            }
        else:
            print('Multiple faces')
    else:
        print('No face detected')




if __name__ == '__main__':
    app = FaceAnalysis(providers=['CUDAExecutionProvider'], allowed_modules=['detection'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cuda:0')

    providers = [
       ("CUDAExecutionProvider", {})
    ]
    parsings_session = ort.InferenceSession('./weights/segformer_B5_ce.onnx', providers=providers)
    input_name = parsings_session.get_inputs()[0].name
    output_names = [output.name for output in parsings_session.get_outputs()]
    
    mean = np.array([0.51315393, 0.48064056, 0.46301059])[None, :, None, None]
    std = np.array([0.21438347, 0.20799829, 0.20304542])[None, :, None, None]

    infer_parsing = lambda img: torch.tensor(
    parsings_session.run(output_names, {
        input_name: ((img[:, [2, 1, 0], ...] / 255. - mean) / std).astype(np.float32)
    })[0],
    device='cuda',
    dtype=torch.float32
)

    image = cv2.imread('./examples/images/hab.jpg')

    res_dict = preprocess_image(image, app, fa, infer_parsing)