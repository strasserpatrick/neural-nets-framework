import json
import numpy as np

def save_training_result_as_json(path, confussion_matrix, lr, momentum, num_frames, window_size, epochs):
    result_dict = {"confussion_matrix"  : confussion_matrix.tolist(),
                   "learn_rate" : lr,
                   "momentum" : momentum,
                   "num_frames" : num_frames,
                   "window_size" : window_size,
                   "epochs" : epochs}
    with open(path, "w") as f:
        json_str = json.dumps(result_dict)
        f.write(json_str)