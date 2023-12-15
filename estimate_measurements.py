

import argparse
import numpy as np
import pickle

from landmark_utils import *
from measurement_utils import *

def load_model(sex):
    model_path = f"models/{sex}.pkl"
    model = pickle.load(open(model_path, "rb"))
    return model


def estimate_measurements(model,landmarks):
    
    results = []
    for meas in MEASUREMENTS_ORDER:
        pred_meas = model[meas].predict(landmarks)
        results.append(pred_meas)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-L","--landmarks_path",
                        required=True,
                        type=str, 
                        default="demo/demo_landmarks.json",
                        help="Path to landmarks to use.")
    parser.add_argument("-S","--sex", 
                        required=True,
                        type=str, 
                        default="male",
                        choices=["male","female"],
                        help="Sex of the subject.")
    parser.add_argument("--scale", 
                        required=False,
                        type=int, 
                        default=1,
                        help="Scale the landmarks into mm if necessary. \
                              Multiply scale with coordinates")
    parser.add_argument("--normalize_viewpoint",
                        action="store_true",
                        help="Rotate the landmarks to have the same viewpoint \
                              as the training data.")
    args = parser.parse_args()


    landmarks = load_landmarks(args.landmarks_path)
    landmarks = process_landmarks(landmarks, args.scale, args.normalize_viewpoint)
    model = load_model(args.sex)
    predicted_measurements = estimate_measurements(model, landmarks)


    print("Measurement Estimation:")
    for i, m in enumerate(MEASUREMENTS_ORDER):
        print(f"{m:45} {predicted_measurements[i].round(2).item():>7.2f}")

